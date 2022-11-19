
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger # Comment out if not using wandb
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging
import torch
from torch import nn as nn, Tensor
import os
import os.path as osp
import pandas as pd
import numpy as np
from positional_encodings.torch_encodings import PositionalEncodingPermute2D, Summer
import timm
from matplotlib import numpy
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import precision
import torchmetrics.functional as tf
import torchvision.transforms as transforms
from randaugment import RandAugment
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler

import torchvision
from torch import nn

import torch
import sys

import json
import random
import torch.multiprocessing as mp
import datetime

# Taken from this repo: https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/main/src/loss_functions/partial_asymmetric_loss.py

class PartialSelectiveLoss(nn.Module):

    def __init__(self, clip, gamma_pos, gamma_neg, gamma_unann, alpha_pos, alpha_neg, alpha_unann, prior_path):
        super(PartialSelectiveLoss, self).__init__()
        
        self.clip = clip
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.gamma_unann = gamma_unann
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.alpha_unann = alpha_unann
        self.prior_path = prior_path

        self.targets_weights = None

        if prior_path is not None:
            print("Prior file was found in given path.")
            df = pd.read_csv(prior_path)
            self.prior_classes = dict(zip(df.values[:, 0], df.values[:, 1]))
            print("Prior file was loaded successfully. ")

    def forward(self, logits, targets):

        # Positive, Negative and Un-annotated indexes
        targets_pos = (targets == 1).float()
        targets_neg = (targets == 0).float()
        targets_unann = (targets == -1).float()

        # Activation
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        prior_classes = None
        if hasattr(self, "prior_classes"):
            prior_classes = torch.tensor(list(self.prior_classes.values())).cuda()

        targets_weights = self.targets_weights
        targets_weights, xs_neg = edit_targets_parital_labels(self.args, targets, targets_weights, xs_neg,
                                                              prior_classes=prior_classes)

        # Loss calculation
        BCE_pos = self.alpha_pos * targets_pos * torch.log(torch.clamp(xs_pos, min=1e-8))
        BCE_neg = self.alpha_neg * targets_neg * torch.log(torch.clamp(xs_neg, min=1e-8))
        BCE_unann = self.alpha_unann * targets_unann * torch.log(torch.clamp(xs_neg, min=1e-8))

        BCE_loss = BCE_pos + BCE_neg + BCE_unann

        # Adding asymmetric gamma weights
        with torch.no_grad():
            asymmetric_w = torch.pow(1 - xs_pos * targets_pos - xs_neg * (targets_neg + targets_unann),
                                     self.gamma_pos * targets_pos + self.gamma_neg * targets_neg +
                                     self.gamma_unann * targets_unann)
        BCE_loss *= asymmetric_w

        # partial labels weights
        BCE_loss *= targets_weights

        return -BCE_loss.sum()


def edit_targets_parital_labels(args, targets, targets_weights, xs_neg, prior_classes=None):
    # targets_weights is and internal state of AsymmetricLoss class. we don't want to re-allocate it every batch
    if args.partial_loss_mode is None:
        targets_weights = 1.0

    elif args.partial_loss_mode == 'negative':
        # set all unsure targets as negative
        targets_weights = 1.0

    elif args.partial_loss_mode == 'ignore':
        # remove all unsure targets (targets_weights=0)
        targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        targets_weights[targets == -1] = 0

    elif args.partial_loss_mode == 'ignore_normalize_classes':
        # remove all unsure targets and normalize by Durand et al. https://arxiv.org/pdf/1902.09720.pdfs
        alpha_norm, beta_norm = 1, 1
        targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        n_annotated = 1 + torch.sum(targets != -1, axis=1)    # Add 1 to avoid dividing by zero

        g_norm = alpha_norm * (1 / n_annotated) + beta_norm
        n_classes = targets_weights.shape[1]
        targets_weights *= g_norm.repeat([n_classes, 1]).T
        targets_weights[targets == -1] = 0

    elif args.partial_loss_mode == 'selective':
        if targets_weights is None or targets_weights.shape != targets.shape:
            targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        else:
            targets_weights[:] = 1.0
        num_top_k = args.likelihood_topk * targets_weights.shape[0]

        xs_neg_prob = xs_neg
        if prior_classes is not None:
            if args.prior_threshold:
                idx_ignore = torch.where(prior_classes > args.prior_threshold)[0]
                targets_weights[:, idx_ignore] = 0
                targets_weights += (targets != -1).float()
                targets_weights = targets_weights.bool()

        negative_backprop_fun_jit(targets, xs_neg_prob, targets_weights, num_top_k)

        # set all unsure targets as negative
        # targets[targets == -1] = 0

    return targets_weights, xs_neg


# @torch.jit.script
def negative_backprop_fun_jit(targets: Tensor, xs_neg_prob: Tensor, targets_weights: Tensor, num_top_k: int):
    with torch.no_grad():
        targets_flatten = targets.flatten()
        cond_flatten = torch.where(targets_flatten == -1)[0]
        targets_weights_flatten = targets_weights.flatten()
        xs_neg_prob_flatten = xs_neg_prob.flatten()
        ind_class_sort = torch.argsort(xs_neg_prob_flatten[cond_flatten])
        targets_weights_flatten[
            cond_flatten[ind_class_sort[:num_top_k]]] = 0


class ComputePrior:
    def __init__(self, classes):
        self.classes = classes
        n_classes = len(self.classes)
        self.sum_pred_train = torch.zeros(n_classes).cuda()
        self.sum_pred_val = torch.zeros(n_classes).cuda()
        self.cnt_samples_train,  self.cnt_samples_val = .0, .0
        self.avg_pred_train, self.avg_pred_val = None, None
        self.path_dest = "./outputs"
        self.path_local = "/class_prior/"

    def update(self, logits, training=True):
        with torch.no_grad():
            preds = torch.sigmoid(logits).detach()
            if training:
                self.sum_pred_train += torch.sum(preds, axis=0)
                self.cnt_samples_train += preds.shape[0]
                self.avg_pred_train = self.sum_pred_train / self.cnt_samples_train

            else:
                self.sum_pred_val += torch.sum(preds, axis=0)
                self.cnt_samples_val += preds.shape[0]
                self.avg_pred_val = self.sum_pred_val / self.cnt_samples_val

    def save_prior(self):

        print('Prior (train), first 5 classes: {}'.format(self.avg_pred_train[:5]))

        # Save data frames as csv files
        if not os.path.exists(self.path_dest):
            os.makedirs(self.path_dest)

        df_train = pd.DataFrame({"Classes": list(self.classes.values()),
                                 "avg_pred": self.avg_pred_train.cpu()})
        df_train.to_csv(path_or_buf=os.path.join(self.path_dest, "train_avg_preds.csv"),
                        sep=',', header=True, index=False, encoding='utf-8')

        if self.avg_pred_val is not None:
            df_val = pd.DataFrame({"Classes": list(self.classes.values()),
                                   "avg_pred": self.avg_pred_val.cpu()})
            df_val.to_csv(path_or_buf=os.path.join(self.path_dest, "val_avg_preds.csv"),
                          sep=',', header=True, index=False, encoding='utf-8')

    def get_top_freq_classes(self):
        n_top = 10
        top_idx = torch.argsort(-self.avg_pred_train.cpu())[:n_top]
        top_classes = np.array(list(self.classes.values()))[top_idx]
        print('Prior (train), first {} classes: {}'.format(n_top, top_classes))

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()/1000

class Query2Label(nn.Module):
    """Modified Query2Label model

    Unlike the model described in the paper (which uses a modified DETR 
    transformer), this version uses a standard, unmodified Pytorch Transformer. 
    Learnable label embeddings are passed to the decoder module as the target 
    sequence (and ultimately is passed as the Query to MHA).
    """
    def __init__(
        self, model, conv_out, num_classes, hidden_dim=256, nheads=8, 
        encoder_layers=6, decoder_layers=6, use_pos_encoding=False):
        """Initializes model

        Args:
            model (str): Timm model descriptor for backbone.
            conv_out (int): Backbone output channels.
            num_classes (int): Number of possible label classes
            hidden_dim (int, optional): Hidden channels from linear projection of
            backbone output. Defaults to 256.
            nheads (int, optional): Number of MHA heads. Defaults to 8.
            encoder_layers (int, optional): Number of encoders. Defaults to 6.
            decoder_layers (int, optional): Number of decoders. Defaults to 6.
            use_pos_encoding (bool, optional): Flag for use of position encoding. 
            Defaults to False.
        """        
        
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_pos_encoding = use_pos_encoding

        self.backbone = TimmBackbone(model)
        self.conv = nn.Conv2d(conv_out, hidden_dim, 1)
        self.transformer = nn.Transformer(
            hidden_dim, nheads, encoder_layers, decoder_layers)

        if self.use_pos_encoding:
            # returns the encoding object
            self.pos_encoder = PositionalEncodingPermute2D(hidden_dim)

            # returns the summing object
            self.encoding_adder = Summer(self.pos_encoder)

        # prediction head
        self.classifier = nn.Linear(num_classes * hidden_dim, num_classes)

        # learnable label embedding
        self.label_emb = nn.Parameter(torch.rand(1, num_classes, hidden_dim))

    def forward(self, x):
        """Passes batch through network

        Args:
            x (Tensor): Batch of images

        Returns:
            Tensor: Output of classification head
        """        
        # produces output of shape [N x C x H x W]
        out = self.backbone(x)
        
        # reduce number of feature planes for the transformer
        h = self.conv(out)
        B, C, H, W = h.shape

        # add position encodings
        if self.use_pos_encoding:
            
            # input with encoding added
            h = self.encoding_adder(h*0.1)

        # convert h from [N x C x H x W] to [H*W x N x C] (N=batch size)
        # this corresponds to the [SIZE x BATCH_SIZE x EMBED_DIM] dimensions 
        # that the transformer expects
        h = h.flatten(2).permute(2, 0, 1)
        
        # image feature vector "h" is sent in after transformation above; we 
        # also convert label_emb from [1 x TARGET x (hidden)EMBED_SIZE] to 
        # [TARGET x BATCH_SIZE x (hidden)EMBED_SIZE]
        label_emb = self.label_emb.repeat(B, 1, 1)
        label_emb = label_emb.transpose(0, 1)
        h = self.transformer(h, label_emb).transpose(0, 1)
        
        # output from transformer was of dim [TARGET x BATCH_SIZE x EMBED_SIZE];
        # however, we transposed it to [BATCH_SIZE x TARGET x EMBED_SIZE] above.
        # below we reshape to [BATCH_SIZE x TARGET*EMBED_SIZE].
        #
        # next, we project transformer outputs to class labels
        h = torch.reshape(h,(B, self.num_classes * self.hidden_dim))

        return self.classifier(h)

class TimmBackbone(nn.Module):
    """Specified timm model without pooling or classification head"""

    def __init__(self, model_name):
        """Downloads and instantiates pretrained model

        Args:
            model_name (str): Name of model to instantiate.
        """
        super().__init__()

        # Creating the model in this way produces unpooled, unclassified features
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0, global_pool=""
        )

    def forward(self, x):
        """Passes batch through backbone

        Args:
            x (Tensor): Batch tensor

        Returns:
            Tensor: Unpooled, unclassified features from image model.
        """

        out = self.model(x)

        return out


class Query2LabelTrainModule(pl.LightningModule):
    def __init__(
        self,
        data,
        backbone_desc,
        conv_out_dim,
        hidden_dim,
        num_encoders,
        num_decoders,
        num_heads,
        batch_size,
        image_dim,
        learning_rate,
        momentum,
        weight_decay,
        n_classes,
        thresh=0.5,
        use_cutmix=False,
        use_pos_encoding=False,
        loss="BCE",
    ):
        super().__init__()

        # Key parameters
        self.save_hyperparameters(ignore=["model", "data"])
        self.data = data
        self.model = Query2Label(
            model=backbone_desc,
            conv_out=conv_out_dim,
            num_classes=n_classes,
            hidden_dim=hidden_dim,
            nheads=num_heads,
            encoder_layers=num_encoders,
            decoder_layers=num_decoders,
            use_pos_encoding=use_pos_encoding,
        )

        for name, param in self.model.named_parameters():                
          if name.startswith('backbone'):
            param.requires_grad = False
        if loss == "BCE":
            self.base_criterion = nn.BCEWithLogitsLoss()
        elif loss == "ASL":
            self.base_criterion = AsymmetricLoss(gamma_neg=1, gamma_pos=0)

        self.criterion = CutMixCriterion(self.base_criterion)

    def forward(self, x):
        x = self.model(x)
        return x

    def evaluate(self, batch, stage=None, imshow=False):
        x, y = batch
        y_hat = self(x)
        loss = self.base_criterion(y_hat, y.type(torch.float))

        rmap = tf.retrieval_average_precision(y_hat, y.type(torch.int))

        category_prec = precision(
            y_hat,
            y.type(torch.int),
            average="macro",
            num_classes=self.hparams.n_classes,
            threshold=self.hparams.thresh,
            multiclass=False,
        )
        category_recall = tf.recall(
            y_hat,
            y.type(torch.int),
            average="macro",
            num_classes=self.hparams.n_classes,
            threshold=self.hparams.thresh,
            multiclass=False,
        )
        category_f1 = tf.f1_score(
            y_hat,
            y.type(torch.int),
            average="macro",
            num_classes=self.hparams.n_classes,
            threshold=self.hparams.thresh,
            multiclass=False,
        )

        overall_prec = precision(
            y_hat, y.type(torch.int), threshold=self.hparams.thresh, multiclass=False
        )
        overall_recall = tf.recall(
            y_hat, y.type(torch.int), threshold=self.hparams.thresh, multiclass=False
        )
        overall_f1 = tf.f1_score(
            y_hat, y.type(torch.int), threshold=self.hparams.thresh, multiclass=False
        )

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_rmap", rmap, prog_bar=True, on_step=False, on_epoch=True)

            self.log(f"{stage}_cat_prec", category_prec, prog_bar=True)
            self.log(f"{stage}_cat_recall", category_recall, prog_bar=True)
            self.log(f"{stage}_cat_f1", category_f1, prog_bar=True)

            self.log(f"{stage}_ovr_prec", overall_prec, prog_bar=True)
            self.log(f"{stage}_ovr_recall", overall_recall, prog_bar=True)
            self.log(f"{stage}_ovr_f1", overall_f1, prog_bar=True)

            # log prediction examples to wandb
            
            pred = self.model(x)
            pred_keys = pred[0].sigmoid().tolist()
            pred_keys = [0 if p < self.hparams.thresh else 1 for p in pred_keys]

            if imshow:
                mapper = COCOCategorizer()
                pred_lbl = mapper.get_labels(pred_keys)
                print("Actual : ", mapper.get_labels(y.type(torch.int)[0].tolist()))
                print("Predicted : ",pred_lbl)

                inv_normalize = transforms.Normalize(
                    mean = [-0.485/0.229, -0.456/0.224, -0.406/0.255],
                    std = [1/0.229, 1/0.224, 1/0.255]
                )

                inv_tensor = inv_normalize(x[0])
                plt.imshow(inv_tensor.permute(1,2,0).detach().cpu().numpy())
                plt.show()
                print("-----------------")
            try:
              self.logger.experiment.log({"val_pred_examples": [wandb.Image(x[0], caption=pred_lbl)]})
            except AttributeError:
                pass
            

    def training_step(self, batch, batch_idx):
        if self.hparams.use_cutmix:
            x, y = batch
            y_hat = self(x)
            # y1, y2, lam = y
            loss = self.criterion(y_hat, y)

        else:
            x, y = batch
            y_hat = self(x)
            loss = self.base_criterion(y_hat, y.type(torch.float))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test", True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                self.hparams.learning_rate,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=len(self.data.train_dataloader()),
                anneal_strategy="cos",
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        # return optimizer


class COCODataModule(pl.LightningDataModule):
    """Datamodule for Lightning Trainer"""

    def __init__(
        self,
        data_dir,
        img_size,
        batch_size=4,
        num_workers=0,
        use_cutmix=False,
        cutmix_alpha=1.0,
    ) -> None:
        """_summary_

        Args:
            data_dir (str): Location of data.
            img_size (int): Desired size for transformed images.
            batch_size (int, optional): Dataloader batch size. Defaults to 128.
            num_workers (int, optional): Number of CPU threads to use. Defaults to 0.
            use_cutmix (bool, optional): Flag to enable Cutmix augmentation. Defaults to False.
            cutmix_alpha (float, optional): Defaults to 1.0.
        """
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.collator = torch.utils.data.dataloader.default_collate

    def prepare_data(self) -> None:
        """Loads metadata file and subsamples it if requested"""
        pass

    def setup(self, stage=None) -> None:
        """Creates train, validation, test datasets

        Args:
            stage (str, optional): Stage. Defaults to None.
        """
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        # mean=[0, 0, 0],
        # std=[1, 1, 1])

        train_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                RandAugment(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        
        self.train_set = CoCoDataset(
            image_dir=(self.data_dir + "train/"),
            anno_path=(self.data_dir + "train.json"),
            input_transform=train_transforms,
            labels_path=(self.data_dir + "labels_train2014.npy"),
        )
        
        self.val_set = CoCoDataset(
            image_dir=(self.data_dir + "test"),
            anno_path=(self.data_dir + "test.json"),
            input_transform=test_transforms,
            labels_path=(self.data_dir + "labels_val2014.npy"),
        )
        """
        self.test_set = CoCoDataset(
            image_dir=(self.data_dir+"valid/"),
            anno_path=(self.data_dir+"valid.json"),
            input_transform=test_transforms,
            labels_path=(self.data_dir+"annotations/labels_test2014.npy"))
"""
        if self.use_cutmix:
            self.collator = CutMixCollator(self.cutmix_alpha)

    def get_num_classes(self):
        """Returns number of classes

        Returns:
            int: number of classes
            
            
        """
        return len(self.classes)

    def train_dataloader(self) -> DataLoader:
        """Creates and returns training dataloader

        Returns:
            DataLoader: Training dataloader
        """
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader:
        """Creates and returns validation dataloader

        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Creates and returns test dataloader

        Returns:
            DataLoader: Test dataloader
        """
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

category_map = {}

for i in range(0,100):
    category_map[str(i)] = i

#print(category_map)
class CoCoDataset(data.Dataset):
    """Custom dataset that will load the COCO 2014 dataset and annotations

    This module will load the basic files as provided here: https://cocodataset.org/#download
    If the labels file does not exist yet, it will be created with the included
    helper functions. This class was largely taken from Shilong Liu's repo at
    https://github.com/SlongLiu/query2labels/blob/main/lib/dataset/cocodataset.py.

    Attributes:
        coco (torchvision dataset): Dataset containing COCO data.
        category_map (dict): Mapping of category names to indices.
        input_transform (list of transform objects): List of transforms to apply.
        labels_path (str): Location of labels (if they exist).
        used_category (int): Legacy var.
        labels (list): List of labels.

    """

    def __init__(
        self,
        image_dir,
        anno_path,
        input_transform=None,
        labels_path=None,
        used_category=-1,
    ):
        """Initializes dataset

        Args:
            image_dir (str): Location of COCO images.
            anno_path (str): Location of COCO annotation files.
            input_transform (list of Transform objects, optional): List of transforms to apply.  Defaults to None.
            labels_path (str, optional): Location of labels. Defaults to None.
            used_category (int, optional): Legacy var. Defaults to -1.
        """
        self.coco = dset.CocoDetection(root=image_dir, annFile=anno_path)
        # with open('./data/coco/category.json','r') as load_category:
        #     self.category_map = json.load(load_category)
        self.category_map = category_map
        self.input_transform = input_transform
        self.labels_path = labels_path
        self.used_category = used_category

        self.labels = []
        if os.path.exists(self.labels_path):
            self.labels = np.load(self.labels_path).astype(np.float64)
            self.labels = (self.labels > 0).astype(np.float64)
        else:
            print("No preprocessed label file found in {}.".format(self.labels_path))
            l = len(self.coco)
            for i in tqdm(range(l)):
                item = self.coco[i]
                # print(i)
                categories = self.getCategoryList(item[1])
                label = self.getLabelVector(categories)
                self.labels.append(label)
            self.save_datalabels(labels_path)
        # import ipdb; ipdb.set_trace()

    def __getitem__(self, index):
        input = self.coco[index][0]
        if self.input_transform:
            input = self.input_transform(input)
        return input, self.labels[index]

    def getCategoryList(self, item):
        """Turns iterable item into list of categories

        Args:
            item (iterable): Any iterable type that contains categories

        Returns:
            list: Categories
        """
        categories = set()
        for t in item:
            categories.add(t["category_id"])
        return list(categories)

    def getLabelVector(self, categories):
        """Creates multi-hot vector for item labels

        Args:
            categories (list): List of categories matching an item

        Returns:
            ndarray: Multi-hot vector for item labels
        """
        label = np.zeros(100)
        # label_num = len(categories)
        for c in categories:
            index = self.category_map[str(c)] - 1
            label[index] = 1.0  # / label_num
        return label

    def __len__(self):
        return len(self.coco)

    def save_datalabels(self, outpath):
        """Saves datalabels to disk for faster loading next time.

        Args:
            outpath (str): Location where labels are to be saved.
        """
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        labels = np.array(self.labels)
        np.save(outpath, labels)

# This implementation of CutMix was taken from the Github repo here:
# https://github.com/hysts/pytorch_cutmix
# The content of the above repo is usable via the standard MIT license.

def cutmix(batch, alpha):
    """Applies random CutMix to images in batch

    Args:
        batch (Tensor): Images and labels
        alpha (float): Alpha value for CutMix algorithm

    Returns:
        tuple (Tensor, tuple): Shuffled images, and tuple containing targets,
        shuffled targets, and lambda (loss weighting value)
    """
    data, targets = batch

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets


class CutMixCollator:
    """Custom Collator for dataloader"""

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = cutmix(batch, self.alpha)
        return batch


class CutMixCriterion:
    """Applies criterion in a weighted fashion based on image shuffling"""

    def __init__(self, criterion):
        """Creates loss function

        Args:
            criterion (torch.nn loss object): Should be a binary loss class
        """
        self.criterion = criterion

    def __call__(self, preds, targets):
        """Applies loss function

        Args:
            preds (Tensor): Vector of prediction logits
            targets (tuple of Tensors): Targets and shuffled targets

        Returns:
            float: calculated loss
        """
        targets1, targets2, lam = targets
        return lam * self.criterion(preds, targets1) + (1 - lam) * self.criterion(
            preds, targets2
        )

class COCOCategorizer:
    """Creates list of English-language labels corresponding to COCO label vector

    Attributes:
        cat_dict (dict): Dictionary mapping label codes to names.
    """

    def __init__(self):
        """Creates label code-name mapping"""
        f = open("pascalvoc_labels.txt")
        #f = open("/content/drive/MyDrive/pfee/mscoco/coco-labels-2014-2017.txt")

        category_list = [line.rstrip("\n") for line in f]
        self.cat_dict = {cat: key for cat, key in enumerate(category_list)}

    def get_labels(self, pred_list):
        """_summary_

        Args:
            pred_list (list of ints): Multi-hot list of label codes from prediction

        Returns:
            list of strings: List of label names in English.
        """
        labels = [self.cat_dict[i] for i in range(len(pred_list)) if pred_list[i] == 1]
        return labels

param_dict = {
    "backbone_desc":"resnest101e",
    "conv_out_dim":2048,
    "hidden_dim":256,
    "num_encoders":1,
    "num_decoders":2,
    "num_heads":8,
    "batch_size":8,
    "image_dim":576,
    "learning_rate":0.0001, 
    "momentum":0.9,
    "weight_decay":0.01, 
    "n_classes":100,
    "thresh":0.5,
    "use_cutmix":True,
    "use_pos_encoding":False,
    "loss":"ASL"
}
coco = COCODataModule(
    "./",
    img_size=param_dict["image_dim"],
    batch_size=param_dict["batch_size"],
    num_workers=16,
    use_cutmix=param_dict["use_cutmix"],
    cutmix_alpha=1.0)
param_dict["data"] = coco
pl_model = Query2LabelTrainModule(**param_dict)

# Save checkpoint every epoch
checkpoint_callback = ModelCheckpoint(dirpath='./checkpoint/')

trainer = pl.Trainer(
    max_epochs=1,
    num_sanity_val_steps=0,
    accelerator='cpu', 
    strategy="ddp",
    devices=1,
    default_root_dir="training/checkpoints/",
    log_every_n_steps=1,
    callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback])


# Train the model
trainer.fit(pl_model, param_dict["data"])

# Load a model from a checkpoint
#trainer.fit(pl_model, param_dict["data"], ckpt_path="/content/training/checkpoints/lightning_logs/version_6/checkpoints/epoch=1-step=2716.ckpt")

# Testing the model on test set
trainer.test(pl_model, coco.test_dataloader())    