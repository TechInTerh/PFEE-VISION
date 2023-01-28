import torch
from torch import nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import precision
import torchmetrics.functional as tf
import torchvision.transforms as transforms
from randaugment import RandAugment
from torch.utils.data import DataLoader

import pytorch_lightning as pl


from pytorch_lightning.loggers import WandbLogger # Comment out if not using wandb
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from positional_encodings.torch_encodings import PositionalEncodingPermute2D, Summer

import numpy as np
import timm
import matplotlib.pyplot as plt

from tqdm import tqdm

import sys
import json
import random

from Loss import *
from DataAugmentation import *
from CocoFormat import *

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
        self.backbone = torch.load("resnet-model.pth")
        #self.backbone = TimmBackbone(model)
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



class_found = np.array([0] * 100)
total = np.array([0] * 100)

y_preds = []
y_label = []

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
        self.over_prec = []
        self.count = 0
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
            self.over_prec.append(overall_prec)
            self.count += 1
            # log prediction examples to wandb
            if (self.count == 32):
               with open("precision.txt", "a") as f:
                   f.write("precision = "+str(sum(self.over_prec)/len(self.over_prec))+'\n')
               self.count = 0
               self.over_prec = []
            pred = self.model(x)
            for i in range(len(pred)):
               pred_keys = pred[i].sigmoid().tolist()
               pred_keys = [0 if p < self.hparams.thresh else 1 for p in pred_keys]

               y_true = y.type(torch.int)[i].tolist()
               #print(y_true)
               y_preds.append(pred_keys)
               y_label.append(y_true)
               for j in range(len(y_true)):
                  if y_true[j] == 1:
                     total[j] += 1
                     if pred_keys[j] == 1:
                        class_found[j] += 1

            if imshow:
                mapper = COCOCategorizer()
                pred_lbl = mapper.get_labels(pred_keys)
                print("Actual : ", mapper.get_labels(y.type(torch.int)[0].tolist()))
                print("Predicted : ",pred_lbl)

                #inv_normalize = transforms.Normalize(
                #    mean = [-0.485/0.229, -0.456/0.224, -0.406/0.255],
                #    std = [1/0.229, 1/0.224, 1/0.255]
                #)

               # inv_tensor = inv_normalize(x[0])
               # plt.imshow(inv_tensor.permute(1,2,0).detach().cpu().numpy())
               # plt.show()
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




category_map = {}

for i in range(0,100):
    category_map[str(i)] = i

#print(category_map)








param_dict = {
    "backbone_desc":"resnest101e",
    "conv_out_dim":2048,
    "hidden_dim":256,
    "num_encoders":1,
    "num_decoders":2,
    "num_heads":8,
    "batch_size":32,
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
data = COCODataModule(
    "./",
    img_size=param_dict["image_dim"],
    batch_size=param_dict["batch_size"],
    num_workers=16,
    use_cutmix=param_dict["use_cutmix"],
    cutmix_alpha=1.0)
param_dict["data"] = data
pl_model = Query2LabelTrainModule(**param_dict)

# Save checkpoint every epoch
checkpoint_callback = ModelCheckpoint(dirpath='./checkpoint/')

trainer = pl.Trainer(
    max_epochs=10,
    num_sanity_val_steps=0,
    accelerator='cpu',
    strategy="ddp",
    devices=1,
    default_root_dir="training/checkpoints/",
    log_every_n_steps=1,
    callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback])


# Train the model
#trainer.fit(pl_model, param_dict["data"])

# Load a model from a checkpoint
trainer.fit(pl_model, param_dict["data"], ckpt_path="checkpoint/epoch=9-step=2300.ckpt")

# Testing the model on test set
trainer.test(pl_model, data.test_dataloader())
print(class_found/total)


print("y_true : ")
print(y_label)
print("y_pred : ")
print(y_preds)
