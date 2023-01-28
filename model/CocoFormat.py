import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from randaugment import RandAugment
from DataAugmentation import *
from torch.utils.data import DataLoader
import torch.utils.data as data
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image
import requests

category_map = {}

for i in range(0,100):
    category_map[str(i)] = i

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
            anno_path=("train_labels.csv"),
            input_transform=train_transforms,
            labels_path=(self.data_dir + "labels_train.npy"),
        )

        self.val_set = CoCoDataset(
            image_dir=(self.data_dir + "test"),
            anno_path=("val_labels.csv"),
            input_transform=test_transforms,
            labels_path=(self.data_dir + "labels_val.npy"),
        )

        self.test_set = CoCoDataset(
            image_dir=(self.data_dir+"valid/"),
            anno_path=("test_labels.csv"),
            input_transform=test_transforms,
            labels_path=(self.data_dir+"labels_test.npy"))

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
        self.category_map = category_map
        self.input_transform = input_transform
        self.labels_path = labels_path
        self.used_category = used_category
        self.df = pd.read_csv(anno_path)
        self.labels = []
        if os.path.exists(self.labels_path):
            self.labels = np.load(self.labels_path).astype(np.float64)
            self.labels = (self.labels > 0).astype(np.float64)
        else:
            print("No preprocessed label file found in {}.".format(self.labels_path))
            l = len(self.df)
            for i in tqdm(range(l)):
                label = self.df.loc[i, "labels"]
                label = label[1:-1].split(",")
                label = [float(j) for j in label]
                self.labels.append(np.array(label))
            self.save_datalabels(labels_path)

    def __getitem__(self, index):
        try:
            image_path = "http://10.100.34.3/IN/" + self.df["REFERENCE"][index] + ".jpg"
            input = Image.open(requests.get(image_path, stream= True).raw).convert("RGB")
            if self.input_transform:
                input = self.input_transform(input)
            return input, self.labels[index]
        except:
            return self.input_transform(Image.fromarray(np.zeros((576,576))).convert("RGB")), np.zeros(100)

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
        return len(self.df)

    def save_datalabels(self, outpath):
        """Saves datalabels to disk for faster loading next time.

        Args:
            outpath (str): Location where labels are to be saved.
        """
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        labels = np.array(self.labels)
        np.save(outpath, labels)

class COCOCategorizer:
    """Creates list of English-language labels corresponding to COCO label vector

    Attributes:
        cat_dict (dict): Dictionary mapping label codes to names.
    """

    def __init__(self):
        """Creates label code-name mapping"""
        f = open("labels.txt")
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