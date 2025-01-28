#gpt made
from argparse import Namespace
from datasets.utils.base_dataset import BaseDataset, XOR_get_loader
from datasets.utils.xor_creation import XORDataset
from backbones.cnnnosharing import CBMNoSharing, MNISTLCNN
import time
import numpy as np

class MNLOGIC(BaseDataset):
    NAME = "xor"

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.return_embeddings = False

    def get_data_loaders(self):
        start = time.time()

        self.dataset_train = XORDataset(
            base_path="data/mnlogic",
            split="train",
            c_sup=self.args.c_sup,
            which_c=self.args.which_c,
        )
        self.dataset_val = XORDataset(
            base_path="data/mnlogic",
            split="val",
        )
        self.dataset_test = XORDataset(
            base_path="data/mnlogic",
            split="test",
        )
        self.dataset_ood = XORDataset(
            base_path="data/mnlogic",
            split="ood",
        )

        print(f"Loaded datasets in {time.time() - start} s.")
        print(
            "Len loaders: \n train:",
            len(self.dataset_train),
            "\n val:",
            len(self.dataset_val),
        )
        print(" len test:", len(self.dataset_test))

        # ======================================
        # Filtraggio del train set per le combo desiderate
        self.filtrate()
        # ======================================

        keep_order = True if self.return_embeddings else False
        self.train_loader = XOR_get_loader(
            self.dataset_train, self.args.batch_size, val_test=keep_order
        )
        self.val_loader = XOR_get_loader(
            self.dataset_val, self.args.batch_size, val_test=True
        )
        self.test_loader = XOR_get_loader(
            self.dataset_test, self.args.batch_size, val_test=True
        )
        self.ood_loader = XOR_get_loader(
            self.dataset_ood, self.args.batch_size, val_test=True
        )

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.backbone == "neural":
            return MNISTLCNN(), None
        return CBMNoSharing(num_images=4, nout=2), None

    def get_split(self):
        return 4, ()

    def get_concept_labels(self):
        return [0, 1]

    def get_labels(self):
        return [0, 1]

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train))
        print("Validation samples", len(self.dataset_val))
        print("Test samples", len(self.dataset_test))
        print("Test OOD samples", len(self.dataset_ood))

    

    def filtrate(self):
        combos_to_keep = [
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 1, 1, 0]
        ]

        old_size = len(self.dataset_train.labels)
        keep_mask = []

        for i in range(old_size):
            c_values = self.dataset_train.concepts[i]
            c_int = c_values.astype(int).tolist()
            keep_mask.append(c_int in combos_to_keep)

        keep_mask = np.array(keep_mask, dtype=bool)

        # Converte list_images in array NumPy (dtype=object se gli elementi sono stringhe di path)
        arr_list_images = np.array(self.dataset_train.list_images, dtype=object)

        # Applica la maschera booleana
        arr_list_images = arr_list_images[keep_mask]
        self.dataset_train.labels = self.dataset_train.labels[keep_mask]
        self.dataset_train.concepts = self.dataset_train.concepts[keep_mask]

        # Se vuoi rimanere con una lista Python per list_images:
        self.dataset_train.list_images = arr_list_images.tolist()

        new_size = len(self.dataset_train.labels)
        print(f"Filtrate train set: mantenuti {new_size} campioni su {old_size} totali.")