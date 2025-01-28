#STRAIGHT FROM DEEPSEEK

from argparse import Namespace
from datasets.utils.base_dataset import BaseDataset, XOR_get_loader
from datasets.utils.xor_creation import XORDataset
from backbones.cnnnosharing import CBMNoSharing, MNISTLCNN
import time
import numpy as np
import joblib
from pathlib import Path

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

        # Aggiungi il filtraggio
        self.filtrate(self.dataset_train)

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

        print(f"Loaded datasets in {time.time()-start} s.")
        print("Len loaders: \n train:", len(self.dataset_train), 
              "\n val:", len(self.dataset_val))
        print(" len test:", len(self.dataset_test))

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

    def filtrate(self, dataset):
        # Combinazioni consentite (convertite da booleani a 0/1)
        valid_combinations = [
            [0, 0, 0, 0],  # False, False, False, False
            [0, 1, 0, 1],  # False, True,  False, True
            [1, 1, 1, 0]   # True,  True,  True,  False
        ]

        # Carica tutti i concetti
        concept_paths = sorted(Path(dataset.base_path, dataset.split).glob("*.joblib"))
        all_concepts = []
        
        for path in concept_paths:
            data = joblib.load(path)
            # Converti i booleani in 0/1
            concepts = [int(c) for c in data['meta']['concepts']]
            all_concepts.append(concepts)

        # Converti in array numpy
        concept_array = np.array(all_concepts)
        
        # Crea la maschera
        mask = np.zeros(len(concept_array), dtype=bool)
        for combo in valid_combinations:
            mask |= np.all(concept_array == combo, axis=1)

        # Applica il filtro a tutti gli elementi del dataset
        dataset.image_paths = [p for i, p in enumerate(dataset.image_paths) if mask[i]]
        dataset.labels = [l for i, l in enumerate(dataset.labels) if mask[i]]
        dataset.concept_paths = [cp for i, cp in enumerate(dataset.concept_paths) if mask[i]]

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
