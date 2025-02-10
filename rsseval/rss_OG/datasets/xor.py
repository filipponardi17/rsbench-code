# xor.py
import time
import numpy as np
import csv  # Assicurati di importare csv
from datasets.utils.base_dataset import BaseDataset, XOR_get_loader
from datasets.utils.xor_creation import XORDataset
from backbones.cnnnosharing import CBMNoSharing, MNISTLCNN
import os
from argparse import Namespace

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
        # Filtraggio del train set per i task selezionati
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
        # Verifica che args abbia il campo current_step
        if not hasattr(self.args, "current_step"):
            raise ValueError("Argument 'current_step' not found in args. Please provide --step when running the experiment.")

        csv_file = "/home/filippo.nardi/rsbench-code/rsseval/rss_OG/csv/output_selection_order1.csv"
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file '{csv_file}' not found.")

        cumulative_patterns = []
        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            # Controlla che il numero di righe sia sufficiente
            if self.args.current_step > len(rows):
                raise ValueError(f"Requested current_step {self.args.current_step} exceeds the number of rows in the CSV ({len(rows)}).")
            # Accumula i pattern dalla colonna "selection_greedy_patterns_expanded" per le righe da 1 fino a current_step
            for i in range(self.args.current_step):
                row = rows[i]
                # Usa la colonna con i pattern espansi
                patterns_str = row["selection_greedy_patterns_expanded"]
                if patterns_str.strip() != "":
                    pattern_list = patterns_str.split(";")
                    for pattern in pattern_list:
                        pattern = pattern.strip()
                        if pattern:
                            # Converte la stringa in lista di interi, es. "0,0,0,0" -> [0, 0, 0, 0]
                            pattern_int = [int(x.strip()) for x in pattern.split(",")]
                            cumulative_patterns.append(pattern_int)
        print(f"Cumulative patterns up to step {self.args.current_step}: {cumulative_patterns}")

        # Crea la maschera di filtraggio: conserva solo i sample per cui il vettore dei concetti (convertito in lista) è presente in cumulative_patterns
        keep_mask = np.array([
            c.tolist() in cumulative_patterns for c in self.dataset_train.concepts
        ], dtype=bool)

        self.dataset_train.labels = self.dataset_train.labels[keep_mask]
        self.dataset_train.concepts = self.dataset_train.concepts[keep_mask]
        self.dataset_train.list_images = np.array(self.dataset_train.list_images, dtype=object)[keep_mask].tolist()

        old_size = len(keep_mask)
        new_size = len(self.dataset_train.labels)
        print(f"Filtrate train set: retained {new_size} samples out of {old_size} total.")