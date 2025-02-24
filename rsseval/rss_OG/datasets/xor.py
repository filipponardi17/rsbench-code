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
        # Filtraggio del train 
        #self.filtrate()
        # ======================================

        # import numpy as np
        # print("train")
        # print(np.unique(self.dataset_train.labels, axis=-1, return_counts=True)) 
        # print("test")
        # print(np.unique(self.dataset_test.labels, axis=-1, return_counts=True))
        # print("val")
        # print(np.unique(self.dataset_val.labels, axis=-1, return_counts=True))
        # quit()

        ########################################################
        #
        #SERVE SE RANDOM (PRIMA CALL) HA TROPPI POCHI TASK
        #
        #########################################################
        # Se il dataset filtrato ha meno sample di args.batch_size, effettua oversampling
        # filtered_count = len(self.dataset_train.labels)
        # if filtered_count < self.args.batch_size:
        #     oversample_count = self.args.batch_size - filtered_count
        #     print(f"Dataset filtrato molto piccolo: {filtered_count} sample. Oversampling {oversample_count} sample casualmente per raggiungere {self.args.batch_size}.")
        #     import numpy as np
        #     # Seleziona indici casuali con ripetizione (oversampling con replacement)
        #     random_indices = np.random.randint(0, filtered_count, oversample_count)
        #     # Supponendo che self.dataset_train.labels e self.dataset_train.concepts siano array numpy
        #     self.dataset_train.labels = np.concatenate([self.dataset_train.labels, self.dataset_train.labels[random_indices]])
        #     self.dataset_train.concepts = np.concatenate([self.dataset_train.concepts, self.dataset_train.concepts[random_indices]])
        #     # Per le immagini, che sono salvate come lista, convertiamo in array, oversample e riconvertiamo a lista
        #     current_images = np.array(self.dataset_train.list_images, dtype=object)
        #     new_images = current_images[random_indices]
        #     self.dataset_train.list_images = np.concatenate([current_images, new_images]).tolist()

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
        # Verifica che gli argomenti 'current_step', 'method' e 'run' siano presenti
        if not hasattr(self.args, "current_step"):
            raise ValueError("Argument 'current_step' not found in args. Please provide --step when running the experiment.")
        if not hasattr(self.args, "method"):
            raise ValueError("Argument 'method' not found in args. Please provide --method when running the experiment.")
        if not hasattr(self.args, "run"):
            raise ValueError("Argument 'run' not found in args. Please provide --run when running the experiment.")

        # Costruisci il percorso del CSV in base al run corrente
        run_number = self.args.run
        csv_file = f"/home/filippo.nardi/rsbench-code/rsseval/rss_OG/csv/rq3_output_selection_order{run_number}.csv"

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file '{csv_file}' not found.")

        # Scegli la colonna da usare in base al metodo specificato
        column_to_use = f"selection_{self.args.method}_patterns_expanded"

        cumulative_patterns = []
        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            if self.args.current_step > len(rows):
                raise ValueError(f"Requested current_step {self.args.current_step} exceeds the number of rows in the CSV ({len(rows)}).")
            # Elaborazione cumulativa dei pattern fino allo step corrente
            for i in range(self.args.current_step):
                row = rows[i]
                patterns_str = row[column_to_use]
                if patterns_str.strip() != "":
                    pattern_list = patterns_str.split(";")
                    for pattern in pattern_list:
                        pattern = pattern.strip()
                        if pattern:
                            # Converte la stringa in lista di interi, es. "0,0,0,0" -> [0, 0, 0, 0]
                            pattern_int = [int(x.strip()) for x in pattern.split(",")]
                            cumulative_patterns.append(pattern_int)
        print(f"Cumulative patterns up to step {self.args.current_step} using column '{column_to_use}': {cumulative_patterns}")

        # Filtra il dataset_train mantenendo solo i sample per cui il vettore dei concetti è presente in cumulative_patterns
        keep_mask = np.array([
            c.tolist() in cumulative_patterns for c in self.dataset_train.concepts
        ], dtype=bool)

        self.dataset_train.labels = self.dataset_train.labels[keep_mask]
        self.dataset_train.concepts = self.dataset_train.concepts[keep_mask]
        self.dataset_train.list_images = np.array(self.dataset_train.list_images, dtype=object)[keep_mask].tolist()

        old_size = len(self.dataset_train.labels) + (len(self.dataset_train.labels) - np.sum(keep_mask))
        new_size = len(self.dataset_train.labels)
        print(f"Filtrate train set: retained {new_size} samples out of {old_size} total.")