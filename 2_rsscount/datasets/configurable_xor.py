# project/datasets/configurable_xor.py

from datasets.xor_dataset import XorDataset


class ConfigurableXOR(XorDataset):
    """
    Classe che implementa XOR ma non genera internamente gvecs, ys.
    Invece, li riceve dall'esterno mediante set_data().
    """
    def __init__(self, args):
        # Chiama il costruttore di XorDataset ma non farà uso di make_data interno
        super().__init__(args)
        # Annulliamo i gvecs e ys per ora
        self.gvecs = None
        self.ys = None

    def set_data(self, gs, ys):
        # gs è un array (o lista di liste) di feature
        # ys è una lista di label corrispondenti
        self.gvecs = gs
        self.ys = ys

    def make_data(self):
        # Non generiamo i dati internamente, assumiamo siano stati già settati
        if self.gvecs is None or self.ys is None:
            raise ValueError("Dati non impostati. Chiamare set_data(gs, ys) prima di make_data().")
        # Qui non facciamo altro, i dati sono già pronti.