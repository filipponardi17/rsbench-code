from countrss_module import Dataset
from pyeda.inter import Xor

class ConfigurableXOR(Dataset):
    def __init__(self, n_variables, cnf_path="configurable_xor"):
        super().__init__([2]*n_variables, cnf_path)
        self.n_variables = n_variables

    def make_data(self):
        # Non fa niente, i dati verranno caricati dall'esterno
        pass

    def load_data(self, gs, ys):
        # gs: array di input (già one-hot encoded, come si fa in XorDataset)
        # ys: array di label
        self.gvecs = gs
        self.ys = ys

    def k(self, cvec, y):
        # Stessa logica di XorDataset: XOR su tutte le variabili
        return Xor(*[cvec[i] for i in range(1, len(cvec), 2)]) if y else ~Xor(*[cvec[i] for i in range(1, len(cvec), 2)])

    def encode_background(self, A):
        return True