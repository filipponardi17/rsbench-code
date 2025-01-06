import itertools as it
import operator as op

class Dataset:
    def __init__(self, domain_sizes, cnf_path):
        self.domain_sizes = domain_sizes
        self.cnf_path = cnf_path
        self.gvecs = None
        self.ys = None

class XorDataset(Dataset):
    """Class implementing a XOR task (for testing purposes)."""

    def __init__(self, args):
        super().__init__(
            [2 for _ in range(args.n_variables)],
            f"xor{args.n_variables}"
        )

    def make_data(self):
        xor = lambda x: list(it.accumulate(x, op.xor, initial=False))[-1]
        self.gvecs, self.ys = self._make_all_data(xor)

    def load_data(self):
        raise NotImplementedError()

    def k(self, cvec, y):
        constraint = Xor(*[cvec[i] for i in range(1, len(cvec), 2)])
        # print("constraint")
        # print(constraint)
        return constraint if y else ~constraint

    def encode_background(self, A):
        return True