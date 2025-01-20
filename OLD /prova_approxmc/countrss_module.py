import argparse
import itertools as it
import operator as op
import numpy as np
import pickle

from abc import abstractmethod
from functools import reduce
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state

from pyeda.inter import exprvars, expr2dimacscnf
from pyeda.inter import And, Or, Xor, Implies, OneHot, Equal
#from countrss_module import Dataset
#from pyeda.inter import Xor


def _pp_solution(sol, nvars, nbits):
    """Pretty-print a pyeda model."""
    
    Asol = np.zeros(shape=(nbits, nbits),
                    dtype=np.int16)
    Osol = np.zeros(shape=(nvars, nvars),
                    dtype=np.int16)
    for k in sol:
        if k.name == 'A' and sol[k]:
            Asol[k.indices] = 1
        elif k.name == 'O' and sol[k]:
            Osol[k.indices] = 1

    print(Asol, "\n")
    print(Osol, "\n")
    

def _prop_or_count(n, p):
    return int(p if p > 1 else np.trunc(n * p))


def _booldot(avec, bvec):
    """Boolean dot product."""
    return reduce(op.or_, [a & b for a, b in zip(avec, bvec)])


def _bind(variables, clauses):
    """Binds clauses-of-ints to pyeda variables."""
    return And(*[
        Or(*[
            variables[int(i) - 1] if i > 0 else ~variables[-int(i) - 1]
            for i in clause
        ])
        for clause in clauses
    ])


def _read_cnf(path):
    """Reads a CNF in DIMACS format."""
    with open(path, "rt") as fp:
        lines = fp.readlines()

    try:
        header = lines[0].strip()
        _, fmt, n_variables, n_clauses = header.split()
        n_variables = int(n_variables)
        n_clauses = int(n_clauses)
        assert fmt == "cnf"
    except:
        raise RuntimeError("not a valid CNF file")

    clauses = []
    for line in lines[1:]:
        line = line.strip().split()
        assert line[-1] == "0"
        clause = list(map(int, line))[:-1]
        clauses.append(clause)
    assert len(clauses) == n_clauses

    return n_variables, clauses

# COPIA INCOLLA EZ
def count_rss(dataset):
    print(f"Building formula: {len(dataset.gvecs)} gvecs, {dataset.n_bits} bits")
    print("len data gvecs", len(dataset.gvecs))
    print("gvecs", dataset.gvecs)
    A = exprvars("A", dataset.n_bits, dataset.n_bits)
    O = exprvars("O", dataset.n_variables, dataset.n_variables)

    formula = And(*[OneHot(*O[:, k]) for k in range(dataset.n_variables)])

    nzb = lambda k1,k2 : Or(A[k1*2, k2*2], A[k1*2, k2*2 + 1],
                            A[k1*2 + 1, k2*2], A[k1*2 + 1, k2*2 + 1])

    formula &= And(*[Equal(O[k1, k2], nzb(k1, k2))
                     for k1 in range(dataset.n_variables)
                     for k2 in range(dataset.n_variables)])

    formula &= And(*[OneHot(*A[:, i])
                     for i in range(dataset.n_bits)])

    formula &= dataset.encode_background(A)

    for gvec, y in zip(dataset.gvecs, dataset.ys):
        cvec = [_booldot(A[i, :], gvec).simplify()
                for i in range(len(gvec))]
        offset = 0
        for vsize in dataset.domain_sizes:
            formula &= OneHot(*cvec[offset:offset+vsize])
            offset += vsize
        formula &= dataset.k(cvec, y)

    # Ora enumeriamo le soluzioni:
    n_sol = 0
    for sol in formula.satisfy_all():
        _pp_solution(sol, dataset.n_variables, dataset.n_bits)
        n_sol += 1
    print(f"{n_sol} solutions for this task")
    return n_sol


class Dataset:
    """Abstract Dataset(task) class."""

    def __init__(self, domain_sizes, cnf_path):
        self.domain_sizes = domain_sizes
        self.cnf_path = cnf_path
        self.gvecs, self.ys = None, None
        self.n_variables = len(domain_sizes)
        self.n_bits = sum(domain_sizes) # n bits per variable

    @abstractmethod
    def make_data(self):
        """Fills the .gvecs and .ys fields with synthetic data."""
        pass

    @abstractmethod
    def load_data(self, path):
        """Fills the .gvecs and .ys fields with actual annotations."""
        pass

    @abstractmethod
    def k(self, cvec, y):
        """Knowledge for a given example."""
        pass

    def _make_all_data(self, infer):
        """Generates all possible ground-truh concept vectors and labels."""
        gs = list(it.product(*[list(range(size)) for size in self.domain_sizes]))
        ys = [infer(g) for g in gs]
        gs, ys = np.array(gs), np.array(ys)

        gs = OneHotEncoder(
            categories=[list(range(size)) for size in self.domain_sizes],
            sparse_output=False
        ).fit_transform(gs)

        return gs.astype(np.uint8), ys.astype(int)

    def subsample(self, p, rng=None):
        """Subsample a portion p (in [0,1]) of the exhaustive dataset."""
        assert self.gvecs is not None
        assert self.ys is not None
        assert len(self.gvecs) == len(self.ys)

        if p != 1:
            rng = check_random_state(rng)
            n_examples = len(self.gvecs)
            n_keep = _prop_or_count(n_examples, p)
            pi = rng.permutation(n_examples)
            self.gvecs = np.array([self.gvecs[i] for i in pi[:n_keep]])
            self.ys = np.array([self.ys[i] for i in pi[:n_keep]])


class CNFDataset(Dataset):
    """Abstract class implementing a logical task over propositions."""

    def __init__(self, args):

        if args.from_cnf is not None:
            basename = ".".join(args.from_cnf.split(".")[:-1])
        else:
            basename = f'rng({args.n_variables},{args.n_clauses},{args.clause_length})'

        super().__init__(
            [2 for _ in range(self.n_variables)],
            f"cnf_{basename}"
        )

    def make_data(self):
        variables = exprvars("v", self.n_variables)
        formula = _bind(variables, self.clauses)

        def _infer(gvec):
            phi = And(*[
                ~variables[i] if g == 0 else variables[i]
                for i, g in enumerate(gvec)
            ])
            phi = formula & phi # already in CNF
            y = 1 if phi.satisfy_one() else 0

            return y

        self.gvecs, self.ys = self._make_all_data(_infer)

    def k(self, cvec, y):
        constraint = _bind([cvec[i] for i in range(1, len(cvec), 2)], self.clauses)
        return constraint if y else ~constraint

    def encode_background(self, A):
        return True


class RandomCNFDataset(CNFDataset):
    """Class implementing a random CNF."""

    def __init__(self, args):
        self.n_variables = args.n_variables
        self.n_clauses = args.n_clauses
        self.clause_length = args.clause_length
        self.clauses = self._sample_random_cnf(
            self.n_variables,
            self.n_clauses,
            self.clause_length,
            args.seed
        )

        '''
        print("Generated CNF:")
        for cl in self.clauses:
            print(" ".join(map(str, cl)))
        '''
            
        super().__init__(args)

    @staticmethod
    def _sample_random_cnf(n, m, k, rng):

        temp_vars = exprvars("v", n)

        def _nontrivial(curr, new):
            f1 = _bind(temp_vars, curr + [new])
            f2 = ~ _bind(temp_vars, [new])
            return (f1.satisfy_one() is not None) and \
                (f2.satisfy_one() is not None)
        
        rng = check_random_state(rng)
        clauses = []
        while len(clauses) < m:
            # NOTE: indices read from cnf files start from 1
            # we do the same
            indices = rng.choice(n, size=k) + 1
            signs = rng.choice([1, -1], size=len(indices))
            
            new_clause = list(indices * signs)
            if _nontrivial(clauses, new_clause):
                clauses.append(new_clause)

        return clauses #list(map(list, clauses))



class FileCNFDataset(CNFDataset):
    """Class implementing a custom CNF read from a DIMACS file."""

    def __init__(self, args):
        self.n_variables, self.clauses = _read_cnf(args.from_cnf)
        super().__init__(args)


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
    

class ConfigurableXOR(Dataset):
    def __init__(self, n_variables, cnf_path="configurable_xor"):
        super().__init__([2]*n_variables, cnf_path)
        self.n_variables = n_variables

    def make_data(self):
        # passiamo dall'esterno con load_data()
        pass

    def load_data(self, gs, ys):
        # gs e ys già pronti in one-hot
        self.gvecs = gs
        self.ys = ys

    def k(self, cvec, y):
        constraint = Xor(*[cvec[i] for i in range(1, len(cvec), 2)])
        # print("constraint")
        # print(constraint)
        return constraint if y else ~constraint

    def encode_background(self, A):
        return True


DATASETS = {
    "cnf": FileCNFDataset,
    "random": RandomCNFDataset,
    "xor": XorDataset,
    "configurable_xor": ConfigurableXOR
}


def _get_args_string(args):
    fields = [
        ("s", args.subsample),
        ("c", args.concept_sup),
        (None, args.seed),
    ]
    basename = '__'.join([
        name + '=' + str(value) if name else str(value)
        for name, value in fields
    ])
    return basename