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
import pyapproxmc as pamc


def _pp_solution(sol, nvars, nbits):
    """(Kept for compatibility) Pretty-print a pyeda model."""
    Asol = np.zeros(shape=(nbits, nbits), dtype=np.int16)
    Osol = np.zeros(shape=(nvars, nvars), dtype=np.int16)
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


def count_rss(dataset, epsilon=0.8, delta=0.2, seed=1):
    """
    Build the RS-encoding formula for `dataset`, then do an *approximate count*
    of the solutions with pyapproxmc.

    Parameters
    ----------
    dataset : Dataset
        The task dataset with .gvecs, .ys, etc.
    epsilon : float, optional
        ApproxMC tolerance; default is 0.8
    delta : float, optional
        ApproxMC confidence; default is 0.2
    seed : int, optional
        RNG seed for ApproxMC; default is 1

    Returns
    -------
    n_sol : int (approx.)
        The approximate number of solutions returned by ApproxMC.
    """

    print(f"Building formula: {len(dataset.gvecs)} gvecs, {dataset.n_bits} bits")

    # Variables A and O
    A = exprvars("A", dataset.n_bits, dataset.n_bits)
    O = exprvars("O", dataset.n_variables, dataset.n_variables)

    # Each row in O is a OneHot
    formula = And(*[OneHot(*O[:, k]) for k in range(dataset.n_variables)])

    # We say O[k1, k2] = 1 iff the block (k1,k2) in A is nonzero
    def nzb(k1, k2):
        return Or(
            A[k1 * 2, k2 * 2], A[k1 * 2, k2 * 2 + 1],
            A[k1 * 2 + 1, k2 * 2], A[k1 * 2 + 1, k2 * 2 + 1]
        )

    formula &= And(*[
        Equal(O[k1, k2], nzb(k1, k2))
        for k1 in range(dataset.n_variables)
        for k2 in range(dataset.n_variables)
    ])

    # Each column in A is OneHot
    formula &= And(*[OneHot(*A[:, i]) for i in range(dataset.n_bits)])

    # Possibly encode extra symbolic background
    formula &= dataset.encode_background(A)

    # Force perfect performance on each example
    for gvec, y in zip(dataset.gvecs, dataset.ys):
        cvec = [_booldot(A[i, :], gvec).simplify()
                for i in range(len(gvec))]
        # For each variable domain, ensure exactly one bit in cvec
        offset = 0
        for vsize in dataset.domain_sizes:
            formula &= OneHot(*cvec[offset:offset + vsize])
            offset += vsize

        # The actual label constraint
        formula &= dataset.k(cvec, y)

    # Convert the final expression to a CNF in DIMACS form
    print("Converting formula to CNF (for approximate counting)...")
    litmap, pyeda_cnf = expr2dimacscnf(formula.tseitin().to_cnf())

    # We'll parse the CNF lines to feed pyapproxmc
    cnf_str = str(pyeda_cnf)    # The CNF in DIMACS string form
    cnf_lines = cnf_str.split('\n')

    # Prepare an ApproxMC counter
    counter = pamc.Counter(epsilon=epsilon, delta=delta, seed=seed)

    # The first line is the 'p cnf <vars> <clauses>' header
    # Each subsequent line is a clause ending in 0
    # We'll parse them into integer literals
    for clause_line in cnf_lines[1:]:
        line = clause_line.strip()
        if not line:
            continue
        clause_tokens = line.split()
        # We expect a trailing '0'
        # Filter out the final 0
        lits = [int(t) for t in clause_tokens if t != '0']
        if not lits:
            continue
        counter.add_clause(lits)

    # Perform approximate counting
    res = counter.count()
    # res = (count, pivot_bits), total = count * 2**pivot_bits
    n_sol = res[0] * 2**res[1]

    print(f"Approx. number of solutions for this task: {n_sol}")
    return n_sol


# ---------------------------------------------------------------------
# Dataset classes remain exactly the same so that nothing breaks when
# called from multiple_tasks.py
# ---------------------------------------------------------------------
class Dataset:
    """Abstract Dataset(task) class."""

    def __init__(self, domain_sizes, cnf_path):
        self.domain_sizes = domain_sizes
        self.cnf_path = cnf_path
        self.gvecs, self.ys = None, None
        self.n_variables = len(domain_sizes)
        self.n_bits = sum(domain_sizes)  # n bits per variable

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
        """Generates all possible ground-truth concept vectors and labels."""
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
        super().__init__([2 for _ in range(self.n_variables)], f"cnf_{basename}")

    def make_data(self):
        variables = exprvars("v", self.n_variables)
        formula = _bind(variables, self.clauses)

        def _infer(gvec):
            # Build partial assignment constraints
            phi = And(*[
                ~variables[i] if g == 0 else variables[i]
                for i, g in enumerate(gvec)
            ])
            phi = formula & phi  # still a pyeda expression
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
        super().__init__(args)

    @staticmethod
    def _sample_random_cnf(n, m, k, rng):
        temp_vars = exprvars("v", n)

        def _nontrivial(curr, new):
            f1 = _bind(temp_vars, curr + [new])
            f2 = ~_bind(temp_vars, [new])
            return (f1.satisfy_one() is not None) and \
                   (f2.satisfy_one() is not None)

        rng = check_random_state(rng)
        clauses = []
        while len(clauses) < m:
            indices = rng.choice(n, size=k) + 1
            signs = rng.choice([1, -1], size=len(indices))
            new_clause = list(indices * signs)
            if _nontrivial(clauses, new_clause):
                clauses.append(new_clause)
        return clauses


class FileCNFDataset(CNFDataset):
    """Class implementing a custom CNF read from a DIMACS file."""

    def __init__(self, args):
        self.n_variables, self.clauses = _read_cnf(args.from_cnf)
        super().__init__(args)


class XorDataset(Dataset):
    """Class implementing an XOR task (for testing)."""

    def __init__(self, args):
        super().__init__([2 for _ in range(args.n_variables)], f"xor{args.n_variables}")

    def make_data(self):
        xor = lambda x: list(it.accumulate(x, op.xor, initial=False))[-1]
        self.gvecs, self.ys = self._make_all_data(xor)

    def load_data(self, path):
        raise NotImplementedError()

    def k(self, cvec, y):
        constraint = Xor(*[cvec[i] for i in range(1, len(cvec), 2)])
        return constraint if y else ~constraint

    def encode_background(self, A):
        return True


class ConfigurableXOR(Dataset):
    """
    Same as XorDataset except we fill the gvecs, ys externally
    via load_data(...).
    """
    def __init__(self, n_variables, cnf_path="configurable_xor"):
        super().__init__([2] * n_variables, cnf_path)
        self.n_variables = n_variables

    def make_data(self):
        # For ConfigurableXOR, data is loaded externally
        pass

    def load_data(self, gs, ys):
        # gs and ys are already one-hot + integer labels
        self.gvecs = gs
        self.ys = ys

    def k(self, cvec, y):
        constraint = Xor(*[cvec[i] for i in range(1, len(cvec), 2)])
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