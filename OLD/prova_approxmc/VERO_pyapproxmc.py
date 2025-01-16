#COPIA DI NEW-GEN-RSS.PY PER MODIFICHE

import argparse
import itertools as it
import operator as op
import numpy as np
import pickle

from abc import abstractmethod
from functools import reduce
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state

# We still import PyEda for building expressions and generating the CNF:
from pyeda.inter import exprvars, expr2dimacscnf
from pyeda.inter import And, Or, Xor, Implies, OneHot, Equal

# Now import pyapproxmc for approximate counting:
import pyapproxmc as pamc


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
    """Reads a CNF in DIMACS format (used by FileCNFDataset)."""
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

        super().__init__(
            [2 for _ in range(self.n_variables)],
            f"cnf_{basename}"
        )

    def make_data(self):
        variables = exprvars("v", self.n_variables)
        formula = _bind(variables, self.clauses)

        def _infer(gvec):
            # We build a partial assignment, then check if formula is satisfiable.
            # But we won't use formula.satisfy_one() from pyeda. Instead, 
            # we'll do a quick check by building a "mini" formula and 
            # seeing if a single solution exists.  Because we only 
            # need to label data, we can do a small check. But to avoid
            # a full-blown solver, let's just do a naive approach:
            
            # Build the partial assignment constraints:
            partial = And(*[
                (~variables[i] if g == 0 else variables[i])
                for i, g in enumerate(gvec)
            ])
            # Combine with main formula:
            check_formula = formula & partial
            # For labeling, we can still do a small pyeda call:
            return 1 if check_formula.satisfy_one() is not None else 0

        self.gvecs, self.ys = self._make_all_data(_infer)

    def k(self, cvec, y):
        # Constraint for each example, used in the top-level formula:
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
        # Helper to ensure the CNF is non-trivial
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
    """Class implementing a XOR task (for testing purposes)."""

    def __init__(self, args):
        super().__init__(
            [2 for _ in range(args.n_variables)],
            f"xor{args.n_variables}"
        )

    def make_data(self):
        # We'll do a small function to label data for XOR.
        xor = lambda x: list(it.accumulate(x, op.xor, initial=False))[-1]
        self.gvecs, self.ys = self._make_all_data(xor)

    def load_data(self, path):
        raise NotImplementedError()

    def k(self, cvec, y):
        constraint = Xor(*[cvec[i] for i in range(1, len(cvec), 2)])
        return constraint if y else ~constraint

    def encode_background(self, A):
        return True


DATASETS = {
    "cnf": FileCNFDataset,
    "random": RandomCNFDataset,
    "xor": XorDataset,
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


def main():
    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument(
        "dataset", choices=sorted(DATASETS.keys()),
        help="dataset to count RSs for"
    )
    parser.add_argument(
        "-s", "--subsample", type=float, default=1.0,
        help="fraction or number of observed gvecs to use (def. 1.0)"
    )
    parser.add_argument(
        "-c", "--concept-sup", type=float, default=0,
        help="fraction or number of gvecs with concept supervision (def. 0.0)"
    )
    parser.add_argument(
        "-D", "--print-data", action="store_true",
        help="print dataset prior to generating the CNF (def. False)",
    )
    parser.add_argument(
        "--store-litmap", action="store_true",
        help="Additionally store the mapping DIMACS indices -> variable names (def. False)",
    )
    parser.add_argument(
        "-E", "--enumerate", action="store_true",
        help="ApproxMC-based counting (replaces old pyeda enumeration).",
    )
    parser.add_argument(
        "-f", "--from-cnf", type=str, default=None,
        help="cnf dataset: read CNF from this"
    )
    parser.add_argument(
        "-n", "--n-variables", type=int, default=None,
        help="random, xor: number of variables (bits)"
    )
    parser.add_argument(
        "-m", "--n-clauses", type=int, default=None,
        help="random: number of clauses"
    )
    parser.add_argument(
        "-k", "--clause-length", type=int, default=None,
        help="random: clause length"
    )
    parser.add_argument(
        "--seed", type=int, default=1,
        help="RNG seed"
    )
    # Extra optional arguments for ApproxMC:
    parser.add_argument(
        "--epsilon", type=float, default=0.8,
        help="ApproxMC tolerance"
    )
    parser.add_argument(
        "--delta", type=float, default=0.2,
        help="ApproxMC confidence"
    )

    args = parser.parse_args()

    # Generate the dataset/task:
    print("Creating dataset")
    dataset = DATASETS[args.dataset](args)
    dataset.make_data()

    # Possibly subsample the labeled data:
    dataset.subsample(args.subsample, args.seed)
    n_examples = len(dataset.gvecs)
    n_csup = _prop_or_count(n_examples, args.concept_sup)
    pi = check_random_state(args.seed).permutation(n_examples)
    csup_mask = np.zeros(n_examples)
    csup_mask[pi[:n_csup]] = 1

    if args.print_data:
        print("Dataset gvecs:\n", dataset.gvecs)
        print("Dataset ys:\n", dataset.ys)

    print(f"Building formula: {len(dataset.gvecs)} gvecs, {dataset.n_bits} bits")

    # A: mapping from expanded inputs (bits) to compressed representation
    A = exprvars("A", dataset.n_bits, dataset.n_bits)
    # O: a discrete presence/absence of blocks in A, used for equivalences
    O = exprvars("O", dataset.n_variables, dataset.n_variables)

    # Build base formula for A & O:
    # A encodes a function C* -> C
    # each C* index is mapped into exactly one C index
    # O i-j is 1 iff A's block (i,j) is nonzero

    formula = And(*[OneHot(*O[:, k])
                    for k in range(dataset.n_variables)])

    # Helper to check if a 2x2 block is nonzero
    def nzb(k1, k2):
        return Or(
            A[k1*2, k2*2], A[k1*2, k2*2 + 1],
            A[k1*2 + 1, k2*2], A[k1*2 + 1, k2*2 + 1]
        )

    # O[k1, k2] <-> "the (k1,k2) block in A is non-zero"
    formula &= And(*[Equal(O[k1, k2], nzb(k1, k2))
                     for k1 in range(dataset.n_variables)
                     for k2 in range(dataset.n_variables)])

    # Each column of A must be OneHot
    formula &= And(*[OneHot(*A[:, i])
                     for i in range(dataset.n_bits)])

    # Encode extra symbolic background from the dataset
    formula &= dataset.encode_background(A)

    # Force the formula to achieve perfect performance on data
    for gvec, y, has_csup in zip(dataset.gvecs, dataset.ys, csup_mask):
        # Transform each original bit into the compressed bit via matrix A
        cvec = [
            _booldot(A[i, :], gvec).simplify()
            for i in range(len(gvec))
        ]
        # Enforce that cvec must sum to exactly one bit set per domain variable:
        offset = 0
        for vsize in dataset.domain_sizes:
            formula &= OneHot(*cvec[offset:offset+vsize])
            offset += vsize
        # Enforce the dataset's knowledge:
        formula &= dataset.k(cvec, y)
        # If we have concept supervision for this example:
        if has_csup:
            # This enforces that cvec[i] matches gvec[i] exactly
            for i in range(dataset.n_bits):
                formula &= (cvec[i] if gvec[i] else ~cvec[i])

    # Convert the formula to CNF and write to disk
    print("Converting formula to CNF...")
    litmap, cnf = expr2dimacscnf(formula.tseitin().to_cnf())

    cnf_path = f"{dataset.cnf_path}__{_get_args_string(args)}.cnf"
    print(f"Writing formula to {cnf_path}")
    with open(cnf_path, "wt") as fp:
        fp.write(str(cnf))

    # Optionally store the literal map
    if args.store_litmap:
        filtered_litmap = {
            str(k): str(v) for k, v in litmap.items()
            if isinstance(k, int) and "A" in str(v)
        }
        with open(cnf_path + ".litmap", "wb") as fp:
            pickle.dump(filtered_litmap, fp)

    # If the user asked for approximate counting, do it now (ApproxMC).
    if args.enumerate:
        print("\n--- ApproxMC counting step ---")
        print(f"Reading back {cnf_path} for approximate counting")
        with open(cnf_path, "rt") as fp:
            lines = list(map(str.strip, fp.readlines()))

        # The first line is the CNF header, then each subsequent line is a clause.
        counter = pamc.Counter(epsilon=args.epsilon, delta=args.delta, seed=args.seed)
        for line in lines[1:]:
            clause = [int(lit) for lit in line.split() if lit != '0']
            counter.add_clause(clause)
        count = counter.count()
        total = count[0] * 2**count[1]
        print(f"\nApprox. # of solutions: {count[0]} * 2**{count[1]} = {total}\n")


if __name__ == "__main__":
    main()