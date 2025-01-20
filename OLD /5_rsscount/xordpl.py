#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


def _pp_solution(sol, nvars, nbits):
    """Pretty-print a pyeda model. Most relevant: extract the matrix A e O."""
    Asol = np.zeros(shape=(nbits, nbits), dtype=np.int16)
    Osol = np.zeros(shape=(nvars, nvars), dtype=np.int16)

    for k in sol:
        if k.name == 'A' and sol[k]:
            Asol[k.indices] = 1
        elif k.name == 'O' and sol[k]:
            Osol[k.indices] = 1

    print("Matrix A (mapping bits -> bits):\n", Asol)
    print("\nMatrix O (check for non-zero blocks in A):\n", Osol)
    return Asol, Osol


def _prop_or_count(n, p):
    """Dato n, se p <= 1.0 interpreta p come frazione, altrimenti come count intero."""
    return int(p if p > 1 else np.trunc(n * p))


def _booldot(avec, bvec):
    """Boolean dot product (OR dei prodotti bit a bit)."""
    return reduce(op.or_, [a & b for a, b in zip(avec, bvec)])


def _bind(variables, clauses):
    """Binds clauses-of-ints to pyeda variables (per CNF random)."""
    return And(*[
        Or(*[
            variables[int(i) - 1] if i > 0 else ~variables[-int(i) - 1]
            for i in clause
        ])
        for clause in clauses
    ])


def _read_cnf(path):
    """Legge un file CNF in formato DIMACS."""
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
        self.n_bits = sum(domain_sizes)  # es: se ho 3 variabili booleane => 3*2=6

    @abstractmethod
    def make_data(self):
        pass

    @abstractmethod
    def load_data(self, path):
        pass

    @abstractmethod
    def k(self, cvec, y):
        """Knowledge base constraint: cvec => y."""
        pass

    def _make_all_data(self, infer):
        """Genera TUTTE le combinazioni di input e poi calcola label con la function infer(...)."""
        gs = list(it.product(*[list(range(size)) for size in self.domain_sizes]))
        ys = [infer(g) for g in gs]
        gs, ys = np.array(gs), np.array(ys)

        # One-hot encoding delle combinazioni
        gs = OneHotEncoder(
            categories=[list(range(size)) for size in self.domain_sizes],
            sparse_output=False
        ).fit_transform(gs)

        return gs.astype(np.uint8), ys.astype(int)

    def subsample(self, p, rng=None):
        """Prende solo p (frazione o conteggio) di esempi dal dataset esaustivo."""
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
    """Abstract class per tasks basati su CNF."""

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
            # Converte gvec in un'assegnazione booleana e controlla se formula è soddisfatta
            phi = And(*[
                ~variables[i] if g == 0 else variables[i]
                for i, g in enumerate(gvec)
            ])
            phi = formula & phi
            y = 1 if phi.satisfy_one() else 0
            return y

        self.gvecs, self.ys = self._make_all_data(_infer)

    def k(self, cvec, y):
        constraint = _bind([cvec[i] for i in range(1, len(cvec), 2)], self.clauses)
        return constraint if y else ~constraint

    def encode_background(self, A):
        return True


class RandomCNFDataset(CNFDataset):
    """CNF random generato a run-time."""

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
            f2 = ~ _bind(temp_vars, [new])
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
    """CNF letto da file DIMACS."""

    def __init__(self, args):
        self.n_variables, self.clauses = _read_cnf(args.from_cnf)
        super().__init__(args)


class XorDataset(Dataset):
    """Implementa il task XOR su n variabili booleane."""

    def __init__(self, args):
        super().__init__([2 for _ in range(args.n_variables)], f"xor{args.n_variables}")

    def make_data(self):
        # XOR di n bit => label = 1 se # di bit a 1 è dispari, 0 altrimenti
        xor = lambda bits: reduce(op.xor, bits, 0)
        self.gvecs, self.ys = self._make_all_data(xor)

    def load_data(self, path):
        raise NotImplementedError()

    def k(self, cvec, y):
        # Costringe Xor(cvec) = y
        # cvec è una lista di lit booleani di pyeda, uno per bit
        constraint = Xor(*[cvec[i] for i in range(len(cvec))])
        return constraint if y else ~constraint

    def encode_background(self, A):
        """In un setting DPL potresti aggiungere vincoli sulle entità, qui no."""
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


# ------------- FUNZIONI PER IL CONTEGGIO REASONING SHORTCUTS ------------- #
def compute_shortcuts_coverage(Asol):
    """
    Esempio di metrica 'coverage' su A:
    - Asol è una matrice binaria (nbits x nbits).
    - Se alcune colonne/righe sono sempre zero, significa che
      parte delle feature non sono mai utilizzate (shortcut).
    - Ritorna coverage in [0,1]: 1 => nessuno shortcut, 0 => collasso totale.
    """
    # coverage = frazione di colonne che contengono almeno un '1'
    used_cols = 0
    nbits = Asol.shape[1]
    for col_idx in range(nbits):
        if np.any(Asol[:, col_idx] == 1):
            used_cols += 1
    coverage = used_cols / nbits
    return coverage


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
        help="Additionally stores the mapping DIMACS indices -> variable names (def. False)",
    )
    parser.add_argument(
        "-E", "--enumerate", action="store_true",
        help="enumerate solutions (def. False)",
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
    args = parser.parse_args()

    # Generazione del dataset
    print("Creating dataset")
    dataset = DATASETS[args.dataset](args)
    dataset.make_data()

    cnf_path = f"{dataset.cnf_path}__{_get_args_string(args)}.cnf"

    # Subsample del dataset
    dataset.subsample(args.subsample, args.seed)

    n_examples = len(dataset.gvecs)
    n_csup = _prop_or_count(n_examples, args.concept_sup)
    pi = check_random_state(args.seed).permutation(n_examples)
    csup_mask = np.zeros(n_examples)
    csup_mask[pi[:n_csup]] = 1

    if args.print_data:
        print(dataset.gvecs)
        print(dataset.ys)

    print(f"Building formula: {len(dataset.gvecs)} gvecs, {dataset.n_bits} bits")

    # Dichiarazione variabili booleane:
    # A: matrice (n_bits x n_bits) => definisce la mappa "C* -> C" (concetti latenti)
    # O: matrice (n_variables x n_variables) => maschera per "non-zero block"
    A = exprvars("A", dataset.n_bits, dataset.n_bits)
    O = exprvars("O", dataset.n_variables, dataset.n_variables)

    # 1) Each row di O deve avere esattamente 1 bit a 1 => OneHot
    formula = And(*[OneHot(*O[:, k])
                    for k in range(dataset.n_variables)])

    # 2) Se O[k1,k2] = 1 => blocco (k1,k2) in A non è zero
    #    Se O[k1,k2] = 0 => blocco in A è zero
    def nzb(k1, k2):
        return Or(A[k1*2, k2*2], A[k1*2, k2*2 + 1],
                  A[k1*2 + 1, k2*2], A[k1*2 + 1, k2*2 + 1])

    formula &= And(*[Equal(O[k1, k2], nzb(k1, k2))
                     for k1 in range(dataset.n_variables)
                     for k2 in range(dataset.n_variables)])

    # 3) Each colonna di A (cioè ogni bit in input) deve essere mappata a exactly 1 row => OneHot
    formula &= And(*[OneHot(*A[:, i])
                     for i in range(dataset.n_bits)])

    # 4) eventuale "background knowledge"
    formula &= dataset.encode_background(A)

    # 5) Perfect performance su data => costringe la rete (A) a rispettare la label
    #    e se c'è concept sup, costringe i bit a combaciare
    for gvec, y, has_csup in zip(dataset.gvecs, dataset.ys, csup_mask):
        cvec = [_booldot(A[i, :], gvec).simplify() for i in range(len(gvec))]
        # Forza i bit cvec (nella nuova "base") a definire label = y
        offset = 0
        for vsize in dataset.domain_sizes:
            formula &= OneHot(*cvec[offset:offset+vsize])
            offset += vsize

        formula &= dataset.k(cvec, y)

        # concept supervision: se l'esempio rientra nei "csup_mask",
        # forziamo i bit a matchare gvec
        if has_csup:
            for i in range(dataset.n_bits):
                formula &= cvec[i] if gvec[i] else ~cvec[i]

    # Converte e salva formula in DIMACS
    print("Converting formula to CNF...")
    litmap, cnf = expr2dimacscnf(formula.tseitin().to_cnf())

    print(f"Writing formula to {cnf_path}")
    with open(cnf_path, "wt") as fp:
        fp.write(str(cnf))

    if args.store_litmap:
        litmap = {
            str(k): str(v) for k, v in litmap.items()
            if type(k) is int and "A" in str(v)
        }
        with open(cnf_path + ".litmap", "wb") as fp:
            pickle.dump(litmap, fp)

    # Se richiesto, enumeriamo tutte le soluzioni => occhio che può esplodere combinatorialmente!
    if args.enumerate:
        n_sol = 0
        total_coverage = 0.0
        for sol in formula.satisfy_all():
            print("\n=== SOL n.", n_sol, "===")
            Asol, _ = _pp_solution(sol, dataset.n_variables, dataset.n_bits)
            coverage = compute_shortcuts_coverage(Asol)
            total_coverage += coverage
            print(f"Coverage (meno scorciatoie => più alto) = {coverage:.2f}")
            n_sol += 1

        print(f"\nTrovate {n_sol} soluzioni totali.")
        if n_sol > 0:
            print(f"Coverage medio: {total_coverage / n_sol:.3f}")


if __name__ == "__main__":
    main()