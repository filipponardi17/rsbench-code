import argparse
import itertools as it
import operator as op
import numpy as np
from functools import reduce
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state
from pyeda.inter import exprvars, expr2dimacscnf, And, Or, Xor, OneHot, Equal

def _pp_solution(sol, nvars, nbits):
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
    #print(f"whatever p is: {p}")
    return int(p if p > 1 else np.trunc(n * p))

def _booldot(avec, bvec):
    return reduce(op.or_, [a & b for a, b in zip(avec, bvec)])

class Dataset:
    def __init__(self, domain_sizes, cnf_path):
        self.domain_sizes = domain_sizes
        self.cnf_path = cnf_path
        self.gvecs, self.ys = None, None
        self.n_variables = len(domain_sizes)
        self.n_bits = sum(domain_sizes)

    def _make_all_data(self, infer):
        gs = list(it.product(*[list(range(size)) for size in self.domain_sizes]))
        # print("primo gs")
        # print(gs)

        ys = [infer(g) for g in gs]
        # print("primo ys")
        # print(ys)
        gs, ys = np.array(gs), np.array(ys)

        gs = OneHotEncoder(categories=[list(range(size)) for size in self.domain_sizes],
                           sparse_output=False).fit_transform(gs)
        
        # print("secondo gs")
        # print(gs)
        return gs.astype(np.uint8), ys.astype(int)

    def subsample(self, p, rng=None):
        # print(f"Proporzione di sottocampionamento richiesta: {p}")
        if p != 1:
            rng = check_random_state(rng)
            n_examples = len(self.gvecs)
            n_keep = _prop_or_count(n_examples, p)
            pi = rng.permutation(n_examples)
            self.gvecs = np.array([self.gvecs[i] for i in pi[:n_keep]])
            self.ys = np.array([self.ys[i] for i in pi[:n_keep]])

class XorDataset(Dataset):
    def __init__(self, args):
        super().__init__([2 for _ in range(args.n_variables)], f"xor{args.n_variables}")

    def make_data(self):
        xor = lambda x: list(it.accumulate(x, op.xor, initial=False))[-1]
        self.gvecs, self.ys = self._make_all_data(xor)

    def load_data(self, path):
        pass

    def k(self, cvec, y):
        constraint = Xor(*[cvec[i] for i in range(1, len(cvec), 2)])
        return constraint if y else ~constraint

    def encode_background(self, A):
        return True

DATASETS = {
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
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS.keys()))
    parser.add_argument("-s", "--subsample", type=float, default=1.0)
    parser.add_argument("-c", "--concept-sup", type=float, default=0)
    parser.add_argument("-E", "--enumerate", action="store_true")
    parser.add_argument("-n", "--n-variables", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    dataset = DATASETS[args.dataset](args)
    dataset.make_data()
    dataset.subsample(args.subsample, args.seed)

    n_examples = len(dataset.gvecs)
    n_csup = _prop_or_count(n_examples, args.concept_sup)
    pi = check_random_state(args.seed).permutation(n_examples)
    csup_mask = np.zeros(n_examples)
    csup_mask[pi[:n_csup]] = 1

    A = exprvars("A", dataset.n_bits, dataset.n_bits)
    O = exprvars("O", dataset.n_variables, dataset.n_variables)

    formula = And(*[OneHot(*O[:, k]) for k in range(dataset.n_variables)])
    print("formula")
    print(formula)  
    


    nzb = lambda k1,k2 : Or(A[k1*2, k2*2], A[k1*2, k2*2 + 1],
                           A[k1*2 + 1, k2*2], A[k1*2 + 1, k2*2 + 1])
    
    print("nzb")
    print(nzb)   

    formula &= And(*[Equal(O[k1, k2], nzb(k1, k2))
                     for k1 in range(dataset.n_variables)
                     for k2 in range(dataset.n_variables)])
    formula &= And(*[OneHot(*A[:, i]) for i in range(dataset.n_bits)])
    formula &= dataset.encode_background(A)

    for gvec, y, has_csup in zip(dataset.gvecs, dataset.ys, csup_mask):
        # print("has_csup")
        # print(has_csup)
        # print("A")
        # print(A)
        cvec = [_booldot(A[i, :], gvec).simplify() for i in range(len(gvec))]
        # print("cvec")
        # print(cvec)
        offset = 0
        # print("dataset.domain_sizes")
        # print(dataset.domain_sizes)
        for vsize in dataset.domain_sizes:
            # print("vsize")
            # print(vsize)
            # print("offset")
            # print(offset)
            # print("gvec")
            # print(gvec)
            # print("A")
            # print(A)
            # print("cvec[offset:offset+vsize]")
            # print(cvec[offset:offset+vsize])
            formula &= OneHot(*cvec[offset:offset+vsize])
            # print("formula")
            #print(formula)
            offset += vsize
        
        # print("dataset.k(cvec, y)")
        # print(dataset.k(cvec, y))
        formula &= dataset.k(cvec, y)
        if has_csup:
            for i in range(dataset.n_bits):
                formula &= cvec[i] if gvec[i] else ~cvec[i]

    if args.enumerate:
        print("args.enumerate")
        n_sol = 0
        for sol in formula.satisfy_all():
            print(f"Solution {n_sol}")
            _pp_solution(sol, dataset.n_variables, dataset.n_bits)
            n_sol += 1                
        print(f"{n_sol} solutions")

if __name__ == "__main__":
    main()