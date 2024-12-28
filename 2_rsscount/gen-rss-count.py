import argparse
import pickle
import sys
from pyeda.inter import exprvars, Xor, OneHot, And, expr2dimacscnf

# Classe ConfigurableXOR minimale
class ConfigurableXOR:
    def __init__(self, args):
        self.n_variables = args.n_variables
        self.data_path = args.data_path
        self.gvecs = None
        self.ys = None

    def make_data(self):
        # Carica gvecs, ys da pickle
        if self.data_path is None:
            raise ValueError("Devi specificare --data-path.")
        with open(self.data_path, "rb") as f:
            self.gvecs, self.ys = pickle.load(f)

    def k(self, cvec, y):
        # XOR dei n_variables bit (ognuno codificato in 2 bit one-hot)
        # cvec ha dimensione n_bits = n_variables * 2
        # Estraiamo la vera variabile come XOR su 2 bit one-hot?
        # In realtà l'XOR del dataset originale era definito come:
        # cvec[i] rappresenta un bit one-hot. Per ogni variabile due bit: cvec[2i], cvec[2i+1].
        # La variabile è 1 se cvec[2i+1] è True, altrimenti 0.
        # Quindi XOR = Xor su i=0..n_variables-1 del bit cvec[2*i+1].
        bits = []
        for i in range(self.n_variables):
            # Il bit variabile = 1 se cvec[2*i+1] è True
            bits.append(cvec[2*i+1])  
        constraint = Xor(*bits)
        return constraint if y == 1 else ~constraint

    def n_bits(self):
        return self.n_variables * 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Dataset name (must be 'configxor')")
    parser.add_argument("--data-path", type=str, default=None, help="Path to pickle with (gvecs, ys)")
    parser.add_argument("-n", "--n-variables", type=int, default=3, help="Number of XOR variables")
    parser.add_argument("-E", "--enumerate", action="store_true", help="Enumerate solutions")
    args = parser.parse_args()

    if args.dataset != "configxor":
        print("Errore: questo script supporta solo dataset='configxor'")
        sys.exit(1)

    # Crea dataset
    dataset = ConfigurableXOR(args)
    dataset.make_data()

    n_vars = dataset.n_variables
    n_bits = dataset.n_bits()

    # Variabili A: Mappa dai bit in input (gvec) al concetto cvec
    # A ha dimensione n_bits x n_bits (vedi codice originale)
    A = exprvars("A", n_bits, n_bits)

    # Costruiamo la formula
    # Regole:
    # 1. Ogni colonna di A è OneHot => definisce come ogni bit di input mappa a un bit di concetto
    formula = And(*[OneHot(*A[:, i]) for i in range(n_bits)])

    # Non abbiamo bisogno di O stavolta, semplifichiamo la formula.
    # Nel codice originale c'era una struttura complessa, ma per XOR bastano le regole di A?
    # Nel codice originale, O e A erano usati per mappare C* -> C. Possiamo mantenerle minime:
    # Se vogliamo mantenerci coerenti col codice originale, lasciamo la formula invariata dove possibile.

    # Ogni esempio (gvec, y): costruiamo cvec come combinazione booleana:
    # cvec[i] = OR su tutti i bit di gvec con A corrispondenti.
    # gvec è lungo n_bits. Per il bit i in cvec, cvec[i] = OR di (A[i,j] AND gvec[j]) su j.
    # Implementiamo cvec[i] come una disgiunzione di (A[i,j] & gvec[j]) su j.
    # gvec[j] è un valore True/False statico -> lo implementiamo come costante booleana?
    # In pyeda possiamo usare costanti True/False per gvec bits.

    from pyeda.inter import exprvar, Or as Or_, And as And_

    # Convert gvec bits in boolean exprvar "G_j" constants
    # In realtà non servono variabili, bastano costanti True/False:
    # True = exprvar('T') == Non serve. pyeda usa True/False python
    # useremo solo python True/False, non c'è bisogno di variabili per gvec.
    
    # cvec[i] = OR_j (A[i,j] if gvec[j]==1)
    # se gvec[j]==0, quel termine non contribuisce
    # se nessun gvec[j]==1 per colonna i? Non succede in XOR dataset one-hot perché ogni variabile ha esattamente un bit a 1.

    # Aggiungiamo i vincoli sugli esempi per ottenere performance perfetta
    for gvec, y in zip(dataset.gvecs, dataset.ys):
        # costruiamo cvec
        cvec = []
        for i in range(n_bits):
            terms = []
            for j in range(n_bits):
                if gvec[j] == 1:
                    terms.append(A[i, j])
            # Se nessun j attivo, cvec[i] = False
            cvec_i = Or_(*terms) if terms else exprvar("FALSE_CONST") # termine fittizio
            cvec.append(cvec_i)

        # Ogni variabile codificata su due bit: OneHot
        # Forziamo OneHot per ogni coppia di bit cvec:
        for v in range(n_vars):
            formula &= OneHot(cvec[2*v], cvec[2*v+1])
        
        # Vincolo XOR
        formula &= dataset.k(cvec, y)

    # Convertiamo in CNF
    cnf = formula.to_cnf()
    # Se enumerate: enumeriamo le soluzioni
    if args.enumerate:
        n_sol = 0
        for sol in cnf.satisfy_all():
            n_sol += 1
        print(f"{n_sol} solutions")
    else:
        # Se non enumerate, stampiamo solo la formula in CNF in DIMACS?
        # Non è richiesto. Ma per sicurezza stampiamo in DIMACS:
        dimacs = expr2dimacscnf(cnf)
        print(dimacs)

if __name__ == "__main__":
    main()