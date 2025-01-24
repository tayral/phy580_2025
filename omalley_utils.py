import numpy as np
import csv
from qat.core import Observable, Term

def load_omalley_data_unordered(filename, keep_Z0Z1=True):
    """
    Args:
        keep_Z0Z1 (bool): whether to keep Z0Z1 term in Hamiltonian. If not, add <HF|g3 Z0Z1 |HF> = -g3
            to constant energy (|HF> = |01>).
    """
    data = np.loadtxt(filename, usecols=(0, 1, 2, 3, 4, 5, 6, 7))

    terms = [('Z', [0]), ('Z', [1]), ('ZZ', [0, 1]), ('XX', [0, 1]), ('YY', [0, 1])]

    distances = data[:, 0]
    hamilt = {}
    qpe_t0 = {}
    for row, d in enumerate(distances):
        if keep_Z0Z1:
            hamilt[d] = Observable(2, pauli_terms = [
                Term(data[row, 2+col], ops, qbits) for col, (ops, qbits) in enumerate(terms)
            ],
                                  constant_coeff=data[row, 1])
        else:
            hamilt[d] = Observable(2, pauli_terms = [
                Term(data[row, 2+col], ops, qbits) for col, (ops, qbits) in enumerate(terms) if col !=2
            ],
                                  constant_coeff=data[row, 1] - data[row, 2+2])
        qpe_t0[d] = data[row, 7]
        """
        if row==0:
            print(data[row, :])
            print(hamilt[d])
        """
    return hamilt, qpe_t0
        
def load_omalley_data_ordered(filename, keep_Z0Z1=True):
    """
    Args:
        filename (str): csv file name
    Returns:
        dict : distance : Hamiltonian
        dict : distance : QPE's t0 time
    """
    
    datafile = open(filename, 'r')
    myreader = csv.reader(datafile)

    orderings = []
    for line_no, row in enumerate(myreader):
        if line_no > 1:
            data = row[0].split(' ')
            t_list = []
            for t in range(4):
                t_list.append(data[8+t])
            #print(t_list)
            orderings.append(t_list)

    terms = [('Z', [0]), ('Z', [1]), ('ZZ', [0, 1]), ('XX', [0, 1]), ('YY', [0, 1])]
    coeff_ordering = {k : v for k, v in zip(['Z0', 'Z1', 'Z0Z1', 'X0X1', 'Y0Y1'], range(5))}
    data = np.loadtxt(filename, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
    distances = data[:, 0]
    hamilt = {}
    qpe_t0 = {}
    def conv_format(term_list):
        """from [X0Y1, Z4] to [("XY", [0,1]), ("Z", [4])]"""
        n_term_list = []
        for t in term_list:
            op = ""
            qbits = []
            for i in range(len(t)//2):
                op += t[2*i]
                qbits.append(int(t[2*i+1]))
            n_term_list.append((op, qbits))
        return n_term_list

    for row, d in enumerate(distances):
        if keep_Z0Z1:
            term_list = ['Z0Z1']+orderings[row]  # adding Z0Z1 because not in trotter sequence
        else:
            term_list = orderings[row]  # adding Z0Z1 because not in trotter sequence
        n_term_list = conv_format(term_list)
        hamilt[d] = Observable(2, pauli_terms = [
            Term(data[row, 2+coeff_ordering[name]], ops, qbits)
            for col, (name, (ops, qbits)) in enumerate(zip(term_list,n_term_list))
        ],
                              constant_coeff=data[row, 1])
        if not keep_Z0Z1:
            hamilt[d].constant_coeff -= data[row, 2+ 2]
        qpe_t0[d] = data[row, 7]
        """
        if row==0:
            print(data[row, :])
            print(term_list)
            print(n_term_list)
            print(hamilt[d])
        """
    return hamilt, qpe_t0

def find_binary_decomp(x, nmax=10):
    """x is 0<x<1, find decomp x = 0.x1x2x3...xm with m=nbits
    
    Args:
        x (float): a scalar
        nbits (int): the number of bits of precision
            
    """
    nbits = 1
    tol = 1e-12
    while nbits < nmax:
        y = 2**nbits * x
        ytilde = np.floor(y)
        bin_repr = "0."+str(bin(int(ytilde)))[2:2+nbits]
        if abs(y-ytilde)<tol:
            print("done (%s bits)"%nbits)
            break
        nbits += 1
    return bin_repr
