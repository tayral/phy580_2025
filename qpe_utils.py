import numpy as np
from qat.lang.AQASM.qftarith import IQFT
from qat.lang.AQASM import H, QRoutine, RX, Program
from qat.lang.AQASM import CNOT, RZ



def construct_Rk_routine(ops, qbits, theta):
    r"""Implements the quantum routine
    
    .. math::
         R_k(\theta) = \exp\left(-i \frac{\theta}{2} P_k\right)
         
    with P_k a Pauli string
    
    Args:
        ops (str): Pauli operators (e.g X, Y, ZZ, etc.)
        qbits (list<int>): qubits on which they act
        theta (Variable): the abstract variable
        
    Returns:
        QRoutine
        
    Notes:
        the indices of the wires of the QRoutine are relative
        to the smallest index in qbits (i.e always start at qb=0)
    """
    min_qb = min(qbits)
    qbits = [qb - min_qb for qb in qbits]  # everything must be defined relative to 0
    qrout = QRoutine()
    with qrout.compute():
        for op, qbit in zip(ops, qbits):
            if op == "X":
                qrout.apply(H, qbit)
            if op == "Y":
                qrout.apply(RX(np.pi/2), qbit)
        for ind_qb in range(len(qbits)-1):
            qrout.apply(CNOT, qbits[ind_qb], qbits[ind_qb+1])
    qrout.apply(RZ(theta), qbits[-1])
    # uncompute() applies U^dagger,
    # with U the unitary corresponding to the gates applied within the "with XX.compute()" context
    qrout.uncompute()
    
    return qrout


def make_controlled_exp_hamiltonian(hamiltonian, t0, n_trotter, order):
    """contruct controlled-U with :math:`U = \exp(-i H t0)`

    
    Args:
        hamiltonian (Observable): the Hamiltonian H
        t0 (float): the evolution time
        n_trotter (int): the number of Trotter slices
        order (int): the Trotterization order (1 or 2)

    Returns:
        QRoutine: the U routine
    """
    routine = QRoutine()
    anc_reg = routine.new_wires(1)
    data_reg = routine.new_wires(hamiltonian.nbqbits)
    for _ in range(n_trotter):
        if order == 1:
            for term in hamiltonian.terms:
                theta = 2 * term.coeff * t0 / n_trotter
                Rk_routine = construct_Rk_routine(term.op, term.qbits, theta)
                try:
                    routine.apply(Rk_routine.ctrl(), anc_reg,
                                  [data_reg[qb] for qb in term.qbits])
                except:
                    print(term.qbits)
                    raise
        elif order == 2:
            for term in hamiltonian.terms:
                theta = term.coeff * t0 / n_trotter
                Rk_routine = construct_Rk_routine(term.op, term.qbits, theta)
                try:
                    routine.apply(Rk_routine.ctrl(), anc_reg,
                                  [data_reg[qb] for qb in term.qbits])
                except:
                    print(term.qbits)
                    raise
            for term in reversed(hamiltonian.terms):
                theta = term.coeff * t0 / n_trotter
                Rk_routine = construct_Rk_routine(term.op, term.qbits, theta)
                try:
                    routine.apply(Rk_routine.ctrl(), anc_reg,
                                  [data_reg[qb] for qb in term.qbits])
                except:
                    print(term.qbits)
                    raise
    return routine
