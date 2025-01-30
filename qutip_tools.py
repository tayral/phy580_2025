#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file qat/analog/qutip_tools.py
@authors Grigori Matein <grigori.matein@atos.net>
         Corentin Bertrand <corentin.bertrand@eviden.com>
@internal
@copyright 2021 Bull S.A.S. - All rights reserved.
           This is not Free or Open Source software.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois
@brief Tools to translate QLM objects to qutip
"""
import re
import copy
import inspect
from typing import List, Optional
import numpy as np

import qat.comm.exceptions.ttypes as exceptions_types
from qat.core import Observable, Term
from qat.core.variables import ArithExpression

from hilbert_space import HilbertSpace

module_name = "qat.analog"


class QutipObservable:

    def __init__(self, data=None):
        self.data = data or []

    def add(self, other):
        self.data.extend(other.data)

    def project_subspace(self, indices: List[int]):
        new_data = []
        for datum in self.data:
            if isinstance(datum, list):
                new_data.append([project_subspace(datum[0], indices), datum[1]])
            else:
                new_data.append(project_subspace(datum, indices))
        self.data = new_data


def project_subspace(qobj, indices: List[int]):
    """
    Project a qutip.QObj into subspace given by indices.
    Analoguous to old Qobj.extract_states from qutip < 5.0
    """
    import qutip as qt
    if qobj.isket:
        return qt.Qobj(qobj.full()[indices, :])
    if qobj.isoper:
        return qt.Qobj(qobj.full()[indices, :][:, indices])

    raise RuntimeError(f"Qutip object type {qobj.type} is not supported.")


def convert_obs_to_qutip(observable, hilbert_space: HilbertSpace):
    """
    Convert a QLM Observable to a Qutip observable.
    The QLM Observable should already be evaluated, i.e. no ArithExpression-s
    inside it (not even 't').
    The Observable, not being abstract at all, can have many Terms, all with
    int or double coefficients.

    Args:
        observable (Observable): an observable
        hilbert_space (HilbertSpace): object representing the Hilbert space structure in term of qubits, fermions and/or bosons.

    Returns:
        qutip.QObj: a qutip observable

    Note:
        if observable is fermionic, we convert to spin representation
        using Jordan-Wigner representation
        all variables in Observable need to be instantiated (i.e cannot be parametric)
    """
    # Import done here for performances
    # pylint: disable=import-outside-toplevel
    from qat.fermion.transforms import transform_to_jw_basis
    import qutip as qt

    # A sanity check that the Observable is already fully evaluated, i.e.
    # no Variable (like 't') or any ArithExpression inside it
    if observable.get_variables():
        raise TypeError("Trying to convert an Observable to Qutip, but it contains "
                        "an ArithExpression.. Observable is:\n" + str(observable))

    observable_ = observable

    # Calculate the Observable's dimension with regard to the posibility of the presence of bosons
    q_systems_in_observable = observable_.nbqbits

    # fermionic observable
    if observable.terms and ("C" in observable.terms[0].op or "c" in observable.terms[0].op):
        observable_ = copy.deepcopy(observable)
        observable_ = transform_to_jw_basis(observable)

    gate_dict = {
        "X": qt.sigmax(),
        "Y": qt.sigmay(),
        "Z": qt.sigmaz(),
        "P": qt.sigmap(),
        "M": qt.sigmam(),
        "I": qt.qeye(q_systems_in_observable),
        "A": lambda n_levels: qt.create(n_levels),
        "a": lambda n_levels: qt.destroy(n_levels),
    }

    subsystems_dims = hilbert_space.subsystems_dims()
    subsytems_ident = [qt.qeye(dim) for dim in subsystems_dims]

    qutip_obs = 0
    for ind, term in enumerate(observable_.terms):

        # Without the line below, the original list will keep being changed ...
        list_ops = subsytems_ident.copy()

        for op, subsys_idx in zip(term.op, term.qbits):

            # If there are bosons, check that each q.system (two-level systems are obviously called qubits)
            # in the observable is applied the approprete operator. 'A' and 'a' are only for bosons and bosons
            # can use no others.
            if op in ("A", "a"):
                if hilbert_space.is_bosonic(subsys_idx):
                    list_ops[subsys_idx] *= gate_dict[op](
                        subsystems_dims[subsys_idx]
                    )
                else:
                    current_line_no = inspect.stack()[0][2]
                    raise exceptions_types.QPUException(code=exceptions_types.ErrorType.INVALID_ARGS,
                                                        modulename=module_name,
                                                        message="Operator %s is bosonic, but was applied on the "
                                                                "two-level quantum system (qubit) number %s. "
                                                                "Please check if the QPU's member "
                                                                "'bosonic_levels' has been specified correctly." \
                                                                % (op, subsys_idx),
                                                        file=__file__,
                                                        line=current_line_no)
            else:
                if not hilbert_space.is_bosonic(subsys_idx):
                    list_ops[subsys_idx] = gate_dict[op]
                else:
                    current_line_no = inspect.stack()[0][2]
                    raise exceptions_types.QPUException(code=exceptions_types.ErrorType.INVALID_ARGS,
                                                        modulename=module_name,
                                                        message="Operator %s is used for spins, but was applied "
                                                                "on the bosonic quantum system %s (i.e. not on a "
                                                                "qubit)." % (op, subsys_idx),
                                                        file=__file__,
                                                        line=current_line_no)

        # Below, term.coeff will be a number, as we check at the top
        # of the function for no Abstract Expressions present
        qutip_obs += term.coeff * qt.tensor(list_ops)

    # Add the constant term
    if abs(observable_.constant_coeff) > 1e-13:
        list_ops = subsytems_ident.copy()
        qutip_obs += observable_.constant_coeff * qt.tensor(list_ops)
    return qutip_obs


def _replace_heaviside(coeff_string: str) -> str:
    """
    A function to replace our heaviside with a numpy function (that Qutip supports)
    """
    res = re.sub(" ", "", coeff_string)
    res = re.sub(
        "heaviside\(([a-z0-9.()\-]+),([0-9.\-]+),([0-9.\-]*)\)",
        r"(np.heaviside(\1-\2, 1)-np.heaviside(\1-\3, 1))",
        res,
    )
    return res


def convert_time_dep_coeff_to_qutip(coeff):
    if not isinstance(coeff, ArithExpression):
        return coeff
    qt_coeff = str(coeff)
    if "heaviside" in qt_coeff:
        qt_coeff = _replace_heaviside(qt_coeff)
    return qt_coeff


def convert_time_dep_obs_to_qutip(obs: Observable, hilbert_space, extra_coeff=1) -> QutipObservable:
    """
    Convert a (potentially) time-dependent Observable to a qutip compatible format.
    """
    if len(obs.get_variables()) == 0:
        return QutipObservable([convert_obs_to_qutip(obs, hilbert_space)])

    time_ind_out = 0
    time_dep_out = []

    for term in obs.terms:
        if isinstance(term.coeff, ArithExpression): # time-dependent
            term_wo_coeff = Observable(obs.nbqbits, pauli_terms=[Term(1, term.op, term.qbits)])
            time_dep_out.append([convert_obs_to_qutip(term_wo_coeff, hilbert_space),
                                 convert_time_dep_coeff_to_qutip(extra_coeff * term.coeff)])
        else: # time-independent
            time_ind_out += convert_obs_to_qutip(Observable(obs.nbqbits, pauli_terms=[term]), hilbert_space)

    # constant coeff
    if isinstance(obs.constant_coeff, ArithExpression):
        cst_coeff = convert_time_dep_coeff_to_qutip(obs.constant_coeff * extra_coeff)
        identity = Observable(obs.nbqbits, pauli_terms=[], constant_coeff=1.)
        time_dep_out.append([convert_obs_to_qutip(identity, hilbert_space), cst_coeff])
    else:
        time_ind_out += obs.constant_coeff

    # mutiply time-indep part by extra coeff
    if isinstance(extra_coeff, ArithExpression):
        time_dep_out.append([time_ind_out, convert_time_dep_coeff_to_qutip(extra_coeff)])
        time_ind_out = 0
    else:
        time_ind_out *= extra_coeff

    # return only what's necessary
    if len(time_dep_out) == 0:
        return QutipObservable([time_ind_out])
    elif time_ind_out == 0:
        return QutipObservable(time_dep_out)
    else:
        time_dep_out.append(time_ind_out)
        return QutipObservable(time_dep_out)


def convert_drive_to_qutip(drive, hilbert_space) -> QutipObservable:
    """
    Convert a drive (from a Schedule) to a qutip-compatible format.

    Note:
        Properly handles drives with time-dependent drive's coefficients or
        Term's coefficients in the drive's Observables.

    Args:
        drive (list(<:class:`~qat.core.variables.ArithExpression`, Observable>)): a drive for which
            the first element of each 2-element list can be an expression of t
        hilbert_space (dict<int, int>, optional) A dictionary with keys representing the levels of
            a given Hilbert space and values for the number of quantum systems in this respective space.
    Returns:
        <input type for QobjEvo>: time dependent hamiltonian in a format ready to be fed to a qutip solver or QobjEvo object.
    """
    out = QutipObservable()
    for coeff, obs in drive:

        if isinstance(coeff, np.ndarray):
            if len(obs.get_variables()) > 0:
                raise ValueError("If one of the drive's coefficient is a numpy array, the corresponding Observable cannot be time-dependent.")

            out.data.append([convert_obs_to_qutip(obs, hilbert_space), coeff])
            continue

        if (len(obs.get_variables()) == 0):
            if isinstance(coeff, ArithExpression):
                out.data.append([convert_obs_to_qutip(obs, hilbert_space),
                            convert_time_dep_coeff_to_qutip(coeff)])
            else:
                out.data.append(convert_obs_to_qutip(coeff * obs, hilbert_space))

        else:
            obs_qutip = convert_time_dep_obs_to_qutip(obs, hilbert_space, coeff)
            out.add(obs_qutip)

    return out

