#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file qat/analog/hilbert_space.py
@authors Corentin Bertrand <corentin.bertrand@eviden.com>
@internal
@copyright 2024 Bull S.A.S. - All rights reserved.
           This is not Free or Open Source software.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois
"""

from operator import itemgetter
from typing import List, Tuple
import numpy as np
from qat.core.assertion import assert_qpu
from qat.core.bits import DefaultRegister
from qat.lang.AQASM.qboso import BosonicRegister

def _job_operators_vs_q_systems_stats(job):
    """
    Return a dict, in which the qubit numbers are separated into three lists:
    one list for qubits, one for fermions and one for bosons.
    """

    operator_types_vs_q_systems_dict = {'qubits': [], 'fermions': [], 'bosons': []}
    qubit_operators = ['X', 'Y', 'Z']
    fermionic_operators = ['c', 'C']
    bosonic_operators = ['a', 'A']
    identity = 'I'
    examined_qubits_list = [False] * job.schedule.nbqbits
    for _, observable in job.schedule.drive:
        for term in observable.terms:

            # We don't allow having bosonic and fermionic operators in the same Term,
            # neither qubit and fermionic ones together, because the jw basis transform
            # method in it's current form won't work
            term_has_qubits = False
            term_has_fermions = False
            term_has_bosons = False

            for qubit_index, operator in zip(term.qbits, term.op):

                # Recognise the type of the quantum system by the operator used on it
                system_type = None
                if operator in qubit_operators:
                    system_type = 'qubits'
                    term_has_qubits = True
                elif operator in fermionic_operators:
                    system_type = 'fermions'
                    term_has_fermions = True
                elif operator in bosonic_operators:
                    system_type = 'bosons'
                    term_has_bosons = True
                elif operator == identity: # we can't decide what the q. system is in this case
                    continue
                else:
                    assert_qpu(False, "The operator %s was not recognised." % (operator))

                # Check if the qubit has already been categorised as being from a different
                # quantum system by looking at the qubits in the rest of the systems
                quantum_systems_types = operator_types_vs_q_systems_dict.keys()
                for quantum_systems_type in quantum_systems_types:
                    if quantum_systems_type == system_type:
                        continue
                    elif qubit_index in operator_types_vs_q_systems_dict[quantum_systems_type]:
                        assert_qpu(False, "Quantum system %s was specified both as a qubit and a boson." % (qubit_index))

                # If it reached until here, then the qubit can be appended to its
                # respective system, if not there already
                if qubit_index not in operator_types_vs_q_systems_dict[system_type]:
                    operator_types_vs_q_systems_dict[system_type].append(qubit_index)
                    examined_qubits_list[qubit_index] = True

            # In general, if a term has both bosonic and fermionic operators -> not allowed (see above)
            if (term_has_fermions and term_has_qubits) or (term_has_fermions and term_has_bosons):
                assert_qpu(False, "Cannot mix fermionic with qubit or bosonic operators in the same term, %s." % term.qbits)

    # Finally, add all the not-operated qubits to the qubits list and sort each list
    unoperated_on_qubits = [qubit_index for qubit_index, qubit_presence in enumerate(examined_qubits_list) if not qubit_presence]
    operator_types_vs_q_systems_dict['qubits'].extend(unoperated_on_qubits)
    for quantum_systems_type in operator_types_vs_q_systems_dict.keys():
        operator_types_vs_q_systems_dict[quantum_systems_type].sort()
    return operator_types_vs_q_systems_dict


class HilbertSpace:

    def __init__(self, n_qubits: int = 0, n_fermions: int = 0, bosons: List[Tuple[int, int]] = None):
        self.n_qubits = int(n_qubits)
        self.n_fermions = int(n_fermions)
        self._bosons = self.clean_bosons(bosons or [])

    def infer_from_job(self, job, bosons: List[Tuple[int, int]]):
        qubit_types_stats = _job_operators_vs_q_systems_stats(job)
        n_bosons_sched = len(qubit_types_stats['bosons'])
        self.n_qubits = len(qubit_types_stats['qubits'])
        self.n_fermions = len(qubit_types_stats['fermions'])
        self._bosons = self.clean_bosons(bosons)

        assert_qpu(self.n_bosons == n_bosons_sched, f"The schedule of the job is acting on {n_bosons_sched} qudit(s), but the total number of bosons specified by the QPU is {self.n_bosons}.")

    @staticmethod
    def clean_bosons(bosons: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        _bosons = [(int(n_modes), int(dim)) for n_modes, dim in bosons]

        # bosons are sorted by increasing dimension
        _bosons.sort(key=itemgetter(1))

        for _, dim in _bosons:
            assert_qpu(dim > 2, "Bosonic modes with 2 levels or less are not supported.")

        return _bosons


    @property
    def bosons(self):
        return self._bosons.copy()

    @property
    def n_bosons(self) -> int:
        return sum(n_modes for n_modes, _ in self._bosons)

    @property
    def n_2_level_systems(self) -> int:
        return self.n_qubits + self.n_fermions

    def dimension_bosons(self) -> int:
        return int(np.prod([dim**n_modes for n_modes, dim in self._bosons]))

    def dimension(self) -> int:
        return 2**(self.n_qubits + self.n_fermions) * self.dimension_bosons()

    @property
    def n_subsystems(self) -> int:
        return self.n_bosons + self.n_qubits + self.n_fermions

    def subsystems_dims(self) -> List[int]:
        out = [2] * self.n_2_level_systems
        for n_modes, dim in self._bosons:
            out += [dim] * n_modes
        return out

    def is_bosonic(self, subsystem_index) -> bool:
        assert subsystem_index < self.n_subsystems
        return subsystem_index >= self.n_2_level_systems

    def take_substate(self, substate_qubits: List[int]):
        assert_qpu(self.n_bosons == 0, "Restricting sampling on some qubits is not supported with bosonic modes.")
        # a more complex logic could be implemented here
        assert_qpu(all(0 <= qubit < self.n_2_level_systems for qubit in substate_qubits), "Measured qubits be within the qubit range.")
        self.n_qubits = len(substate_qubits)
        self.n_fermions = 0

    def get_quantum_registers(self):
        """
        Two-level quantum systems will need a DefaultRegister and bosonic q. systems will need a BosonicRegister.
        This method prepares the respective registers of each system - they are needed for simulations in
        SAMPLE mode and will enter the Result.
        """
        qregs = []
        if self.n_2_level_systems > 0:
            qregs.append(DefaultRegister(offset=0, length=self.n_2_level_systems))

        for n_modes, dim in self._bosons:
            qregs.append(BosonicRegister(length=n_modes, levels=dim))

        return qregs

    def __str__(self):
        return f"Hilbert Space: {self.n_qubits} qubits, {self.n_fermions} fermions, {self._bosons} bosons."
