#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file qat/analog/result.py
@authors Corentin Bertrand <corentin.bertrand@eviden.com>
@internal
@copyright 2024 Bull S.A.S. - All rights reserved.
           This is not Free or Open Source software.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois
"""
from math import sqrt
from typing import List

from qat.core import Result, Sample
from qat.core.simutil import get_substate


def get_substate_all(sample_list: List[Sample], nbqubits: int, substate_qubits: List[int], counts_only: bool) -> List[Sample]:
    """
    Take the substate of all samples form a list. If `counts_only` is false,
    the list is expected to have probabilities (or amplitudes) which are summed
    correctly after the substate is taken.

    Args:
        sample_list (list<Sample>): list of samples with or without probabilities
        nbqubits (int): number of qubits in original samples
        substate_qubits (list<int>): list of qubits over which the samples must be taken
        counts_only (bool): true for when there is no probability in the samples.

    Return:
        (list<Sample>): a new list of samples
    """
    if counts_only:
        return [Sample(state=get_substate(spl.state, nbqubits, substate_qubits)) for spl in sample_list]

    out: List[Sample] = []
    for spl in sample_list:
        subst = get_substate(spl.state.int, nbqubits, substate_qubits)
        new_spl = Sample(state=subst,
                         probability=spl.probability,
                         amplitude=spl.amplitude,
                         err=spl.err)
        _add_sample_to_list_with_proba(out, new_spl)

    return out


def _add_sample_to_list_with_proba(sample_list: List[Sample], sample: Sample):
    """
    Add a sample to a list of sample, summing probabilities correctly.

    The samples are assumed to have a probability or amplitude.
    `sample_list` is modified inplace.
    """
    if sample.amplitude is None and sample.probability is None:
        raise ValueError("`sample` should contain a probability or amplitude.")

    idx = 0
    for spl in sample_list:
        if spl.state == sample.state:
            break
        idx += 1

    if idx < len(sample_list):  # already exists in the list
        spl = sample_list[idx]

        if spl.amplitude is None and spl.probability is None:
            raise ValueError("The list of samples provided contains samples with neither a probability or amplitude.")

        # we need to add up the two probabilities
        prob = spl.probability or (spl.amplitude.real**2 + spl.amplitude.imag**2)
        if sample.probability is not None:
            prob += sample.probability
        else:
            prob += sample.amplitude.real**2 + sample.amplitude.imag**2

        error = sample.err
        if sample.err is not None and spl.err is not None:
            error = sqrt(sample.err**2 + spl.err**2)

        sample_list[idx] = Sample(state=spl.state, probability=prob, err=error)

    else:  # this is a new state
        sample_list.append(sample)
