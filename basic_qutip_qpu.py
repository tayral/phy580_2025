#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file qat/analog/qutip_qpu.py
@authors Grigori Matein <grigori.matein@atos.net>
         Thomas Ayral <thomas.ayral@atos.net>
@internal
@copyright 2021 Bull S.A.S. - All rights reserved.
           This is not Free or Open Source software.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois
@brief Uses the qutip library to support a time dep schedule, Lindblad and stochastic noise, Bosonic Hamiltonians
"""

from typing import List
from time import time as now

import math
import inspect
import logging
import warnings
import json
import copy
import numpy as np

from packaging.version import Version
import qat.comm.exceptions.ttypes as exceptions_types
from qat.comm.datamodel.ttypes import ComplexNumber
from qat.comm.resource.ttypes import ResourceModel, AllocationModel, NodeType
from qat.comm.shared.ttypes import ProcessingType
from qat.core.assertion import assert_qpu
from qat.core import Observable, Term, Result, Sample
from qat.core.json import QLMJsonEncoder
from qat.core.qpu import QPUHandler
from qat.core.variables import ArithExpression, Variable
from qat.core.wrappers.res_data import ResData
from qat.core.wrappers.batch import Batch, Job

from qutip_tools import convert_drive_to_qutip, convert_time_dep_obs_to_qutip, convert_obs_to_qutip
from hilbert_space import HilbertSpace
from result import get_substate_all

module_name = "qat.analog"
_LOGGER = logging.getLogger("qat.analog.qutip_qpu.service")

def check_has_schedule(job):
    if job.schedule is None:
        current_line_no = inspect.stack()[0][2]
        raise exceptions_types.QPUException(code=exceptions_types.ErrorType.INVALID_ARGS,
                                            modulename="qat.analog",
                                            message="For analog computations, please provide "
                                                    "a Job with a Schedule (no circuits allowed).",
                                            file=__file__,
                                            line=current_line_no)

def _use_qutip_4() -> bool:
    """
    This function checks if we use Qutip < 5.0.0.

    If so, "qt.Options" should be used. This class is deprecated
    since Qutip 5.0.0

    Returns:
        bool: we use qutip < 5.0.0
    """
    import qutip as qt  # pylint: disable=import-outside-toplevel
    return Version(qt.__version__).major < 5


# To be tested better and on its own!
def drive_reformatter(drive):
    """
    A function which takes a drive of a Schedule and creates a new one
    in which every Observable has only one Term.
    """

    if isinstance(drive, list):
        n_qbits = drive[0][1].nbqbits
    elif isinstance(drive, Observable):
        n_qbits = drive.nbqbits
    else:
        raise TypeError("Unknown type of the Schedule's drive.")

    new_drive = []
    for drive_coeff, drive_obs in drive:
        for term in drive_obs.terms:
            new_drive.append((drive_coeff * term.coeff,
                              Observable(n_qbits, pauli_terms=[Term(1.0, term.op, term.qbits)])))
        if drive_obs.constant_coeff:
            new_drive.append((drive_coeff * drive_obs.constant_coeff,
                              Observable(n_qbits, pauli_terms=[Term(1.0, "".join(["I" for qbit_i in range(n_qbits)]),
                                                                    [qbit_i for qbit_i in range(n_qbits)])])))
    return new_drive


class QutipQPU(QPUHandler):
    """
    An analog QPU using the QuTiP library. It performs a time-dependent
    evolution of a quantum state under a Hamiltonian specified in a
    :class:`~qat.core.Schedule`.

    The QPU supports executing various types of Hamiltonians like spin, bosonic
    and fermionic schedules and observables (spin and bosonic can be combined).
    It also includes schedules with stochastic parameters and, alternatively, a
    noise description of the environment in terms of Lindblad operators -
    take a look at the :ref:`'Analog' Noise models<jump_op_and_stoch_noise>` section.

    For the noisy evolutions, a `stochastic` method of simulation is provided.
    It would need `n_samples` to be specified, i.e. the number of stochastic
    trajectories.

    When working with bosons, one should specify the number of modes and the number
    of their excitations as tuples in the `bosonic_levels` list. If one further wishes
    to specify the :code:`psi_0` entering the :meth:`~qat.core.Schedule.to_job` of
    :class:`~qat.core.Schedule`, the states of the systems should be ordered from
    belonging to modes with less to modes with more excitations.

    Args:
        n_steps (int, optional): number of simulation steps for `mesolve` and `mcsolve` of
            QuTiP. Default is 100.
        n_procs (int, optional): in `stochastic` mode - the number of processes
            to run trajectories in parallel. Only for jump operators.
        qutip_options (dict, optional): extra options passed to the Qutip solver.
        nsteps (int, optional): alias for n_steps (legacy)
    """
    def __init__(self,
                 n_steps=100,
                 n_procs=None,
                 verbose=False,
                 qutip_options=None,
                 **kwargs):

        self._n_steps = int(kwargs.pop('nsteps', n_steps))
        self._n_procs = kwargs.pop("num_cpus", None) or n_procs
        self._n_procs_rm = None  # allocation from resource manager

        self._hilbert_space = None
        self._bosonic_levels = []


        self.verbose = verbose
        self._qt_options = {"nsteps": self._n_steps}
        if qutip_options is not None:
            self._qt_options.update(qutip_options)

        # hidden arguments to save all states. If one of them is set to True,
        # states are saved at times given by `monitor_times`. For Sample mode
        # only.
        self.states_vs_t = False
        self.require_all_rho = False

        super().__init__(**kwargs)

    @property
    def hilbert_space(self) -> HilbertSpace:
        return copy.deepcopy(self._hilbert_space)

    def _set_psi_0(self, job):
        """
        Factoring out the setting of psi_0 and the checks around it.
        Previously part of sumbit_job().
        """
        # Import done here for performances
        # pylint: disable=import-outside-toplevel
        import qutip as qt

        expected_psi_vector_size = self._hilbert_space.dimension()
        psi_0 = None

        # Make a check that the psi_0 obeys the QPU's member 'hilbert_space' and assign psi_0 accordingly
        if job.psi_0 is not None:
            psi_len = len(job.psi_0)

            # Check if the initial state is given as '0100110874', i.e. bosonic
            if isinstance(job.psi_0, str):
                expected_psi_string_size = self._hilbert_space.n_subsystems
                if psi_len != expected_psi_string_size:
                    current_line_no = inspect.stack()[0][2]
                    raise exceptions_types.QPUException(
                        code=exceptions_types.ErrorType.INVALID_ARGS,
                        modulename=module_name,
                        message=("The job's psi_0 specifies the states of %s of the quantum "
                                 "systems, but the QPU expected %s.") % (psi_len,
                                                                         expected_psi_string_size),
                        file=__file__,
                        line=current_line_no,
                    )

                # Creating a biig vector - from the info about the states of each q. system - basically a tensor product
                list_of_qt_vectors = []
                for idx, dim in enumerate(self._hilbert_space.subsystems_dims()):
                    try:
                        state = int(job.psi_0[idx])
                        if state >= dim:
                            current_line_no = inspect.stack()[0][2]
                            raise exceptions_types.QPUException(code=exceptions_types.ErrorType.INVALID_ARGS,
                                                                modulename=module_name,
                                                                message=("Please specify the state of each quantum system "
                                                                            "as not exceeding the number of its levels. "
                                                                            "Quantum system number %s of job's psi_0 (counting "
                                                                            "from 0) was initially specified to have %s "
                                                                            "levels (counting from 1), yet this is "
                                                                            "<= than the state %s given here (counting "
                                                                            "from 0)." % (idx,
                                                                                        dim,
                                                                                        state)),
                                                                file=__file__,
                                                                line=current_line_no)
                        list_of_qt_vectors.append(qt.basis(dim, state))

                    except ValueError:
                        if job.psi_0[idx] in ('+', '-'):
                            if dim == 2:
                                qt_state = qt.basis(2, 0)
                                qt_state += qt.basis(2, 1) if job.psi_0[idx] == "+" else -qt.basis(2, 1)
                                qt_state /= np.sqrt(2)
                                list_of_qt_vectors.append(qt_state)
                            else:
                                current_line_no = inspect.stack()[0][2]
                                raise exceptions_types.QPUException(code=exceptions_types.ErrorType.INVALID_ARGS,
                                                                    modulename=module_name,
                                                                    message=("Bosonic states cannot be instantiated via "
                                                                                "'+' or '-' as qubits."),
                                                                    file=__file__,
                                                                    line=current_line_no)
                        else:
                            raise exceptions_types.QPUException(
                                message=f"Unsupported initial state {job.psi_0}"
                            )

                psi_0 = qt.tensor(list_of_qt_vectors)

            # Otherwise, check if the initial state is given as a complex vector with the corresponding size
            elif isinstance(job.psi_0, np.ndarray):
                if psi_len != expected_psi_vector_size:
                    current_line_no = inspect.stack()[0][2]
                    raise exceptions_types.QPUException(
                        code=exceptions_types.ErrorType.INVALID_ARGS,
                        modulename=module_name,
                        message=("The job's psi_0 has a dimension %s, but the QPU "
                                 "expected %s.") % (psi_len,
                                                    expected_psi_vector_size),
                        file=__file__,
                        line=current_line_no,
                    )
                psi_0 = qt.Qobj(
                    [[elm.real + 1j * elm.imag] for elm in job.psi_0],
                    dims=[self._hilbert_space.subsystems_dims(), [1] * job.schedule.nbqbits],
                )  # [1] because they are all vectors
            else:
                raise exceptions_types.QPUException(
                    code=exceptions_types.ErrorType.INVALID_ARGS,
                    modulename=module_name,
                    message=f"Invalid psi_0 type: got {type(job.psi_0)}",
                    file=__file__,
                    line=inspect.stack()[0][2]
                )
        else:
            current_line_no = inspect.stack()[0][2]
            raise exceptions_types.QPUException(
                code=exceptions_types.ErrorType.INVALID_ARGS,
                modulename=module_name,
                message="The job's psi_0 attribute should not be None.",
                file=__file__,
                line=current_line_no,
            )
        return psi_0, expected_psi_vector_size

    def _construct_H_res_total(self, schedule, time_list):
        H_res_total = convert_drive_to_qutip(schedule.drive, self._hilbert_space)
        return H_res_total.data


    def submit_job(self, job):
        """
        Args:
            job (Job): a job

        Returns:
            Result object
        """
        # Import done here for performances
        # pylint: disable=import-outside-toplevel
        import qutip as qt

        runtime_start = now()

        # A preliminary check: nbshots =! 0 is not allowed
        if job.nbshots is not None and job.nbshots > 0:
            current_line_no = inspect.stack()[0][2]
            raise exceptions_types.QPUException(code=exceptions_types.ErrorType.INVALID_ARGS,
                                                modulename=module_name,
                                                message="Supporting only nbshots=0, got %s instead" % job.nbshots,
                                                file=__file__,
                                                line=current_line_no)
        # Check that the job has a Schedule (no circuits allowed), otherwise raise an error
        check_has_schedule(job)

        # Check that there is a tmax
        t_final = job.schedule.tmax
        if not t_final or t_final <= 0.0:
            current_line_no = inspect.stack()[0][2]
            raise exceptions_types.QPUException(code=exceptions_types.ErrorType.INVALID_ARGS,
                                                modulename=module_name,
                                                message="Please provide a positive tmax in the Schedule",
                                                file=__file__,
                                                line=current_line_no)
        if self.verbose:
            print("[QutipQPU] Job with tmax = ", t_final)

        # Use t_final to create the time_list for the discrete simulation
        time_list = np.linspace(0.0, t_final, self._n_steps)

        # build hilbert space.
        self._hilbert_space = HilbertSpace()
        self._hilbert_space.infer_from_job(job, self._bosonic_levels)

        # With the now available information for the spaces (and the number of qubits in
        # particular as this is not initially known (i.e. at QPU creation) if bosons are
        # present) set up the jump operators (if given) in QuTiP format.
        self.qutip_c_ops = []

        # Check and set psi_0
        psi_0, expected_psi_vector_size = self._set_psi_0(job)

        # Track if other observables were supplied in the job, so as to get their expectation values as well
        n_other_obs = None
        has_observables = hasattr(job, "observables")
        if has_observables and job.observables != None:
            n_other_obs = len(job.observables)
        else:
            job.observables = None
            n_other_obs = 0

        meta_data = {}

        ############  OBSERVABLE MODE  ############


        if job.type == ProcessingType.OBSERVABLE:

            self._qt_options["store_states"] = False


            # Create the QuTiP Observable(s) to be given to QuTiP's me/mcsolve
            # Qutip doesn't accept or work with e_ops having time-dep. or arith.
            # expressions as coefficients
            if job.observable is not None and not job.observable.get_variables():
                qt_e_obss = [convert_obs_to_qutip(job.observable, self._hilbert_space)]

                # If other observables were given - add them to qt_e_obss
                ## need to have the "observables" option  better documented in schedule.py and job.py
                if job.observables is not None:

                    # Check that the observables are also constant
                    if any([obs.get_variables() for obs in job.observables]):
                        current_line_no = inspect.stack()[0][2]
                        raise exceptions_types.QPUException(
                            code=exceptions_types.ErrorType.INVALID_ARGS,
                            modulename=module_name,
                            message="The observables in 'job.observables' have to be constant, "
                                    "i.e. including no time-dependent or abstract coefficients.",
                            file=__file__,
                            line=current_line_no,
                        )
                    # If obs are constant, add them to the list
                    qt_e_obss += [convert_obs_to_qutip(obs, self._hilbert_space) for obs in job.observables]
            else:
                current_line_no = inspect.stack()[0][2]
                raise exceptions_types.QPUException(
                    code=exceptions_types.ErrorType.INVALID_ARGS,
                    modulename=module_name,
                    message="In OBSERVABLE mode you need to specify an observable. "
                            "The observable has to be constant, i.e. including no "
                            "time-dependent or abstract coefficients.",
                    file=__file__,
                    line=current_line_no,
                )

            # Main procedure for the simulation and extraction of the results
            # Extract the expectation value at the required time(s) for the 0th and other obs (if any)
            if n_other_obs > 0:
                all_expec_vals_OTHER_obs__traj_aver__2Darray = np.zeros((n_other_obs, len(time_list)))


            # Handle stochastic parameters, if any, and construct Hamiltonian
            H_res_total = self._construct_H_res_total(job.schedule, time_list)

            # Execute the respective simulator and get the result
            qt_result = qt.mesolve(H_res_total, psi_0, time_list,
                                   c_ops=self.qutip_c_ops, e_ops=qt_e_obss,
                                   options=qt.Options(**self._qt_options) if _use_qutip_4() else self._qt_options)

            # Extract <O> at the required time(s) for the 0th Observable
            expec_vals_0th_obs__traj_aver__1Darray = qt_result.expect[0][:]

            # Extract the <O> at the required time(s) for the rest of the Observables, if any  
            if n_other_obs > 0:
                for other_obs_i in range(n_other_obs):
                    all_expec_vals_OTHER_obs__traj_aver__2Darray[other_obs_i] = qt_result.expect[1 + other_obs_i][:]

            # Fill the result and return it
            result = Result(
                value=expec_vals_0th_obs__traj_aver__1Darray[-1],  # final expec. value of 0th obs
                # final expec. value of OTHER obs
                values=all_expec_vals_OTHER_obs__traj_aver__2Darray[:, -1] if n_other_obs > 0 else None,
                value_data={
                    str(t): ComplexNumber(re=v, im=0.0)
                    for t, v in zip(time_list, expec_vals_0th_obs__traj_aver__1Darray)
                },
                values_data=[
                    {
                        str(t): ComplexNumber(re=v, im=0.0)
                        for t, v in zip(time_list, expec_vals_ith_obs__traj_aver__1Darray)
                    } for expec_vals_ith_obs__traj_aver__1Darray in all_expec_vals_OTHER_obs__traj_aver__2Darray
                ] if n_other_obs > 0 else None,
                meta_data=meta_data
            )


        ############  SAMPLE MODE  ############


        elif job.type == ProcessingType.SAMPLE:

            self._qt_options["store_states"] = True

            # Warn users if they give Observable(s) in this SAMPLE mode.
            if job.observable is not None:
                warnings.warn("SAMPLE mode simulation was chosen, but the Job has an Observable.")

            # Depending on whether the user wants the states (or rho) at all times
            # or only the last one, we prepare the storage list size and the times
            # to be iterated.
            # If the user explicitely wants the states at all times - they should have
            # set an attribute "states_vs_t" of the QutipQPU object to "True" (if they ask,
            # we'll tell them). Same to get the rho at all times - by both setting a
            # "require_all_rho" to "True", asking us for this info, but also calling
            # the "compute_density_matrix()" function.
            if (hasattr(self, "states_vs_t") and self.states_vs_t) or (
                hasattr(self, "require_all_rho") and self.require_all_rho):
                n_times_to_store_states = len(time_list)
                iterated_times_of_states = range(len(time_list))
            else:
                n_times_to_store_states = 1
                iterated_times_of_states = [-1] # i.e. the last one

            # Prepare an empty array to be filled with the amplitudes of the required state(s).
            # Maybe we need to have a sparse version of this..
            n_amps = expected_psi_vector_size 

            # Create an array to store the (averaged) density matrix and
            # the list of samples - both at all times
            rho_at_chosen_times_traj_aver = np.zeros((n_times_to_store_states, n_amps, n_amps),
                                                     dtype=complex)
            sample_2Dlist_for_last_or_all_times = [] # lists with Samples will be added to this list


            n_samples = self._n_samples if self._sim_method == "stochastic" else 1

            # Firstly, simulate every trajectory and record it
            qt_result_list = []
            for _ in range(n_samples):

                # Handle stochastic parameters, if any, and construct Hamiltonian
                H_res_total = self._construct_H_res_total(job.schedule, time_list)

                # And this one below will sometimes raise a:
                # "DeprecationWarning: an integer is required (got type numpy.float64)."
                # The problem doesn't seem on our side though.
                qt_result_list.append(qt.mesolve(H_res_total, psi_0, time_list,
                                                 c_ops=self.qutip_c_ops, e_ops=[],
                                                 options=qt.Options(**self._qt_options) if _use_qutip_4() else self._qt_options,
                                                 ))

            # Iterate over the chosen times (all or only last one), then go
            # through the states of every trajectory and sum their probs
            for time_i in iterated_times_of_states:
                amplitudes_determ = np.zeros(n_amps, dtype=complex) # only used when not in "stochastic" mode, n_samples = 1
                probs_from_all_traj = np.zeros(n_amps, dtype=float)

                for traj_i in range(n_samples):
                    # Extract the QuTiP state (or the density matrix) for the required traj and time
                    qt_state = qt_result_list[traj_i].states[time_i]
                    np_state = qt_state.full()  # make numpy array

                    # Prepare to extract the comp. basis state - needed to compute rho
                    amps_from_current_traj = np.zeros(n_amps, dtype=complex)

                    # Extract amplitudes and / or probabilities
                    # and add them to the total from the other trajs
                    non_zero_el_rows, non_zero_el_columns = np_state.nonzero()
                    for row, column in zip(non_zero_el_rows, non_zero_el_columns):
                        # if it's a state: - when there are NO jump ops
                        if qt_state.isket:
                            amplitude = np_state[row, column]
                            prob = abs(amplitude)**2
                        # if it's a density matrix - when there are jump ops
                        elif qt_state.isoper:
                            # only the diagonal terms have physical meaning (e.g. |01><01|
                            # and not |10><11|), i.e. they are probability
                            if row != column: continue
                            amplitude = None
                            prob = np_state[row, column].real
                        else:
                            raise TypeError("Unknown type %s" % qt_state.type)

                        probs_from_all_traj[row] += prob
                        amps_from_current_traj[row] = (amplitude if amplitude is not None else 0)


                    amplitudes_determ = amps_from_current_traj

                # Average the probs over the trajectories
                traj_aver_probs = probs_from_all_traj / n_samples

                # Create a new list of Samples (in the 2D list of lists) for this time_i
                sample_2Dlist_for_last_or_all_times.append([])
                # For every state - create a Sample
                for idx in range(n_amps):
                    if abs(amplitudes_determ[idx]) > 0 and self._sim_method != "stochastic":
                        amplitude = amplitudes_determ[idx]
                    else:
                        amplitude = None
                    if traj_aver_probs[idx] > 1e-13:
                        state_i = idx 
                        sample_2Dlist_for_last_or_all_times[time_i].append(Sample(state=state_i,
                                                                                  amplitude=amplitude,
                                                                                  probability=traj_aver_probs[idx]))

            #### NOW CREATE A RESULT

            # No need to serialize and deserialize it afterwards in fact..
            # (as we only return it, we don't keep it in a serializable object like the Result)

            if n_times_to_store_states > 0:
                # If the user required either the states or the rhos at all
                # times - we fill the respective fields
                if self.states_vs_t:
                    meta_data['states_vs_t'] = json.dumps(sample_2Dlist_for_last_or_all_times, cls=QLMJsonEncoder)
                if self.require_all_rho:
                    self.all_rho = rho_at_chosen_times_traj_aver

            spl_list = sample_2Dlist_for_last_or_all_times[-1]
            if job.qubits is not None:
                assert_qpu(self._hilbert_space.n_bosons == 0, "Restricting sampling on some qubits is not supported with bosonic modes.")
                spl_list = get_substate_all(spl_list, job.schedule.nbqbits, job.qubits, job.nbshots != 0)
                self._hilbert_space.take_substate(job.qubits)

            qregs = self._hilbert_space.get_quantum_registers()

            result = Result(raw_data=spl_list, qregs=qregs, meta_data=meta_data)

        else:  # not observable or sample mode
            current_line_no = inspect.stack()[0][2]
            raise exceptions_types.QPUException(code=exceptions_types.ErrorType.INVALID_ARGS,
                                                modulename=module_name,
                                                message="Unsupported Job type %s" % job.type,
                                                file=__file__,
                                                line=current_line_no)

        result.meta_data["simulation_time"] = str(now() - runtime_start)

        return result

