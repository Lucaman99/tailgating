"""Basic functionality"""
import numpy as np
import pennylane as qml
from tqdm.notebook import tqdm


def vqe(circuit, H, dev, optimizer, steps, params, sparse=False, bar=True, diff_method="adjoint"):
    """
    Performs the VQE (Variational Quantum Eigensolver) process for a given circuit and Hamiltonian.
    Optimizes a function of the form C(theta) = < psi(theta) | H | psi(theta) >.

    Args
        circuit (function): A quantum function, implementing a series of parametrized gates
        H (qml.Hamiltonian, qml.SparseHamiltonian): A Hamiltonian to be optimized
        dev (qml.Device): The device on which to perform VQE
        optimizer (qml.GradientDescentOptimizer): The optimizer used during VQE
        steps (int): The number of steps taken by the VQE procedure
        params (Iterable): Initial parameters for VQE optimization
    Kwargs
        sparse (bool): Indicated whether to simulate using sparse methods
        bar (bool): Indicates whether to display a progress bar during optimization
        diff_method (str): The differentiation method to use for VQE (Note: Only works for non-sparse VQE)
    Returns
        (Optimized energy, optimized parameters): (float, Iterable)
    """
    diff_method = "parameter-shift" if sparse else diff_method

    @qml.qnode(dev, diff_method=diff_method)
    def cost_fn(params):
        circuit(params)
        return qml.expval(H)

    nums = tqdm(range(steps)) if bar else range(steps)

    for s in nums:
        params, energy, grad = optimizer.step_and_cost_and_grad(cost_fn, params)
        if np.allclose(grad, 0.0):
            break
        if bar:
            nums.set_description("Energy = {}".format(energy))

    return energy, params


def adapt_vqe(H, dev, operator_pool, hf_state, optimizer, max_steps, vqe_steps, bar=False):
    """Performs the original ADAPT-VQE procedure using the sparse VQE method.
    See [arXiv:1812.11173v2] for more details.

    Args
        H (qml.Hamiltonian): A Hamiltonian used to perform VQE
        dev (qml.Device): A device on which to perform the simulations
        operator_pool (Iterable[function]): A collection of parametrized quantum gates which will make up the operator pool
        Each element is of type (float or array) -> (qml.Operation)
        hf_state (array): The Hartree-Fock state
        optimizer (qml.GradientDescentOptimizer): The optimizer used for VQE
        steps (float): The number of times the adaptive loop should be executed
        vqe_steps (float): The number of steps that VQE should take, for each adaptive loop
    Kwargs
        bar (bool): Specifies whether to show a progress bar
    Returns
        (Iterable[function]): The sequence of quantum operations yielded from ADAPT-VQE
        (Iterable[float]): The optimized parameters of the circuit consisting of the outputted quantum operations
    """
    optimal_params = []
    seq = []
    termination = False
    counter = 0

    while not termination or counter < max_steps:
        grads = []
        for op in operator_pool:

            # Constructs the new circuit
            @qml.qnode(dev, diff_method='parameter-shift')
            def cost_fn(param):
                qml.BasisState(hf_state, wires=dev.wires)
                for operation, p in zip(seq, optimal_params):
                    operation(p)
                op(param)
                return qml.expval(H)

            # Computes the gradient of the circuit
            grad_fn = qml.grad(cost_fn)(0.0)
            grads.append(grad_fn)

        abs_ops = [abs(x) for x in grads]
        if np.allclose(abs_ops, 0.0):
            termination = True
            break
        chosen_op = operator_pool[abs_ops.index(max(abs_ops))]

        def vqe_circuit(params):
            qml.BasisState(hf_state, wires=dev.wires)
            for operation, p in zip(seq, params[:len(params) - 1]):
                operation(p)
            chosen_op(params[len(params) - 1])

        energy, optimal_params = vqe(vqe_circuit, H, dev, optimizer, vqe_steps, optimal_params + [0.0], sparse=True, bar=bar)
        seq.append(chosen_op)
        counter += 1
    return seq, optimal_params


def gate_pool(active_electrons, active_orbitals):
    """
    Generates a gate pool and single and double excitations
    """
    singles, doubles = qml.qchem.excitations(electrons=active_electrons, orbitals=2 * active_orbitals)
    pool = []

    for s in singles:
        pool.append(lambda p, w=s: qml.SingleExcitation(p, wires=w))
    for d in doubles:
        pool.append(lambda p, w=d: qml.DoubleExcitation(p, wires=w))
    return pool