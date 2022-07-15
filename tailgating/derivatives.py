"""Functionality for computing energy derivatives"""
import pennylane as qml
import numpy as np
from tqdm.notebook import tqdm


def compute_parameter_hessian(energy_fn, params, bar=False):
    """Computes the parameter Hessian of a cost function. This is all second derivatives of a quantum circuit with
    respect to circuit parameters.

    Args
        energy_fn (func): A quantum function that returns the expectation value of a Hamiltonian
        params (Iterable): The parameters at which to evaluate the second derivatives of the quantum function
    Kwargs
        bar(bool): Specifies whether to show a progress bar
    Returns
        numpy.array: Parameter Hessian
    """
    num_params = len(params)
    E_theta = qml.grad(energy_fn)
    hess_matrix = []

    prog = tqdm(range(num_params)) if bar else range(num_params)

    for a in prog:
        if bar:
            prog.set_description("Calculating Row: {}".format(a))
        objective_fn = lambda theta: E_theta(theta)[a]
        row = qml.grad(objective_fn)(params).flatten()
        hess_matrix.append(row)

    return np.array(hess_matrix)


def compute_second_derivative(index, H1, H2, parameter_hessian, circuit, dev, optimal_params, diff_method="adjoint"):
    """Computes an entry of the Hessian using provided data.

    Args
        entry ([int, int]): The entry of the molecular Hessian to compute
        energy_data
        theta_data
        H2
        circuit
        dev
        optimal_params
    Returns
        float: Molecular Hessian entry
    """
    num_coordinates = len(H1)

    # Computes each of the derivatives of the Hamiltonian
    prog_1 = tqdm(range(num_coordinates))

    E_dots = []
    theta_stars = []

    for i in prog_1:
        prog_1.set_description("Calculating Energy: {}".format(i))
        @qml.qnode(dev, diff_method=diff_method)
        def circ(params):
            circuit(params)
            return qml.expval(H1[i])
        E_dot_i = qml.grad(circ)(optimal_params)
        E_dots.append(np.array([float(y) for y in E_dot_i]))

    prog_1 = tqdm(range(num_coordinates))

    for i in prog_1:
        prog_1.set_description("Calculating Theta: {}".format(i))
        theta_stars.append(np.linalg.solve(parameter_hessian, -1 * E_dots[i]))

    i, j = index[0], index[1]
    E_dot_i, E_dot_j, theta_star = E_dots[i], E_dots[j], theta_stars[i]

    # Computes the expectation value of the second derivative of the Hamiltonian
    H_dot_dot = H2

    # Compute second derivative of the Hamiltonian and its corresponding expectation value
    @qml.qnode(dev, diff_method=diff_method)
    def circ(params):
        circuit(params)
        return qml.expval(H_dot_dot)
    E_dot_dot = circ(optimal_params)

    return np.dot(E_dot_j, theta_star) + E_dot_dot

def compute_entry_from_data(entry, energy_data, theta_data, H2, circuit, dev, optimal_params, sparse=False, diff_method="adjoint"):
    """Computes an entry of the Hessian using provided data.

    Args
        entry ([int, int]): The entry of the molecular Hessian to compute
        energy_data
        theta_data
        H2
        circuit
        dev
        optimal_params
    Returns
        float: Molecular Hessian entry
    """
    i, j = entry[0], entry[1]
    E_dot_i, E_dot_j, theta_star = energy_data[i], energy_data[j], theta_data[i]

    # Computes the expectation value of the second derivative of the Hamiltonian
    H_dot_dot = H2[i][j]

    # Compute second derivative of the Hamiltonian and its corresponding expectation value
    diff_method = "parameter-shift" if sparse else diff_method

    @qml.qnode(dev, diff_method=diff_method)
    def circ(params):
        circuit(params)
        return qml.expval(H_dot_dot)
    E_dot_dot = circ(optimal_params)

    return np.dot(E_dot_j, theta_star) + E_dot_dot

# Generates the Hessian
def hessian(H1, H2, circuit, dev, optimal_params, parameter_hessian, bar=False, sparse=False, diff_method="adjoint"):
    """Computes the molecular Hessian.

    Args
        H1
        H2
        circuit
        dev
        optimal_params
        parameter_hessian
    Kwargs
        bar
    Returns
        numpy.array: Molecular Hessian
    """
    num_coordinates = len(H1)

    # Computes each of the derivatives of the Hamiltonian
    prog_1 = tqdm(range(num_coordinates))

    E_dots = []
    theta_stars = []
    diff_method = "parameter-shift" if sparse else diff_method

    for i in prog_1:
        prog_1.set_description("Calculating Energy: {}".format(i))
        @qml.qnode(dev, diff_method=diff_method)
        def circ(params):
            circuit(params)
            return qml.expval(H1[i])
        E_dot_i = qml.grad(circ)(optimal_params)
        E_dots.append(np.array([float(y) for y in E_dot_i]))

    prog_1 = tqdm(range(num_coordinates))

    for i in prog_1:
        prog_1.set_description("Calculating Theta: {}".format(i))
        theta_stars.append(np.linalg.solve(parameter_hessian, -1 * E_dots[i]))

    matrix = np.zeros((num_coordinates, num_coordinates))

    for i in range(num_coordinates):
        prog_2 = tqdm(range(num_coordinates))
        for j in prog_2:
            if i <= j:
                prog_2.set_description("Calculating Entry: {}".format([i, j]))
                res = compute_entry_from_data([i, j], E_dots, theta_stars, H2, circuit, dev, optimal_params, diff_method=diff_method)
                matrix[i][j] = res
                matrix[j][i] = res
    return matrix