"""Functionality for creating molecular Hamiltonians and Hamiltonian derivatives"""
import pennylane as qml
import numpy as np
import autohf as hf
import autograd.numpy as anp
import autograd
from autograd.differential_operators import make_jvp
from tqdm.notebook import tqdm


angs_bohr = 1.8897259885789

def generate_basis_set(molecule):
    """
    Generates a basis set corresponding to a molecule
    """
    basis_name = molecule.basis_name
    structure = molecule.symbols
    basis_params = hf.basis_set_params(basis_name, structure)
    hf_b = []

    for b in basis_params:
        t = []
        for func in b:
            L, exp, coeff = func
            t.append(hf.AtomicBasisFunction(L, C=anp.array(coeff), A=anp.array(exp)))
        hf_b.append(t)
    return hf_b


def charge_structure(molecule):
    """
    Computes the charge structure of a molecule
    """
    num_elecs, charges = 0, []
    symbols = molecule.symbols

    for s in symbols:
        c = qml.hf.basis_data.atomic_numbers[s]
        charges.append(c)
        num_elecs += c
    num_elecs -= molecule.charge
    return num_elecs, charges


def hamiltonian(molecule, wires, core, active):
    """
    Computes an electronic Hamiltonian with Hartree-Fock, using the AutoHF library.
    Args
        molecule: chemistry.Molecule object
        wires : (Iterable) Wires on which the Hamiltonian acts
    Returns
        qml.Hamiltonian
    """
    structure = molecule.symbols
    num_elecs, charges = charge_structure(molecule)
    basis = molecule.basis_name
    hf_b = generate_basis_set(molecule)

    def H(R):
        Ri = R.reshape((len(charges), 3))
        geometry = list(zip(structure, (1 / angs_bohr) * Ri))

        arguments = []
        new_b_set = []
        for i, b in enumerate(hf_b):
            arguments.extend([[Ri[i]]] * len(b))
            new_b_set.extend(b)

        integrals = hf.electron_integrals_flat(num_elecs, charges, new_b_set, occupied=core, active=active, cracked=False)(list(Ri), *arguments)

        n = len(active)
        num = (n ** 2) + 1

        core_ad, one_elec, two_elec = integrals[0], integrals[1:num].reshape((n, n)), integrals[num:].reshape((n, n, n, n))
        nuc_energy = core_ad + hf.nuclear_energy(charges)(Ri)
        return hf.build_h_from_integrals(geometry, one_elec, two_elec, nuc_energy, wires, basis=basis, multiplicity=molecule.mult, charge=molecule.charge)
    return H


def d_hamiltonian(molecule, wires, core, active):
    """
    Computes an exact derivative of an electronic Hamiltonian with respect to nuclear coordinates using the
    AutoHF library.

    Args
        molecule: chemistry.Molecule object
        wires : (Iterable) Wires on which the Hamiltonian acts
    Returns
        qml.Hamiltonian
    """
    structure = molecule.symbols
    num_elecs, charges = charge_structure(molecule)
    basis = molecule.basis_name
    hf_b = generate_basis_set(molecule)

    def dH(R, vec):
        re_fn = lambda r : r.reshape((len(charges), 3))
        Ri = re_fn(R)
        geometry = list(zip(structure, (1 / angs_bohr) * Ri))

        def transform(r):
            re = re_fn(r)
            arguments = []
            for i, b in enumerate(hf_b):
                arguments.extend([[re[i]]] * len(b))
            return re, *arguments

        new_b_set = sum(hf_b, [])
        fn = lambda r : hf.electron_integrals_flat(num_elecs, charges, new_b_set, occupied=core, active=active, cracked=False)(*transform(r))
        integrals = make_jvp(fn)(R)(vec)[1]

        n = len(active)
        num = (n ** 2) + 1
        core_ad, one_elec, two_elec = integrals[0], integrals[1:num].reshape((n, n)), integrals[num:].reshape(
            (n, n, n, n))

        nuc_fn = autograd.grad(lambda r : hf.nuclear_energy(charges)(re_fn(r)))(R)
        nuc_energy = core_ad + np.dot(nuc_fn, vec)
        return hf.build_h_from_integrals(geometry, one_elec, two_elec, nuc_energy, wires, basis=basis, multiplicity=molecule.mult, charge=molecule.charge)
    return dH


def dd_hamiltonian(molecule, wires, core, active):
    """
    Computes an exact second derivative of an electronic Hamiltonian with respect to nuclear coordinates, using the
    AutoHF library.

    Args
        molecule: chemistry.Molecule object
        wires : (Iterable) Wires on which the Hamiltonian acts
    Returns
        qml.Hamiltonian
    """
    structure = molecule.symbols
    num_elecs, charges = charge_structure(molecule)
    basis = molecule.basis_name
    hf_b = generate_basis_set(molecule) 

    def ddH(R, vec1, vec2):
        re_fn = lambda r: r.reshape((len(charges), 3))
        Ri = re_fn(R)
        geometry = list(zip(structure, (1 / angs_bohr) * Ri))

        def transform(r):
            re = re_fn(r)
            arguments = []
            for i, b in enumerate(hf_b):
                arguments.extend([[re[i]]] * len(b))
            return re, *arguments

        new_b_set = sum(hf_b, [])
        fn = lambda r: hf.electron_integrals_flat(num_elecs, charges, new_b_set, occupied=core, active=active, cracked=False)(
            *transform(r))
        d_fn = lambda r : make_jvp(fn)(r)(vec1)[1]
        integrals = make_jvp(d_fn)(R)(vec2)[1]

        n = len(active)
        num = (n ** 2) + 1
        core_ad, one_elec, two_elec = integrals[0], integrals[1:num].reshape((n, n)), integrals[num:].reshape(
            (n, n, n, n))

        nuc_fn = autograd.hessian(lambda r: hf.nuclear_energy(charges)(re_fn(r)))(R)
        nuc_energy = core_ad + np.dot(vec1, nuc_fn @ vec2)
        return hf.build_h_from_integrals(geometry, one_elec, two_elec, nuc_energy, wires, basis=basis,
                                         multiplicity=molecule.mult, charge=molecule.charge)
    return ddH


def generate_d_hamiltonian(molecule, wires, bar=True):
    """
    Generates first and second Hamiltonian derivatives
    """
    def H(R):
        dh = d_hamiltonian(molecule, wires)
        ddh = dd_hamiltonian(molecule, wires)

        H1 = []
        bar_range = tqdm(range(len(R))) if bar else range(len(R))
        for j in bar_range:
            vec = np.array([1.0 if j == k else 0.0 for k in range(len(R))])
            H1.append(dh(R, vec))

        H2 = [[0 for l in range(len(R))] for k in range(len(R))]
        for j in range(len(R)):
            bar_range = tqdm(range(len(R))) if bar else range(len(R))
            for k in bar_range:
                if j <= k:
                    vec1 = np.array([1.0 if j == l else 0.0 for l in range(len(R))])
                    vec2 = np.array([1.0 if k == l else 0.0 for l in range(len(R))])
                    val = ddh(R, vec1, vec2)
                    H2[j][k], H2[k][j] = val, val
        return H1, H2
    return H