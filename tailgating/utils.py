"""Utility functions"""
import numpy as np
import itertools
import copy

c = 299792458  # speed of light in m/s

def vec_vals(h, masses):
    """
    Converts a set of eigenvalues of a molecular Hessian into wavenumbers.
    Args
        h: The molecular Hessian
        masses: The masses of each atom in the molecule
    """
    aum = 1.66053906660e-27  # atomic unit of mass in Kg
    har = 4.3597447222071e-18  # convert hartree to joul
    bhr = 5.29177210903e-11  # convert bohr to m
    conv = np.sqrt(har / bhr ** 2 / aum)  # convert frequency from au to SI

    # Converts the Hessians
    h_new = copy.deepcopy(h)

    for i, m1 in enumerate(masses):
        for j, m2 in enumerate(masses):
            entries = itertools.product(range(3 * i, 3 * (i + 1)), range(3 * j, 3 * (j + 1)))
            for x, y in entries:
                h_new[x][y] *= 1 / np.sqrt(m1 * m2)

    # Finds the updated eigenvectors and eigenvalues
    vals, vecs = np.linalg.eig(h_new)

    # Converts into frequencies
    f = np.sqrt(vals) * conv

    # Converts into wavenumbers
    wn = f / (2.0 * np.pi * c) / 100  # wavenumber in 1/cm

    s = sorted(zip(vecs.T, wn), reverse=True, key=lambda x : x[1])

    new_vecs = []
    new_vals = []
    for vec, val in s:
        new_vecs.append(vec)
        new_vals.append(val)

    return np.array(new_vecs).T, new_vals