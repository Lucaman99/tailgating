# tailgating

Numerical demonstration of the tailgating procedure

## Overview

A basic example implemented in PennyLane highlighting the tailgating procedure. I also make use the `autohf` library ([Github repo](https://github.com/Lucaman99/autohf)): the prototype version of the `pennylane.hf` module, as well as the `bigvqe` library ([Github repo](https://github.com/Lucaman99/bigvqe)): a package for faster computation of sparse fermionic Hamiltonians.

**Important:** The examples highlighted in this repository make use of the PennyLane library, with the following method added to the `GradientDescentOptimizer` class:

```python
def step_and_cost_and_grad(self, objective_fn, *args, grad_fn=None, **kwargs):
    g, forward = self.compute_grad(objective_fn, args, kwargs, grad_fn=grad_fn)
    new_args = self.apply_grad(g, args)

    if forward is None:
        forward = objective_fn(*args, **kwargs)

    # unwrap from list if one argument, cleaner return
    if len(new_args) == 1:
        return new_args[0], forward, g
    return new_args, forward, g
```

This simply allows us to return the value of the gradient, the value of the cost function, and the updated parameters simultaneously when performing gradient descent.

## Installation

To install this package, run:

```
python3 setup.py build_ext --inplace install --user
```

## Cite

Please cite our paper!
