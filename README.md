![mpi4torch Logo](./doc/_static/img/mpi4torch-logo-extrawhitespace.png)

--------------------------------------------------------------------------------

mpi4torch is an automatic-differentiable wrapper of MPI functions for the pytorch tensor library.

MPI stands for Message Passing Interface and is the de facto standard communication interface on
high-performance computing resources. To facilitate the usage of pytorch on these resources an MPI wrapper
that is transparent to pytorch's automatic differentiation (AD) engine is much in need. This library tries
to bridge this gap.

# Installation

mpi4torch is also hosted on PyPI. However, due to the ABI-incompatibility of the different MPI implementations it
is not provided as a binary wheel and needs to be built locally. Hence, you should have an appropriate C++ compiler
installed, as well as the  **development files of your MPI library** be present. The latter are usually provided
through the *module system* of your local cluster, and you should consult the manuals of your cluster for this,
or through the package manager of your Linux distribution.

Once the dependencies have been satisfied the installation can be triggered by the usual
```
    pip install mpi4torch
```

# Usage

It is **highly advised** to first read [the basic usage chapter of the documentation](https://mpi4torch.readthedocs.io/en/latest/basic_usage.html)
before jumping into action, since there are some implications of the pytorch AD design on the usage of mpi4torch.
In other words, there are some footguns lurking!

You have been warned, but if you insist on an easy usage example, consider the following code snippet,
which is an excerpt from [examples/simple_linear_regression.py](examples/simple_linear_regression.py)

```python
   comm = mpi4torch.COMM_WORLD

   def lossfunction(params):
       # average initial params to bring all ranks on the same page
       params = comm.Allreduce(params, mpi4torch.MPI_SUM) / comm.size

       # compute local loss
       localloss = torch.sum(torch.square(youtput - some_parametrized_function(xinput, params)))

       # sum up the loss among all ranks
       return comm.Allreduce(localloss, mpi4torch.MPI_SUM)
```

Here we have parallelized a loss function simply by adding two calls to `Allreduce`. For a more thorough
discussion of the example see [here](https://mpi4torch.readthedocs.io/en/latest/examples.html#simple-data-parallel-example).

# Tests

Running tests is as easy as
```
    mpirun -np 2 nose2
```

# Project Status

[![Tests](https://github.com/helmholtz-analytics/mpi4torch/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/helmholtz-analytics/mpi4torch/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/mpi4torch/badge/?version=latest)](https://mpi4torch.readthedocs.io/en/latest/?badge=latest)
