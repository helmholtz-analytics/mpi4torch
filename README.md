mpi4torch is an automatic-differentiable wrapper of MPI functions for the pytorch tensor library.

MPI stands for Message Passing Interface and is the de facto standard communication interface on
high-performance computing resources. To facilitate the usage of pytorch on these resources an MPI wrapper
that is transparent to pytorch's automatic differentiation (AD) engine is much in need. This library tries
to bridge this gap.

Note that this library is quite lowlevel and tries to stick as closely as possible to the MPI function
names. If you need more convenience, please consult the [HeAT library](https://github.com/helmholtz-analytics/heat),
which (hopefully soon) will internally use this library.

**IMPORTANT** Before directly jumping right into action and starting to port your possibly existing MPI
calls to this library, read the section below on the assumptions this library makes and the consequences
it has for AD.

**WARNING** The software is still in an early development phase. Especially the API
might still be subject to change.

[Documentation is available here](https://stuff.knechtges.com/torchmpi/html)

# Installation

Make sure you have the development tools for MPI installed, either through some module system
if you are working on a bigger cluster, or through the package repositories of your linux
distribution.

Then installing from a local checkout is as easy as
```
    pip install -r requirements.txt
    pip install .
```

# Assumptions and design principles

Pytorch's AD engine uses the concept of a directed acyclic graph (DAG) to provide the AD functionality. This graph
is built incrementally during the forward exectution step, and then executed backward during the backward step.
Essential for this to work is the assumption that all nodes in the DAG are more or less pure functions, whose
output soley depends on their input and not on some global state. However, this is already the first problem
for implementing AD-capable MPI function calls: E.g. just sending or receiving data are functions that from a
DAG perspective have no output or input, i.e. these nodes behave very much like constants and would not be
tracked by the AD engine. The problem at hand is that the AD engine does not see the hidden dependencies that arise
due to the communication with other processes. As such, **the user of this library has to encode these dependencies
manually**. In other words, your code will not work magically with AD by just porting it to this library.

To encode these dependencies, the library provides different measures. One of them is that every function, even
functions like `Send`, have input and output tensors. Another very important tool in this library is the
`mpi4torch.JoinDummies` function:
```
def JoinDummies(tensor: torch.Tensor, args:List[torch.Tensor]) -> torch.Tensor:
```
This function in principle only loops through the first argument. Therefore, during forward execution this
operation is essentially a no-op, since it just returns the first argument. However, since the AD engine does
not know this fact, it believes that the result of the `JoinDummies` function also depends on the second
argument, a list of tensors. This way the user can introduce virtual dependencies. These virtual dependencies
are especially useful to provide synchronization points and to avoid deadlocks. To understand this, **the user
has to acknowledge that the AD engine does not necessarily respect the order in which the commands have been
written to the file**, i.e., commands which are executed after each other in the forward step are not necessarily
executed in reverse order in the backward step unless their is an explicit dependence of the second command on
the result of the first. E.g. consider the following pseudo code, where two processes exchange data through
the common Isend-Recv-Wait idiom
```python
    handle = comm.Isend(sendbuffer,0,0)
    comm.Recv(recvbuffer,0,0)
    comm.Wait(handle)
    res = some_other_function(recvbuffer)
```
In the forward step, everything would work as expected, the code would be executed sequentially as it is was written
to the file. However, in the backward step the information is missing that the `Recv` call has to happen after
the `Isend` call and before the `Wait` call. `JoinDummies` helps here to encode this implicit dependency
```python
    handle = comm.Isend(sendbuffer,0,0)
    res1 = comm.Recv(mpi4torch.JoinDummies(recvbuffer,[handle.dummy),0,0)
    res2 = comm.Wait(mpi4torch.JoinDummiesHandle(handle,[res1]))
    res = some_other_function(mpi4torch.JoinDummies(res1,[res2]))
```
There are now two specialities when dealing with handles:

- If the first argument of a `JoinDummies` function is a handle use `JoinDummiesHandle` instead.
- If you want to pass a handle to the second argument of `JoinDummies` pass `handle.dummy` instead.

The rationale for these rules is that for non-blocking communication in backward mode the receiving
or sending buffer, which is stored in `handle`,
needs to stay in scope. Hence, any bifurcating usage of the second handle entry has to be avoided.

To sum up:
- If you write a pure function that shall be automatic differentiable and uses MPI internally, make it a pure
  function and ensure that all MPI calls in this function are through one way or another connected to both,
  the input and output of the function. This way the AD engine can propagate the gradients through your function
  without missing any communication.
- Use `JoinDummies` and `JoinDummiesHandle` to make the AD engine aware of the hidden dependencies.

# Usage

An easy and low-hanging fruit usage is to make your code data-parallel. E.g. consider the following code snippet,
which is an excerpt from the example in [examples/simple_linear_regression.py](examples/simple_linear_regression.py)

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
discussion of the example consult the [documentation](https://stuff.knechtges.com/torchmpi/html/examples.html).

# Tests

Running tests
```
    mpirun -np 2 nose2
```
