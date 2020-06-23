*******************
Examples
*******************

Simple data parallel example
============================

Let us assume we are in typical supervised learning situation. We have plenty of data (``xinput``, ``youtput``),
and we search for unknown parameters minimizing some norm or general functional, simply referred to as the
loss function. Furthermore, we assume that the
loss function is just a summation of losses per data point. E.g. consider the following squared error:

.. code-block:: python

   def lossfunction(params):
       # compute local loss
       localloss = torch.sum(torch.square(youtput - some_parametrized_function(xinput, params)))
       return localloss

This function is usually fed into an gradient-based optimizer to find the optimal parameters.
We want to argue in the following that parallelizing this code in a data-parallel way is often as easy as
adding two calls to  :py:meth:`mpi4torch.MPI_Communicator.Allreduce`:

.. code-block:: python
   :emphasize-lines: 3,9

   def lossfunction(params):
       # average initial params to bring all ranks on the same page
       params = comm.Allreduce(params, mpi4torch.MPI_SUM) / comm.size

       # compute local loss
       localloss = torch.sum(torch.square(youtput - some_parametrized_function(xinput, params)))

       # sum up the loss among all ranks
       return comm.Allreduce(localloss, mpi4torch.MPI_SUM)

:py:meth:`mpi4torch.MPI_Communicator.Allreduce` is used once to compute the average of the incoming parameters
and once to collect the total loss.

Embedded in a whole program this may look like (the code is also available in the git repository
in the examples folder):

.. literalinclude:: ../examples/simple_linear_regression.py
   :linenos:

Note that although the averaging in line 29 might seem superfluous at first --- since all ranks start
off with the same initial set of parameters --- having the adjoint of :py:meth:`mpi4torch.MPI_Communicator.Allreduce`
in the backward pass
is essential for all instances of the LBFGS optimizer to perform the same update on all ranks.

For the second call to :py:meth:`mpi4torch.MPI_Communicator.Allreduce` in line 35 it is actually the other way
around: Here the forward pass is crucial, but the backward pass merely adds up the ones coming from the
different ranks, which (surprise) results in a vector of length 1 that just contains the communicator size.

It is easy to see that the forward pass is indpendent of the number of ranks used to compute the result. That
the parallelized backward pass also gives the same result may at first seem a bit surprising,
as we already saw that the gradient with respect to ``localloss`` will just store the size
of the MPI communicator. However,
the corresponding backward code of the averaging in line 29 divides again through ``comm.size``, such that
in total all gradients from all ranks are simply added up. The final
gradient as stored in ``params.grad`` is thus also independent of the number of processes.

Starting off with the same
parameters on all ranks, it is thus ensured that all local LBFGS instances see the same parameters, the same losses
and the same gradients, and thus perform the identical operations and give the same result.



