import torch
from ._mpi import *
from typing import List

__all__ = [
    "MPI_MAX",
    "MPI_MIN",
    "MPI_SUM",
    "MPI_PROD",
    "MPI_LAND",
    "MPI_BAND",
    "MPI_LOR",
    "MPI_BOR",
    "MPI_LXOR",
    "MPI_BXOR",
    "MPI_MINLOC",
    "MPI_MAXLOC",
    "WaitHandle",
    "JoinDummies",
    "JoinDummiesHandle",
    "MPI_Communicator",
    "COMM_WORLD",
    "comm_from_mpi4py",
    "deactivate_cuda_aware_mpi_support"
]

@torch.jit.script
class WaitHandle:
    """Class representing a wait handle, as they are returned from one of the non-blocking MPI calls."""

    def __init__(self, raw_handle: List[torch.Tensor]):
        self._handle = raw_handle

    @property
    def dummy(self):
        """A dummy variable that allows for the usage of the ``WaitHandle`` as one of the
        second arguments of :py:func:`mpi4torch.JoinDummies` and :py:func:`mpi4torch.JoinDummiesHandle`.
        """

        return self._handle[0]

@torch.jit.script
def JoinDummies(loopthrough: torch.Tensor, dummies:List[torch.Tensor]) -> torch.Tensor:
    """This function joins multiple dummy dependencies with the DAG.

    From the perspective of the forward pass, this function is mostly a no-op, since it simply
    loops through its first argument, and discards the ``dummies`` argument.

    However, for the backward pass, the AD engine still considers the ``dummies`` as actual
    dependencies. The main use of this function is thus to manually encode dependencies
    that the AD engine does not see on its own. See also the introductory text in
    the :ref:`section_implications_mpi4torch` section on how to use this function.

    Parameters
    ----------
    loopthrough:
        Variable to pass through.
    dummies:
        List of tensors that are added as dummy dependencies to the DAG.

    Returns
    -------
    :py:class:`torch.tensor`:
        Tensor that is a shallow copy of ``loopthrough``, but whose ``grad_fn``
        is ``JoinDummiesBackward``.
    """
    return torch.ops.mpi4torch.JoinDummies(loopthrough, dummies)

@torch.jit.script
def JoinDummiesHandle(handle: WaitHandle, dummies:List[torch.Tensor]) -> WaitHandle:
    """This function has the same purpose as :py:func:`JoinDummies`, but accepts :py:class:`mpi4torch.WaitHandle`
    as its first argument.

    Parameters
    ----------
    handle:
        :py:class:`mpi4torch.WaitHandle` to pass through.
    dummies:
        List of tensors that are added as dummy dependencies to the DAG.

    Returns
    -------
    :py:class:`mpi4torch.WaitHandle`:
        A wait handle with the additional dummy dependenices added.
    """
    raw_handle = handle._handle
    return WaitHandle([ torch.ops.mpi4torch.JoinDummies(raw_handle[0], dummies), raw_handle[1], raw_handle[2] ])

@torch.jit.script
class MPI_Communicator:
    """MPI communicator wrapper class

    The only supported ways to construct an ``MPI_Communicator`` are currently either through :py:const:`mpi4torch.COMM_WORLD` or
    :py:func:`mpi4torch.comm_from_mpi4py`.

    Note
    ----
    All methods with an underscore suffix are in-place operations.
    """

    def __init__(self, comm: torch.classes.mpi4torch.MPI_Comm_Wrapper):
        self._comm = comm

    @property
    def rank(self) -> int:
        """The rank or identification number of the local process with respect to this communicator.

        The processes participating in a communicator are consecutively given ranks
        in the interval [0, :py:attr:`mpi4torch.MPI_Communicator.size` - 1].
        """
        return self._comm.GetRank()

    @property
    def size(self) -> int:
        """The size of the MPI communicator, i.e. the number of processes involved."""
        return self._comm.GetSize()

    # This is currently not supported by torch.jit.script:
    #def __getattr__(self, attrName):
    #    if attrName in self.__dict__["_comm"]._method_names():
    #        return self.__dict__["_comm"].__getattr__(attrName)
    #    return self.__dict__[attrName]
    # So we need to write out every function by hand

    def Allreduce(self, tensor: torch.Tensor, op: int) -> torch.Tensor:
        """Combines values from all processes and distributes the result back to all processes.

        The combination operation is performed element-wise on the tensor.

        This is the wrapper function of `MPI_Allreduce <https://www.open-mpi.org/doc/v4.1/man3/MPI_Allreduce.3.php>`_.

        Parameters
        ----------
        tensor:
            :py:class:`torch.Tensor` that shall be combined. It needs to have the same shape on all processes.
        op:
            Operation to combine the results. Only supported operations are :py:const:`mpi4torch.MPI_MAX`,
            :py:const:`mpi4torch.MPI_MIN`, :py:const:`mpi4torch.MPI_SUM`, :py:const:`mpi4torch.MPI_PROD`, :py:const:`mpi4torch.MPI_LAND`,
            :py:const:`mpi4torch.MPI_BAND`, :py:const:`mpi4torch.MPI_LOR`, :py:const:`mpi4torch.MPI_BOR`, :py:const:`mpi4torch.MPI_LXOR`,
            :py:const:`mpi4torch.MPI_BXOR`, :py:const:`mpi4torch.MPI_MINLOC`,
            :py:const:`mpi4torch.MPI_MAXLOC`

        Returns
        -------
        :py:class:`torch.Tensor`:
            Combined tensor of the same shape as the input `tensor`.

        Note
        ----
        Only :py:const:`mpi4torch.MPI_SUM` is supported in the backwards pass at the moment.
        """
        return self._comm.Allreduce(tensor, op)

    def Bcast_(self, tensor: torch.Tensor, root: int) -> torch.Tensor:
        """Broadcasts a tensor from the `root` process to all other processes.

        This is an in-place operation.

        This is the wrapper function of `MPI_Bcast <https://www.open-mpi.org/doc/v4.1/man3/MPI_Bcast.3.php>`_.

        Parameters
        ----------
        tensor:
            :py:class:`torch.Tensor` that shall be broadcasted. The tensor needs to have the same shape on all processes,
            since it is an in-place operation.
        root:
            The root process, whose tensor shall be broadcasted to the others.

        Returns
        -------
        :py:class:`torch.Tensor`:
            For `rank == root` this is the same as the input `tensor`. For all other processes this is the input `tensor` filled with the content
            from the `root` process.
        """
        return self._comm.Bcast_(tensor, root)

    def Reduce_(self, tensor: torch.Tensor, op: int, root: int) -> torch.Tensor:
        """Reduces multiple tensors of the same shape, scattered over all processes, to a single tensor of the same shape stored on the `root` process.

        The combination operation is performed element-wise on the tensor.

        This is an in-place operation.

        This is the wrapper function of `MPI_Reduce <https://www.open-mpi.org/doc/v4.1/man3/MPI_Reduce.3.php>`_.

        Parameters
        ----------
        tensor:
            :py:class:`torch.Tensor` that shall be reduced. The tensor needs to have the same shape on all processes,
            since it is an element-wise operation.
        op:
            Operation to combine the results. Only supported operations are :py:const:`mpi4torch.MPI_MAX`,
            :py:const:`mpi4torch.MPI_MIN`, :py:const:`mpi4torch.MPI_SUM`, :py:const:`mpi4torch.MPI_PROD`, :py:const:`mpi4torch.MPI_LAND`,
            :py:const:`mpi4torch.MPI_BAND`, :py:const:`mpi4torch.MPI_LOR`, :py:const:`mpi4torch.MPI_BOR`, :py:const:`mpi4torch.MPI_LXOR`,
            :py:const:`mpi4torch.MPI_BXOR`, :py:const:`mpi4torch.MPI_MINLOC`,
            :py:const:`mpi4torch.MPI_MAXLOC`
        root:
            The root process, where the resulting tensor shall be gathered.

        Returns
        -------
        :py:class:`torch.Tensor`:
            For `rank == root` the result stores the reduced tensor. For all other processes the content of the resulting tensor is undefined,
            with the exception that the result shall still suffice as input for the second argument of :py:func:`mpi4torch.JoinDummies`.

        Note
        ----
        Only :py:const:`mpi4torch.MPI_SUM` is supported in the backwards pass at the moment.
        """
        return self._comm.Reduce_(tensor, op, root)

    def Gather(self, tensor: torch.Tensor, gatheraxis: int, root: int) -> torch.Tensor:
        return self._comm.Gather(tensor, gatheraxis, root)

    def Allgather(self, tensor: torch.Tensor, gatheraxis: int) -> torch.Tensor:
        return self._comm.Allgather(tensor, gatheraxis)

    def Scatter(self, tensor: torch.Tensor, scatteraxis: int, numelem: int, root: int) -> torch.Tensor:
        return self._comm.Scatter(tensor, scatteraxis, numelem, root)

    def Alltoall(self, tensor: torch.Tensor, gatheraxis: int, scatteraxis: int,
                 numelem: int) -> torch.Tensor:
        return self._comm.Alltoall(tensor, gatheraxis, scatteraxis, numelem)

    def Isend(self, tensor: torch.Tensor, dest: int, tag: int) -> WaitHandle:
        return WaitHandle(self._comm.Isend(tensor, dest, tag))

    def Irecv(self, tensor: torch.Tensor, source: int, tag: int) -> WaitHandle:
        return WaitHandle(self._comm.Irecv(tensor, source, tag))

    def Wait(self, waithandle: WaitHandle) -> torch.Tensor:
        return self._comm.Wait(waithandle._handle)

    def Send(self, tensor: torch.Tensor, dest: int, tag: int) -> torch.Tensor:
        handle = self._comm.Isend(tensor, dest, tag)
        return self._comm.Wait(handle)

    def Recv(self, tensor: torch.Tensor, source: int, tag: int) -> torch.Tensor:
        handle = self._comm.Irecv(tensor, source, tag)
        return self._comm.Wait(handle)

COMM_WORLD = MPI_Communicator(torch.ops.mpi4torch.COMM_WORLD())
"""
World communicator ``MPI_COMM_WORLD``.
"""

try:
    from  mpi4py import MPI as __mpi4py_MPI

    def comm_from_mpi4py(comm: __mpi4py_MPI.Comm) -> MPI_Communicator:
        """Converts an ``mpi4py`` communicator to an :py:class:`mpi4torch.MPI_Communicator`.
        """

        fortran_handle = comm.py2f();
        return MPI_Communicator(torch.ops.mpi4torch.comm_from_fortran(fortran_handle))
except ModuleNotFoundError:
    def comm_from_mpi4py(comm) -> MPI_Communicator:
        """Converts an ``mpi4py`` communicator to an :py:class:`mpi4torch.MPI_Communicator`.
        """

        raise RuntimeError("mpi4py is not available!")
