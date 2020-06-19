import torch
from ._mpi import *
from mpi4py import MPI as __mpi4py_MPI
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

    def __init__(self, raw_handle: List[torch.Tensor]):
        self._handle = raw_handle

    @property
    def dummy(self):
        return self._handle[0]

@torch.jit.script
def JoinDummies(tensor: torch.Tensor, args:List[torch.Tensor]) -> torch.Tensor:
    return torch.ops.torchmpi.JoinDummies(tensor, args)

@torch.jit.script
def JoinDummiesHandle(handle: WaitHandle, dummies:List[torch.Tensor]) -> WaitHandle:
    raw_handle = handle._handle
    return WaitHandle([ torch.ops.torchmpi.JoinDummies(raw_handle[0], dummies), raw_handle[1], raw_handle[2] ])

@torch.jit.script
class MPI_Communicator:
    def __init__(self, comm: torch.classes.torchmpi.MPI_Comm_Wrapper):
        self._comm = comm

    @property
    def rank(self) -> int:
        return self._comm.GetRank()

    @property
    def size(self) -> int:
        return self._comm.GetSize()

    # This is currently not supported by torch.jit.script:
    #def __getattr__(self, attrName):
    #    if attrName in self.__dict__["_comm"]._method_names():
    #        return self.__dict__["_comm"].__getattr__(attrName)
    #    return self.__dict__[attrName]
    # So we need to write out every function by hand

    def Allreduce(self, tensor: torch.Tensor, op: int) -> torch.Tensor:
        return self._comm.Allreduce(tensor, op)

    def Bcast_(self, tensor: torch.Tensor, root: int) -> torch.Tensor:
        return self._comm.Bcast_(tensor, root)

    def Reduce_(self, tensor: torch.Tensor, op: int, root: int) -> torch.Tensor:
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

COMM_WORLD = MPI_Communicator(torch.ops.torchmpi.COMM_WORLD())

def comm_from_mpi4py(comm: __mpi4py_MPI.Comm) -> MPI_Communicator:
    fortran_handle = comm.py2f();
    return MPI_Communicator(torch.ops.torchmpi.comm_from_fortran(fortran_handle))
