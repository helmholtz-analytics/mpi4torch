import torch
from . import _mpi
from typing import List

WaitHandle = List[torch.Tensor]

@torch.jit.script
def JoinDummies(tensor: torch.Tensor, args:List[torch.Tensor]) -> torch.Tensor:
    return torch.ops.torchmpi.JoinDummies(tensor, args)

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

    def Allreduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._comm.Allreduce(tensor)

    def Bcast_(self, tensor: torch.Tensor, root: int) -> torch.Tensor:
        return self._comm.Bcast_(tensor, root)

    def Reduce_(self, tensor: torch.Tensor, root: int) -> torch.Tensor:
        return self._comm.Reduce_(tensor, root)

    def Gather(self, tensor: torch.Tensor, gatheraxis: int, root: int) -> torch.Tensor:
        return self._comm.Gather(tensor, gatheraxis, root)

    def Allgather(self, tensor: torch.Tensor, gatheraxis: int) -> torch.Tensor:
        return self._comm.Allgather(tensor, gatheraxis)

    def Scatter(self, tensor: torch.Tensor, scatteraxis: int, numelem: int, root: int) -> torch.Tensor:
        return self._comm.Scatter(tensor, scatteraxis, numelem, root)

    def Isend(self, tensor: torch.Tensor, dest: int, tag: int) -> WaitHandle:
        return self._comm.Isend(tensor, dest, tag)

    def Irecv(self, tensor: torch.Tensor, source: int, tag: int) -> WaitHandle:
        return self._comm.Irecv(tensor, source, tag)

    def Wait(self, waithandle: WaitHandle) -> torch.Tensor:
        return self._comm.Wait(waithandle)

COMM_WORLD = MPI_Communicator(torch.ops.torchmpi.COMM_WORLD())
