import torch
from . import _mpi

class MPI_Comm_Pywrapper:
    def __init__(self,comm):
        self._comm = comm

    @property
    def rank(self):
        return self._comm.GetRank()

    @property
    def size(self):
        return self._comm.GetSize()

    def __getattr__(self, attrName):
        if attrName in self.__dict__["_comm"]._method_names():
            return self.__dict__["_comm"].__getattr__(attrName)
        return self.__dict__[attrName]

COMM_WORLD = MPI_Comm_Pywrapper(torch.ops.torchmpi.COMM_WORLD())

def JoinDummies(tensor, *args):
    return torch.ops.torchmpi.JoinDummies(tensor, args)
