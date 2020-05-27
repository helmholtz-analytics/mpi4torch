import torch
from . import _mpi

rank = _mpi.GetRank()
npes = _mpi.GetSize()

def Allreduce(tensor):
    return torch.ops.torchmpi.MPIAllreduce(tensor)

def Bcast_(tensor, root):
    return torch.ops.torchmpi.MPIBcast_(tensor, root)

def Reduce_(tensor, root):
    return torch.ops.torchmpi.MPIReduce_(tensor, root)

def JoinDummies(tensor, *args):
    return torch.ops.torchmpi.JoinDummies(tensor,args)

def Isend(tensor, dest, tag):
    return torch.ops.torchmpi.MPIIsend(tensor, dest, tag)

def Irecv(tensor, dest, tag):
    return torch.ops.torchmpi.MPIIrecv(tensor, dest, tag)

def Wait(req):
    return torch.ops.torchmpi.MPIWait(req)
