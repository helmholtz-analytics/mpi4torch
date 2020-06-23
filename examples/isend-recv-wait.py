import torch
import mpi4torch

comm = mpi4torch.COMM_WORLD

a = torch.tensor([1.0 + comm.rank]).requires_grad_()

handle = comm.Isend(a,(comm.rank+1)%comm.size, 0)
recvbuffer = mpi4torch.JoinDummies(torch.empty_like(a), [handle.dummy])
b = comm.Recv(recvbuffer, (comm.rank-1+comm.size)%comm.size, 0)
wait_ret = comm.Wait(mpi4torch.JoinDummiesHandle(handle,[b]))

res = mpi4torch.JoinDummies(a+b, [wait_ret])
print(res)

res.backward()
print(a.grad)
