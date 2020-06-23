import torch
import mpi4torch
import unittest

comm = mpi4torch.COMM_WORLD

class TestJoinDummies(unittest.TestCase):
    def test_simple_allreduce(self):
        tmp = torch.rand(10, dtype=torch.double).requires_grad_()
        tmp2 = torch.rand(10, dtype=torch.double).requires_grad_()
        tmp3 = torch.rand(10, dtype=torch.double).requires_grad_()
        res = comm.Allreduce(tmp,mpi4torch.MPI_SUM)
        res2 = mpi4torch.JoinDummies(res,[tmp2,tmp3])
        res2.sum().backward()
        self.assertTrue((tmp2.grad == torch.zeros(10, dtype=torch.double)).all())
        self.assertTrue((tmp3.grad == torch.zeros(10, dtype=torch.double)).all())
        self.assertTrue((tmp.grad == comm.size * torch.ones(10, dtype=torch.double)).all())

