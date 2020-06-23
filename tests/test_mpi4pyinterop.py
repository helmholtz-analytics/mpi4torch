import torch
import mpi4torch
import unittest
import mpi4py.MPI as MPI

class TestMpi4PyInteroperability(unittest.TestCase):
    def test_rank_and_size(self):
        comm1 = MPI.COMM_WORLD
        comm2 = mpi4torch.comm_from_mpi4py(comm1)
        self.assertEqual(comm1.rank, comm2.rank)
        self.assertEqual(comm1.size, comm2.size)

    def test_simple_allreduce(self):
        comm1 = MPI.COMM_WORLD
        comm2 = mpi4torch.comm_from_mpi4py(comm1)
        tmp = torch.rand(10, dtype=torch.double).requires_grad_()
        res = comm2.Allreduce(tmp,mpi4torch.MPI_SUM)
        res.sum().backward()
        self.assertTrue((tmp.grad == comm2.size * torch.ones(10, dtype=torch.double)).all())

