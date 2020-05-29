import torch
import torchmpi
import unittest

comm = torchmpi.COMM_WORLD

class TestAllreduce(unittest.TestCase):
    def test_simple(self):
        tmp = torch.rand(10, dtype=torch.double).requires_grad_()
        res = comm.Allreduce(tmp)
        res.sum().backward()
        self.assertTrue((tmp.grad == comm.size * torch.ones(10, dtype=torch.double)).all())

    def test_torchscript(self):
        tmp = torch.rand(10, dtype=torch.double).requires_grad_()
        @torch.jit.script
        def myfunc(x,comm_: torchmpi.MPI_Communicator):
            return comm_.Allreduce(x)
        res = myfunc(tmp,comm)
        res.sum().backward()
        self.assertTrue((tmp.grad == comm.size * torch.ones(10, dtype=torch.double)).all())

class TestReduce(unittest.TestCase):
    def test_simple_inplace(self):
        tmp = torch.rand(10, dtype=torch.double).requires_grad_()
        res = comm.Reduce_(tmp,0)
        res.sum().backward()
        self.assertTrue((tmp.grad == torch.ones(10,dtype=torch.double)).all())

    def test_noinplace_exception(self):
        # the 0. addition is just to make the resulting tmp variable not a leaf in the DAG,
        # sine we currently dont see a way to add this safeguard for leaf nodes.
        tmp = 0. + torch.rand(10, dtype=torch.double).requires_grad_()
        res = tmp + comm.Reduce_(tmp,0)
        with self.assertRaises(RuntimeError):
            res.sum().backward()

class TestBcast(unittest.TestCase):
    def test_simple_inplace(self):
        tmp = torch.rand(10, dtype=torch.double).requires_grad_()
        res = comm.Bcast_(tmp,0)
        res.sum().backward()
        if comm.rank == 0:
            self.assertTrue((tmp.grad == comm.size * torch.ones(10,dtype=torch.double)).all())
        else:
            self.assertTrue((tmp.grad == torch.zeros(10,dtype=torch.double)).all())
