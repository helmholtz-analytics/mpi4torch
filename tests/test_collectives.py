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

class TestGather(unittest.TestCase):
    def test_basic_functionality(self):
        numdim = 4
        tmp = torch.rand([2,5,numdim,2,3],dtype=torch.double)
        tmp[0,0,:,0,0] = comm.rank
        res = comm.Gather(tmp, 2, 0)
        if comm.rank == 0:
            tmp2 = torch.squeeze(torch.sum(res[0,0,:,0,0]))
            self.assertTrue((tmp2 == numdim * (comm.size - 1) * comm.size // 2).all())

    def test_basic_ad(self):
        numdim = 4
        tmp = torch.rand([2,5,numdim,2,3],dtype=torch.double).requires_grad_()
        res = comm.Gather(tmp, 2, 0)
        res.sum().backward()
        self.assertTrue((tmp.grad == torch.ones_like(tmp)).all())

class TestAllgather(unittest.TestCase):
    def test_basic_functionality(self):
        numdim = 4
        tmp = torch.rand([2,5,numdim,2,3],dtype=torch.double)
        tmp[0,0,:,0,0] = comm.rank
        res = comm.Allgather(tmp, 2)
        tmp2 = torch.squeeze(torch.sum(res[0,0,:,0,0]))
        self.assertTrue((tmp2 == numdim * (comm.size - 1) * comm.size // 2).all())

    def test_basic_ad(self):
        numdim = 4
        tmp = torch.rand([2,5,numdim,2,3],dtype=torch.double).requires_grad_()
        res = comm.Allgather(tmp, 2)
        res.sum().backward()
        self.assertTrue((tmp.grad == comm.size * torch.ones_like(tmp)).all())

class TestScatter(unittest.TestCase):
    def test_basic_functionality(self):
        if comm.rank == 0:
            tmp = torch.rand([2,5,comm.size,2,3],dtype=torch.double)
            for i in range(comm.size):
                tmp[0,0,i,0,0] = i
        else:
            tmp = torch.rand([1],dtype=torch.double)
        res = comm.Scatter(tmp, 2, 1, 0)
        self.assertTrue((res[0,0,:,0,0] == comm.rank).all())

    def test_scattergather(self):
        if comm.rank == 0:
            tmp = torch.rand([2,5,comm.size,2,3],dtype=torch.double)
        else:
            tmp = torch.rand([1],dtype=torch.double)
        res = comm.Scatter(tmp, 2, 1, 0)
        res2 = comm.Gather(res, 2, 0)
        if comm.rank == 0:
            self.assertTrue((res2 == tmp).all())

    def test_basic_ad(self):
        if comm.rank == 0:
            tmp = torch.rand([2,5,comm.size,2,3],dtype=torch.double).requires_grad_()
        else:
            tmp = torch.rand([1],dtype=torch.double).requires_grad_()
        res = comm.Scatter(tmp, 2, 1, 0)
        res.sum().backward()
        if comm.rank == 0:
            self.assertTrue((tmp.grad == torch.ones_like(tmp)).all())
        else:
            self.assertTrue((tmp.grad == torch.zeros_like(tmp)).all())
