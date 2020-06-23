import torch
import mpi4torch
import unittest

comm = mpi4torch.COMM_WORLD

class TestAllreduce(unittest.TestCase):
    def test_simple(self):
        tmp = torch.rand(10, dtype=torch.double).requires_grad_()
        res = comm.Allreduce(tmp,mpi4torch.MPI_SUM)
        res.sum().backward()
        self.assertTrue((tmp.grad == comm.size * torch.ones(10, dtype=torch.double)).all())

    def test_torchscript(self):
        tmp = torch.rand(10, dtype=torch.double).requires_grad_()
        @torch.jit.script
        def myfunc(x,comm_: mpi4torch.MPI_Communicator):
            return comm_.Allreduce(x,mpi4torch.MPI_SUM)
        res = myfunc(tmp,comm)
        res.sum().backward()
        self.assertTrue((tmp.grad == comm.size * torch.ones(10, dtype=torch.double)).all())

class TestReduce(unittest.TestCase):
    def test_simple_inplace(self):
        tmp = torch.rand(10, dtype=torch.double).requires_grad_()
        res = comm.Reduce_(tmp,mpi4torch.MPI_SUM,0)
        res.sum().backward()
        self.assertTrue((tmp.grad == torch.ones(10,dtype=torch.double)).all())

    def test_noinplace_exception(self):
        # the 0. addition is just to make the resulting tmp variable not a leaf in the DAG,
        # sine we currently dont see a way to add this safeguard for leaf nodes.
        tmp = 0. + torch.rand(10, dtype=torch.double).requires_grad_()
        res = tmp + comm.Reduce_(tmp,mpi4torch.MPI_SUM,0)
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

class TestAlltoall(unittest.TestCase):
    def test_gatherscatter_equivalence(self):
        tmp = torch.rand([3,4,1,4,comm.size,2],dtype=torch.double)
        res1 = comm.Scatter(comm.Gather(tmp,2,0),4,1,0)
        res2 = comm.Alltoall(tmp,2,4,1)
        self.assertTrue((res2 == res1).all())

    def test_gatherscatter_equivalence_varying_numelem(self):
        tmp = torch.rand([3,4,comm.rank+1,4,comm.size*(comm.size+1)//2,2],dtype=torch.double)
        res1 = comm.Scatter(comm.Gather(tmp,2,0),4,comm.rank+1,0)
        res2 = comm.Alltoall(tmp,2,4,comm.rank+1)
        self.assertTrue((res2 == res1).all())

    def test_gatheraxis_scatteraxis_equal(self):
        tmp = torch.rand([3,4,comm.rank+1,2],dtype=torch.double)
        tmp[0,0,:,0] = torch.arange(comm.rank*(comm.rank+1)//2, (comm.rank+1)*(comm.rank+2)//2)
        res = comm.Alltoall(tmp,2,2,comm.size-comm.rank)
        total_numelem = comm.size*(comm.size+1)//2
        correct_res = torch.arange(total_numelem - (comm.size-comm.rank)*(comm.size-comm.rank+1)//2,
                                   total_numelem - (comm.size-comm.rank-1)*(comm.size-comm.rank)//2,
                                   dtype=torch.double)
        self.assertTrue((res[0,0,:,0] == correct_res).all())

    def test_identity_equivalence(self):
        tmp = torch.rand([3,4,2,4,3*comm.size,2],dtype=torch.double)
        res = comm.Alltoall(tmp,2,4,3)
        res2 = comm.Alltoall(res,4,2,2)
        self.assertTrue((res2 == tmp).all())

    def test_basic_ad(self):
        tmp = torch.rand([3,4,2,4,comm.size,2],dtype=torch.double).requires_grad_()
        res = comm.Alltoall(tmp,2,4,1)
        res.sum().backward()
        self.assertTrue((tmp.grad == torch.ones_like(tmp)).all())
