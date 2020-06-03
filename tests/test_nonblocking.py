import torch
import torchmpi
import unittest

comm = torchmpi.COMM_WORLD

class TestNonBlocking(unittest.TestCase):
    def test_simple_isendirecv(self):
        tmp = torch.rand(10000000, dtype=torch.double).requires_grad_()
        req = comm.Isend(tmp,(comm.rank+1)%comm.size,0)
        req2 = comm.Irecv(torchmpi.JoinDummies(torch.empty_like(tmp),[req[0]]),(comm.rank+comm.size-1)%comm.size,0)
        res = comm.Wait(torchmpi.JoinDummiesHandle(req,[req2[0]]))
        res2 = comm.Wait(torchmpi.JoinDummiesHandle(req2,[res]))
        res3 = res2 * comm.rank
        res3.sum().backward()
        self.assertTrue((tmp.grad == ((comm.rank + 1 )%comm.size) * torch.ones_like(tmp)).all())

