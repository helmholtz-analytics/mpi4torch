import torch
import torchmpi
import unittest

class TestNonBlocking(unittest.TestCase):
    def test_simple_isendirecv(self):
        tmp = torch.rand(10000000, dtype=torch.double).requires_grad_()
        req = torchmpi.Isend(tmp,(torchmpi.rank+1)%torchmpi.npes,0)
        req2 = torchmpi.Irecv(torchmpi.JoinDummies(torch.empty_like(tmp),req[0]),(torchmpi.rank+torchmpi.npes-1)%torchmpi.npes,0)
        res = torchmpi.JoinDummies(torchmpi.Wait(req),req2[0])
        res2 = torchmpi.JoinDummies(torchmpi.Wait(req2),res)
        print("Rank:", torchmpi.rank, "Send:", tmp)
        print("Rank:", torchmpi.rank, "Recv:",res2)
        res3 = res2 * torchmpi.rank
        res3.sum().backward()
        print("Grad: ", tmp.grad)
        self.assertTrue((tmp.grad == ((torchmpi.rank + 1 )%torchmpi.npes) * torch.ones_like(tmp)).all())

