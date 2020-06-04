import torch
import torchmpi

comm = torchmpi.COMM_WORLD

xinput = comm.rank + torch.rand([1000],dtype=torch.double)

def some_parametrized_function(inp, params):
    return (params[2] * inp + params[1]) * inp + params[0]

gen_params = torch.tensor([0.1, 1.0, -2.0])

youtput = some_parametrized_function(xinput, gen_params)

def lossfunction(params):
    # average initial params to bring all ranks on the same page
    params = comm.Allreduce(params,torchmpi.MPI_SUM) / comm.size

    # compute local loss
    localloss = torch.sum(torch.square(youtput - some_parametrized_function(xinput, params)))

    # average loss among all ranks
    return comm.Allreduce(localloss, torchmpi.MPI_SUM) / comm.size

params = torch.arange(3, dtype=torch.double).requires_grad_()

num_iterations = 1 # LBFGS only needs one iteration with so few parameters and a linear problem
optimizer = torch.optim.LBFGS([params], 1)

for i in range(num_iterations):
    def closure():
        loss = lossfunction(params)
        optimizer.zero_grad()
        loss.backward()
        return loss
    optimizer.step(closure)

if comm.rank == 0:
    print("Final parameters: ", params)




