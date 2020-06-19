import torch
import torchmpi
import mpi4py.MPI

comm = torchmpi.COMM_WORLD

torch.manual_seed(42)

num_points = 10000
chunk_size = num_points // comm.size
rest = num_points % comm.size
if comm.rank < rest:
    chunk_size += 1
    offset = chunk_size * comm.rank
else:
    offset = chunk_size * comm.rank + rest

xinput = 2.0 * torch.rand([num_points],dtype=torch.double)[offset:offset+chunk_size]

def some_parametrized_function(inp, params):
    return (params[2] * inp + params[1]) * inp + params[0]

gen_params = torch.tensor([0.1, 1.0, -2.0])

youtput = some_parametrized_function(xinput, gen_params)

def lossfunction(params):
    # average initial params to bring all ranks on the same page
    params = comm.Allreduce(params, torchmpi.MPI_SUM) / comm.size

    # compute local loss
    localloss = torch.sum(torch.square(youtput - some_parametrized_function(xinput, params)))

    # sum up the loss among all ranks
    return comm.Allreduce(localloss, torchmpi.MPI_SUM)

params = torch.arange(3, dtype=torch.double).requires_grad_()

# LBFGS only needs one outer iteration for a linear problem
# with so few parameters
num_iterations = 1
optimizer = torch.optim.LBFGS([params], 1)

for i in range(num_iterations):
    def closure():
        loss = lossfunction(params)
        optimizer.zero_grad()
        loss.backward()
        if comm.rank == 0:
            print("Params: ", params)
            print("Loss  : ", loss)
        return loss
    optimizer.step(closure)

# only print output on rank 0
if comm.rank == 0:
    print("Final parameters: ", params)
