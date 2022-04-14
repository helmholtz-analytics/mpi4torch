
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/utils.h>
//#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/script.h>
#include <torch/extension.h>

#include <mpi.h>

#if defined(OPEN_MPI) && OPEN_MPI
// Needed for checking cuda-awareness
#include <mpi-ext.h>
#endif

#include <iostream>
#include <stdexcept>
#include <sstream>

using torch::Tensor;
using torch::ScalarType;
using torch::autograd::variable_list;

namespace
{

#if defined(MPIX_CUDA_AWARE_SUPPORT)
// if it is at compiletime already clear that OpenMPI has now support, we deactivate
// the cuda-aware mpi support directly
#define MPI4TORCH_BUILT_WITH_CUDA_AWARENESS MPIX_CUDA_AWARE_SUPPORT
#else
#define MPI4TORCH_BUILT_WITH_CUDA_AWARENESS 0
#endif

#if MPI4TORCH_BUILT_WITH_CUDA_AWARENESS

bool have_cuda_aware_mpi_support = false;

void inline __setup_have_cuda_aware_mpi_support()
{
    // OpenMPI (and presumably also Parastation MPI) provides this runtime query function
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    have_cuda_aware_mpi_support = MPIX_Query_cuda_support();
#else
    have_cuda_aware_mpi_support = false;
#endif
}

#else
const bool have_cuda_aware_mpi_support = false;
#endif

void deactivate_cuda_aware_mpi_support()
{
#if MPI4TORCH_BUILT_WITH_CUDA_AWARENESS
    have_cuda_aware_mpi_support = false;
#endif
}

struct MPIDeviceHelper
{
    MPIDeviceHelper(const Tensor& input)
        : device(input.device()), devicetype(device.type()), mpidevice(c10::kCPU)
    {
        setup();
    }

    MPIDeviceHelper(const c10::Device& _device)
        : device(_device), devicetype(device.type()), mpidevice(c10::kCPU)
    {
        setup();
    }

    Tensor fromDeviceToMPI(const Tensor& input)
    {
        if (input.device() == mpidevice) {
            return input;
        }
        return input.to(mpidevice);
    }

    Tensor fromMPIToDevice(const Tensor& output)
    {
        if (output.device() == device) {
            return output;
        }
        return output.to(device);
    }

    c10::Device device;
    c10::DeviceType devicetype;
    c10::Device mpidevice;

private:
    void setup()
    {
        if (devicetype == c10::kCPU) {
            mpidevice = device;
        } else if (devicetype == c10::kCUDA && have_cuda_aware_mpi_support) {
            mpidevice = device;
        }
    }
};

MPI_Datatype torch2mpitype(ScalarType in)
{
    switch(in)
    {
    case ScalarType::Byte:
        return MPI_BYTE;
    case ScalarType::Char:
        return MPI_CHAR;
    case ScalarType::Short:
        return MPI_SHORT;
    case ScalarType::Int:
        return MPI_INT;
    case ScalarType::Long:
        return MPI_LONG;
    case ScalarType::Float:
        return MPI_FLOAT;
    case ScalarType::Double:
        return MPI_DOUBLE;
    default:
        break;
        // just to silence compiler warnings of unhandeled switch cases
    }
    throw std::invalid_argument("Failure to match torch::ScalarType to MPI_Datatype!");
}

void check_mpi_return_value(int ierr)
{
    if (ierr != MPI_SUCCESS) {
        std::ostringstream oss;
        oss << ierr;
        throw std::runtime_error("MPI call failed with error code " + oss.str());
    }
}

struct MPI_Comm_Wrapper : torch::CustomClassHolder
{
    MPI_Comm_Wrapper(const MPI_Comm comm_ = MPI_COMM_NULL) : comm(comm_) {}

    MPI_Comm comm;

    int64_t GetRank();
    int64_t GetSize();

    Tensor MPIAllreduce(const Tensor& input, int64_t op);
    Tensor MPIBcast_(const Tensor& input, int64_t root);
    Tensor MPIReduce_(const Tensor& input, int64_t op, int64_t root);

    Tensor MPIGather(const Tensor& input, int64_t gatheraxis, int64_t root);
    Tensor MPIAllgather(const Tensor& input, int64_t gatheraxis);
    Tensor MPIScatter(const Tensor& input, int64_t scatteraxis, int64_t numelem, int64_t root);
    Tensor MPIAlltoall(const Tensor& input, int64_t gatheraxis, int64_t scatteraxis, int64_t numelem);

    variable_list MPIIsend(const Tensor& input, int64_t dest, int64_t tag);
    variable_list MPIIrecv(const Tensor& input, int64_t source, int64_t tag);
    Tensor MPIWait(const variable_list& input);
};

c10::intrusive_ptr<MPI_Comm_Wrapper> comm_world()
{
    return c10::make_intrusive<MPI_Comm_Wrapper>(MPI_COMM_WORLD);
}

c10::intrusive_ptr<MPI_Comm_Wrapper> comm_from_fortran(int64_t fortran_handle)
{
    return c10::make_intrusive<MPI_Comm_Wrapper>(MPI_Comm_f2c(fortran_handle));
}

Tensor JoinDummies(const Tensor& loopthrough, const variable_list& list);

int64_t MPI_Comm_Wrapper::GetRank()
{
    int rank;
    MPI_Comm_rank(comm,&rank);
    return rank;
}

int64_t MPI_Comm_Wrapper::GetSize()
{
    int size;
    MPI_Comm_size(comm,&size);
    return size;
}

struct MPIBackwardNode : public torch::autograd::Node
{
    MPI_Comm_Wrapper comm;
};

struct MPIUnimplementedNode : MPIBackwardNode
{
    variable_list apply(variable_list&& grads) override {
        throw std::runtime_error("This backward operation is currently unimplemented!");
    }
    std::string name() const override {
        return std::string("MPIUnimplementedNode");
    }
};

enum Mpi4torchCollectiveOps : int64_t
{
    mpi4torch_op_max,
    mpi4torch_op_min,
    mpi4torch_op_sum,
    mpi4torch_op_prod,
    mpi4torch_op_land,
    mpi4torch_op_band,
    mpi4torch_op_lor,
    mpi4torch_op_bor,
    mpi4torch_op_lxor,
    mpi4torch_op_bxor,
    mpi4torch_op_minloc,
    mpi4torch_op_maxloc
};

MPI_Op __get_mpi_op(int64_t op)
{
    switch(op)
    {
    case mpi4torch_op_max:
        return MPI_MAX;
    case mpi4torch_op_min:
        return MPI_MIN;
    case mpi4torch_op_sum:
        return MPI_SUM;
    case mpi4torch_op_prod:
        return MPI_PROD;
    case mpi4torch_op_land:
        return MPI_LAND;
    case mpi4torch_op_band:
        return MPI_BAND;
    case mpi4torch_op_lor:
        return MPI_LOR;
    case mpi4torch_op_bor:
        return MPI_BOR;
    case mpi4torch_op_lxor:
        return MPI_LXOR;
    case mpi4torch_op_bxor:
        return MPI_BXOR;
    case mpi4torch_op_minloc:
        return MPI_MINLOC;
    case mpi4torch_op_maxloc:
        return MPI_MAXLOC;
    default:
        break;
    }
    throw std::invalid_argument("mpi4torch: Collective operation not supported!");
}

struct MPIAllreduceSumBackward : public MPIBackwardNode {
    variable_list apply(variable_list&& grads) override;
    std::string name() const override {
        return std::string("MPIAllreduceSumBackward");
    }

    void release_variables() override {
        return;
    }
};

variable_list MPIAllreduceSumBackward::apply (variable_list&& grads)
{
    variable_list grad_inputs(1);
    if (should_compute_output(0)) {
        grad_inputs[0] = comm.MPIAllreduce(grads[0], mpi4torch_op_sum);
    }
    return grad_inputs;
}

Tensor MPI_Comm_Wrapper::MPIAllreduce(const Tensor& input, int64_t op)
{
    std::shared_ptr<MPIBackwardNode> grad_fn;
    auto mpiop = __get_mpi_op(op);
    if (torch::autograd::compute_requires_grad(input)) {
        if (op == mpi4torch_op_sum) {
            grad_fn = std::shared_ptr<MPIAllreduceSumBackward> (new MPIAllreduceSumBackward(), torch::autograd::deleteNode);
        } else {
            grad_fn = std::shared_ptr<MPIUnimplementedNode>(new MPIUnimplementedNode(), torch::autograd::deleteNode);
        }
        grad_fn->comm = *this;
        grad_fn->set_next_edges(torch::autograd::collect_next_edges(input));
    }
    auto result = ([&]() {
        at::AutoDispatchBelowADInplaceOrView guard;

        MPIDeviceHelper devhelper(input);

        // make input contiguous
        auto input_cont = devhelper.fromDeviceToMPI(input).contiguous();

        auto recv = torch::empty_like(input_cont);

        check_mpi_return_value(
            MPI_Allreduce(input_cont.data_ptr(), recv.data_ptr(), input_cont.numel(),
                      torch2mpitype(input_cont.scalar_type()), mpiop, comm)
        );

        return devhelper.fromMPIToDevice(recv);
    })();
    if (grad_fn) {
        set_history(torch::autograd::flatten_tensor_args(result), grad_fn);
    }
    return result;
}

struct MPIBcastInPlaceBackward : public MPIBackwardNode {
    MPIBcastInPlaceBackward(int _root) : root(_root) {}
    variable_list apply(variable_list&& grads) override;
    std::string name() const override {
        return std::string("MPIBcastInPlaceBackward");
    }

    void release_variables() override {
        return;
    }

    int root;
};

variable_list MPIBcastInPlaceBackward::apply (variable_list&& grads)
{
    variable_list grad_inputs(1);
    if (should_compute_output(0)) {
        grad_inputs[0] = comm.MPIReduce_(grads[0],mpi4torch_op_sum,root);
    }
    return grad_inputs;
}

Tensor MPI_Comm_Wrapper::MPIBcast_(const Tensor& input, int64_t root)
{
    // TODO: check for root being in int range
    std::shared_ptr<MPIBcastInPlaceBackward> grad_fn;
    if (torch::autograd::compute_requires_grad(input)) {
        grad_fn = std::shared_ptr<MPIBcastInPlaceBackward> (new MPIBcastInPlaceBackward(static_cast<int>(root)),
                                                     torch::autograd::deleteNode);
        grad_fn->comm = *this,
        grad_fn->set_next_edges(torch::autograd::collect_next_edges(input));
    }
    auto result = ([&]() {
        at::AutoDispatchBelowADInplaceOrView guard;

        MPIDeviceHelper devhelper(input);

        // 1. Make input contiguous
        // 2. Call variable_data() to make a shallow copy of the input tensor without the autograd history,
        //    such that it can be savely returned from this function.
        auto input_cont = devhelper.fromDeviceToMPI(input).contiguous().variable_data();

        check_mpi_return_value(
            MPI_Bcast(input_cont.data_ptr(), input_cont.numel(),
                  torch2mpitype(input_cont.scalar_type()),
                  static_cast<int>(root), comm)
        );

        return devhelper.fromMPIToDevice(input_cont);
    })();
    if (grad_fn) {
        set_history(torch::autograd::flatten_tensor_args(result), grad_fn);
    }
    return result;
}

struct MPIReduceSumInPlaceBackward : public MPIBackwardNode {
    MPIReduceSumInPlaceBackward(int _root) : root(_root) {}
    variable_list apply(variable_list&& grads) override;
    std::string name() const override {
        return std::string("MPIReduceSumInPlaceBackward");
    }

    void release_variables() override {
        return;
    }

    int root;
};

variable_list MPIReduceSumInPlaceBackward::apply (variable_list&& grads)
{
    variable_list grad_inputs(1);
    // TODO: for these simple functions the should_compute_output check is superfluous
    if (should_compute_output(0)) {
        // NOTE: It is probably safe to use in-place operations in the backward mode,
        //       since I currently cannot think of any way how a bifurcation could
        //       enter the DAG.
        // TODO: Proof that it is safe!
        grad_inputs[0] = comm.MPIBcast_(grads[0],root);
    }
    return grad_inputs;
}

struct MPINoInplaceBackward : public torch::autograd::Node {
    variable_list apply(variable_list&& grads) override
    {
        throw std::runtime_error("Reuse of variables passed to in-place MPI kernels not supported! Try using the return value");
    }
    std::string name() const override {
        return std::string("MPINoInplaceBackward");
    }
};

Tensor MPI_Comm_Wrapper::MPIReduce_(const Tensor& input, int64_t op, int64_t root)
{
    // TODO: check for root being in int range
    std::shared_ptr<MPIBackwardNode> grad_fn;
    auto mpiop = __get_mpi_op(op);
    if (torch::autograd::compute_requires_grad(input)) {
        if (op == mpi4torch_op_sum) {
            grad_fn = std::shared_ptr<MPIReduceSumInPlaceBackward> (new MPIReduceSumInPlaceBackward(static_cast<int>(root)),
                                                      torch::autograd::deleteNode);
        } else {
            grad_fn = std::shared_ptr<MPIUnimplementedNode> (new MPIUnimplementedNode(), torch::autograd::deleteNode);
        }
        grad_fn->comm = *this;
        grad_fn->set_next_edges(torch::autograd::collect_next_edges(input));
    }
    auto result = ([&]() {
        at::AutoDispatchBelowADInplaceOrView guard;

        MPIDeviceHelper devhelper(input);

        // 1. Make input contiguous
        // 2. Call variable_data() to make a shallow copy of the input tensor without the autograd history,
        //    such that it can be savely returned from this function.
        auto input_cont = devhelper.fromDeviceToMPI(input).contiguous().variable_data();

        const int rank = GetRank();

        void* sendbuf = input_cont.data_ptr();
        if (rank == root) {
            // One is only allowed to pass MPI_IN_PLACE for the root process
            // cf. https://stackoverflow.com/a/17744793
            sendbuf = MPI_IN_PLACE;
        }

        check_mpi_return_value(MPI_Reduce(sendbuf, input_cont.data_ptr(), input_cont.numel(),
                                          torch2mpitype(input_cont.scalar_type()),
                                          mpiop, static_cast<int>(root), comm));

        if (rank != root) {
            // We fill the non-root results with zeros to make the function properly behaved.
            // TODO: We could potentially let the return-value be undefined and save some ops?
            input_cont.zero_();
        }

        return devhelper.fromMPIToDevice(input_cont);
    })();
    if (grad_fn) {
        set_history(torch::autograd::flatten_tensor_args(result), grad_fn);

        // We only activate this safeguard if the input variable is not a leaf in the DAG,
        // otherwise we would get errors from the AccumulateGrad Node.
        if (input.grad_fn()) {
            // prohibit misuse of input in autograd
            auto& input_non_const = const_cast<Tensor&>(input);
            set_history(input_non_const, std::shared_ptr<MPINoInplaceBackward>(new MPINoInplaceBackward(),
                                                                     torch::autograd::deleteNode));
        }
    }
    return result;
}

struct MPIGatherBackward : public MPIBackwardNode {
    MPIGatherBackward(int64_t _gatheraxis, int64_t _root)
        : gatheraxis(_gatheraxis), root(_root) {}
    variable_list apply(variable_list&& grads) override;
    std::string name() const override {
        return std::string("MPIGatherBackward");
    }

    void release_variables() override {
        return;
    }

    int64_t gatheraxis;
    int64_t root;
};

variable_list MPIGatherBackward::apply (variable_list&& grads)
{
    variable_list grad_inputs(1);
    // TODO: for these simple functions the should_compute_output check is superfluous
    if (should_compute_output(0)) {
        auto next_node = next_edge(0).function;
        auto input_nr = next_edge(0).input_nr;
        const int64_t numelem = next_node->input_metadata(input_nr).shape()[(size_t) gatheraxis];
        grad_inputs[0] = comm.MPIScatter(grads[0],gatheraxis, numelem, root);
    }
    return grad_inputs;
}

Tensor MPI_Comm_Wrapper::MPIGather(const Tensor& input, int64_t gatheraxis, int64_t root)
{
    // TODO: check for root being in int range
    std::shared_ptr<MPIGatherBackward> grad_fn;
    if (torch::autograd::compute_requires_grad(input)) {
        grad_fn = std::shared_ptr<MPIGatherBackward>
            (new MPIGatherBackward(gatheraxis, root),
             torch::autograd::deleteNode);
        grad_fn->comm = *this;
        grad_fn->set_next_edges(torch::autograd::collect_next_edges(input));
    }
    auto result = ([&]() {
        at::AutoDispatchBelowADInplaceOrView guard;

        MPIDeviceHelper devhelper(input);

        // 1. Make input contiguous
        auto input_cont = devhelper.fromDeviceToMPI(input).contiguous();

        auto sizes = input_cont.sizes();
        const size_t ndim = sizes.size();
        const int npes = (int)GetSize();

        int64_t beforegatheraxis64 = 1;
        for (size_t i = 0; i < (size_t)gatheraxis; ++i) {
            beforegatheraxis64 *= sizes[i];
        }

        int64_t aftergatheraxis64 = 1;
        for (size_t i = (size_t)gatheraxis + 1; i < ndim; ++i) {
            aftergatheraxis64 *= sizes[i];
        }

        if (beforegatheraxis64 > INT_MAX || aftergatheraxis64 * sizes[(size_t)gatheraxis] > INT_MAX) {
            throw std::runtime_error("MPI_Gather: Tensor sizes exceed INT_MAX!");
        }

        const int beforegatheraxis = (int)beforegatheraxis64;
        const int gatheraxissize = (int)sizes[(size_t)gatheraxis];
        const int aftergatheraxis = (int)aftergatheraxis64;

        std::vector<int> recvcounts(npes); // TODO: only allocate on root process

        check_mpi_return_value(MPI_Gather(&gatheraxissize, 1, MPI_INT,
                                          &recvcounts[0], 1,
                                          MPI_INT, root, comm));

        std::vector<int> displs(npes); // TODO: only allocate on root process
        //displs[0] = 0; // This is a noop
        for (size_t i = 1; i < (size_t)npes; ++i) {
            int64_t tmpadd = (int64_t) displs[i-1] + (int64_t) recvcounts[i-1];

            if (tmpadd > INT_MAX) {
                throw std::runtime_error("MPI_Gather: Tensor sizes exceed INT_MAX!");
            }
            displs[i] = (int) tmpadd;
        }
        const int newgatheraxissize = displs[npes-1] + recvcounts[npes-1]; // TODO: add overflow check

        MPI_Datatype tmpdatatype1;
        check_mpi_return_value(MPI_Type_vector(beforegatheraxis, aftergatheraxis, aftergatheraxis * gatheraxissize,
                                               torch2mpitype(input_cont.scalar_type()), &tmpdatatype1));
        check_mpi_return_value(MPI_Type_commit(&tmpdatatype1));

        MPI_Datatype sendtype;
        MPI_Aint basic_lb, basic_extent;
        check_mpi_return_value(MPI_Type_get_extent(torch2mpitype(input_cont.scalar_type()), &basic_lb,
                                                   &basic_extent));
        check_mpi_return_value(MPI_Type_create_resized(tmpdatatype1, 0, aftergatheraxis * basic_extent,
                                                       &sendtype));
        check_mpi_return_value(MPI_Type_commit(&sendtype));

        MPI_Datatype tmpdatatype2;
        check_mpi_return_value(MPI_Type_vector(beforegatheraxis, aftergatheraxis,
                                               newgatheraxissize * aftergatheraxis,
                                               torch2mpitype(input_cont.scalar_type()), &tmpdatatype2));
        check_mpi_return_value(MPI_Type_commit(&tmpdatatype2));
        MPI_Datatype recvtype;
        check_mpi_return_value(MPI_Type_create_resized(tmpdatatype2, 0, aftergatheraxis * basic_extent,
                                                       &recvtype));
        check_mpi_return_value(MPI_Type_commit(&recvtype));

        std::vector<int64_t> newsizes(sizes.begin(), sizes.end());
        newsizes[(size_t)gatheraxis] = newgatheraxissize;

        auto recvtensor = torch::empty(newsizes, input_cont.options(), c10::MemoryFormat::Contiguous);

        check_mpi_return_value(MPI_Gatherv(input_cont.data_ptr(), gatheraxissize, sendtype,
                                           recvtensor.data_ptr(), &recvcounts[0], &displs[0], recvtype,
                                           static_cast<int>(root), comm));

        check_mpi_return_value(MPI_Type_free(&tmpdatatype1));
        check_mpi_return_value(MPI_Type_free(&tmpdatatype2));
        check_mpi_return_value(MPI_Type_free(&sendtype));
        check_mpi_return_value(MPI_Type_free(&recvtype));

        return devhelper.fromMPIToDevice(recvtensor);
    })();
    if (grad_fn) {
        set_history(torch::autograd::flatten_tensor_args(result), grad_fn);
    }
    return result;
}

struct MPIAllgatherBackward : public MPIBackwardNode {
    MPIAllgatherBackward(int64_t _gatheraxis) : gatheraxis(_gatheraxis) {}
    variable_list apply(variable_list&& grads) override;
    std::string name() const override {
        return std::string("MPIAllgatherBackward");
    }

    void release_variables() override {
        return;
    }

    int64_t gatheraxis;
};

variable_list MPIAllgatherBackward::apply (variable_list&& grads)
{
    variable_list grad_inputs(1);
    // TODO: for these simple functions the should_compute_output check is superfluous
    if (should_compute_output(0)) {
        auto next_node = next_edge(0).function;
        auto input_nr = next_edge(0).input_nr;
        const int64_t numelem = next_node->input_metadata(input_nr).shape()[(size_t) gatheraxis];
        grad_inputs[0] = comm.MPIScatter(grads[0], gatheraxis, numelem, 0);
        for (int64_t root = 1; root < comm.GetSize(); ++root) {
            grad_inputs[0] += comm.MPIScatter(grads[0], gatheraxis, numelem, 1);
        }
    }
    return grad_inputs;
}

Tensor MPI_Comm_Wrapper::MPIAllgather(const Tensor& input, int64_t gatheraxis)
{
    // TODO: check for root being in int range
    std::shared_ptr<MPIAllgatherBackward> grad_fn;
    if (torch::autograd::compute_requires_grad(input)) {
        grad_fn = std::shared_ptr<MPIAllgatherBackward>
            (new MPIAllgatherBackward(gatheraxis), torch::autograd::deleteNode);
        grad_fn->comm = *this;
        grad_fn->set_next_edges(torch::autograd::collect_next_edges(input));
    }
    auto result = ([&]() {
        at::AutoDispatchBelowADInplaceOrView guard;

        MPIDeviceHelper devhelper(input);

        // 1. Make input contiguous
        auto input_cont = devhelper.fromDeviceToMPI(input).contiguous();

        auto sizes = input_cont.sizes();
        const size_t ndim = sizes.size();
        const int npes = (int)GetSize();

        int64_t beforegatheraxis64 = 1;
        for (size_t i = 0; i < (size_t)gatheraxis; ++i) {
            beforegatheraxis64 *= sizes[i];
        }

        int64_t aftergatheraxis64 = 1;
        for (size_t i = (size_t)gatheraxis + 1; i < ndim; ++i) {
            aftergatheraxis64 *= sizes[i];
        }

        if (beforegatheraxis64 > INT_MAX || aftergatheraxis64 * sizes[(size_t)gatheraxis] > INT_MAX) {
            throw std::runtime_error("MPI_Gather: Tensor sizes exceed INT_MAX!");
        }

        const int beforegatheraxis = (int)beforegatheraxis64;
        const int gatheraxissize = (int)sizes[(size_t)gatheraxis];
        const int aftergatheraxis = (int)aftergatheraxis64;

        std::vector<int> recvcounts(npes);

        check_mpi_return_value(MPI_Allgather(&gatheraxissize, 1, MPI_INT,
                                             &recvcounts[0], 1,
                                             MPI_INT, comm));

        std::vector<int> displs(npes);
        //displs[0] = 0; // This is a noop
        for (size_t i = 1; i < (size_t)npes; ++i) {
            int64_t tmpadd = (int64_t) displs[i-1] + (int64_t) recvcounts[i-1];

            if (tmpadd > INT_MAX) {
                throw std::runtime_error("MPI_Gather: Tensor sizes exceed INT_MAX!");
            }
            displs[i] = (int) tmpadd;
        }
        const int newgatheraxissize = displs[npes-1] + recvcounts[npes-1];

        MPI_Datatype tmpdatatype1;
        check_mpi_return_value(MPI_Type_vector(beforegatheraxis, aftergatheraxis, aftergatheraxis * gatheraxissize,
                                               torch2mpitype(input_cont.scalar_type()), &tmpdatatype1));
        check_mpi_return_value(MPI_Type_commit(&tmpdatatype1));

        MPI_Datatype sendtype;
        MPI_Aint basic_lb, basic_extent;
        check_mpi_return_value(MPI_Type_get_extent(torch2mpitype(input_cont.scalar_type()), &basic_lb,
                                                   &basic_extent));
        check_mpi_return_value(MPI_Type_create_resized(tmpdatatype1, 0, aftergatheraxis * basic_extent,
                                                       &sendtype));
        check_mpi_return_value(MPI_Type_commit(&sendtype));

        MPI_Datatype tmpdatatype2;
        check_mpi_return_value(MPI_Type_vector(beforegatheraxis, aftergatheraxis,
                                               newgatheraxissize * aftergatheraxis,
                                               torch2mpitype(input_cont.scalar_type()), &tmpdatatype2));
        check_mpi_return_value(MPI_Type_commit(&tmpdatatype2));
        MPI_Datatype recvtype;
        check_mpi_return_value(MPI_Type_create_resized(tmpdatatype2, 0, aftergatheraxis * basic_extent,
                                                       &recvtype));
        check_mpi_return_value(MPI_Type_commit(&recvtype));

        std::vector<int64_t> newsizes(sizes.begin(), sizes.end());
        newsizes[(size_t)gatheraxis] = newgatheraxissize;

        auto recvtensor = torch::empty(newsizes, input_cont.options(), c10::MemoryFormat::Contiguous);

        check_mpi_return_value(MPI_Allgatherv(input_cont.data_ptr(), gatheraxissize, sendtype,
                                              recvtensor.data_ptr(), &recvcounts[0], &displs[0], recvtype,
                                              comm));

        check_mpi_return_value(MPI_Type_free(&tmpdatatype1));
        check_mpi_return_value(MPI_Type_free(&tmpdatatype2));
        check_mpi_return_value(MPI_Type_free(&sendtype));
        check_mpi_return_value(MPI_Type_free(&recvtype));

        return devhelper.fromMPIToDevice(recvtensor);
    })();
    if (grad_fn) {
        set_history(torch::autograd::flatten_tensor_args(result), grad_fn);
    }
    return result;
}

struct MPIScatterBackward : public MPIBackwardNode {
    MPIScatterBackward(int64_t _scatteraxis, int64_t _root)
        : scatteraxis(_scatteraxis), root(_root) {}
    variable_list apply(variable_list&& grads) override;
    std::string name() const override {
        return std::string("MPIScatterBackward");
    }

    void release_variables() override {
        return;
    }

    int64_t scatteraxis;
    int64_t root;
};

variable_list MPIScatterBackward::apply (variable_list&& grads)
{
    variable_list grad_inputs(1);
    // TODO: for these simple functions the should_compute_output check is superfluous
    if (should_compute_output(0)) {
        auto tmp = comm.MPIGather(grads[0],scatteraxis, root);
        if (comm.GetRank() == root) {
            grad_inputs[0] = tmp;
        } else {
            auto next_node = next_edge(0).function;
            auto input_nr = next_edge(0).input_nr;
            grad_inputs[0] = JoinDummies(next_node->input_metadata(input_nr).zeros_like(), {tmp});
        }
    }
    return grad_inputs;
}

Tensor MPI_Comm_Wrapper::MPIScatter(const Tensor& input, int64_t scatteraxis, int64_t numelem, int64_t root)
{
    // TODO: check for root being in int range
    std::shared_ptr<MPIScatterBackward> grad_fn;
    if (torch::autograd::compute_requires_grad(input)) {
        grad_fn = std::shared_ptr<MPIScatterBackward>
            (new MPIScatterBackward(scatteraxis, root),
             torch::autograd::deleteNode);
        grad_fn->comm = *this;
        grad_fn->set_next_edges(torch::autograd::collect_next_edges(input));
    }
    auto result = ([&]() {
        at::AutoDispatchBelowADInplaceOrView guard;

        MPIDeviceHelper devhelper(input);

        // 1. Make input contiguous
        auto input_cont = GetRank() == root ? devhelper.fromDeviceToMPI(input).contiguous() : input;

        size_t ndim = input_cont.sizes().size();
        check_mpi_return_value(MPI_Bcast(&ndim, 1, MPI_LONG, root, comm));
        std::vector<int64_t> sizes;
        if (GetRank() == root) {
            sizes = std::move(input_cont.sizes().vec());
        } else {
            sizes.resize(ndim);
        }
        check_mpi_return_value(MPI_Bcast(&sizes[0], (int)ndim, MPI_LONG, root, comm));

        const int npes = (int)GetSize();

        int64_t beforescatteraxis64 = 1;
        for (size_t i = 0; i < (size_t)scatteraxis; ++i) {
            beforescatteraxis64 *= sizes[i];
        }

        int64_t afterscatteraxis64 = 1;
        for (size_t i = (size_t)scatteraxis + 1; i < ndim; ++i) {
            afterscatteraxis64 *= sizes[i];
        }

        if (beforescatteraxis64 > INT_MAX || afterscatteraxis64 * sizes[(size_t)scatteraxis] > INT_MAX) {
            throw std::runtime_error("MPI_Scatter: Tensor sizes exceed INT_MAX!");
        }

        const int beforescatteraxis = (int)beforescatteraxis64;
        const int scatteraxissize = (int)sizes[(size_t)scatteraxis];
        const int afterscatteraxis = (int)afterscatteraxis64;
        const int newscatteraxissize = (int) numelem;

        std::vector<int> sendcounts(npes); // TODO: only allocate on root process

        check_mpi_return_value(MPI_Gather(&newscatteraxissize, 1, MPI_INT,
                                          &sendcounts[0], 1,
                                          MPI_INT, root, comm));

        std::vector<int> displs(npes); // TODO: only allocate on root process
        //displs[0] = 0; // This is a noop
        for (size_t i = 1; i < (size_t)npes; ++i) {
            int64_t tmpadd = (int64_t) displs[i-1] + (int64_t) sendcounts[i-1];

            if (tmpadd > INT_MAX) {
                throw std::runtime_error("MPI_Scatter: Tensor sizes exceed INT_MAX!");
            }
            displs[i] = (int) tmpadd;
        }
        if (root == GetRank() && scatteraxissize != displs[npes-1] + sendcounts[npes-1]) {
            throw std::runtime_error("MPI_Scatter: finaltensor.shape[scatteraxis] != sum(numelem)!");
        }

        MPI_Datatype tmpdatatype1;
        check_mpi_return_value(MPI_Type_vector(beforescatteraxis, afterscatteraxis,
                                               afterscatteraxis * scatteraxissize,
                                               torch2mpitype(input_cont.scalar_type()), &tmpdatatype1));
        check_mpi_return_value(MPI_Type_commit(&tmpdatatype1));

        MPI_Datatype sendtype;
        MPI_Aint basic_lb, basic_extent;
        check_mpi_return_value(MPI_Type_get_extent(torch2mpitype(input_cont.scalar_type()), &basic_lb,
                                                   &basic_extent));
        check_mpi_return_value(MPI_Type_create_resized(tmpdatatype1, 0, afterscatteraxis * basic_extent,
                                                       &sendtype));
        check_mpi_return_value(MPI_Type_commit(&sendtype));

        MPI_Datatype tmpdatatype2;
        check_mpi_return_value(MPI_Type_vector(beforescatteraxis, afterscatteraxis,
                                               newscatteraxissize * afterscatteraxis,
                                               torch2mpitype(input_cont.scalar_type()), &tmpdatatype2));
        check_mpi_return_value(MPI_Type_commit(&tmpdatatype2));
        MPI_Datatype recvtype;
        check_mpi_return_value(MPI_Type_create_resized(tmpdatatype2, 0, afterscatteraxis * basic_extent,
                                                       &recvtype));
        check_mpi_return_value(MPI_Type_commit(&recvtype));

        std::vector<int64_t> newsizes(sizes.begin(), sizes.end());
        newsizes[(size_t)scatteraxis] = newscatteraxissize;

        auto recvtensor = torch::empty(newsizes, input_cont.options().device(devhelper.mpidevice),
                                       c10::MemoryFormat::Contiguous);

        check_mpi_return_value(MPI_Scatterv(input_cont.data_ptr(), &sendcounts[0], &displs[0], sendtype,
                                            recvtensor.data_ptr(), newscatteraxissize, recvtype,
                                            static_cast<int>(root), comm));

        check_mpi_return_value(MPI_Type_free(&tmpdatatype1));
        check_mpi_return_value(MPI_Type_free(&tmpdatatype2));
        check_mpi_return_value(MPI_Type_free(&sendtype));
        check_mpi_return_value(MPI_Type_free(&recvtype));

        return devhelper.fromMPIToDevice(recvtensor);
    })();
    if (grad_fn) {
        set_history(torch::autograd::flatten_tensor_args(result), grad_fn);
    }
    return result;
}

struct MPIAlltoallBackward : public MPIBackwardNode {
    MPIAlltoallBackward(int64_t _gatheraxis, int64_t _scatteraxis)
        : gatheraxis(_gatheraxis), scatteraxis(_scatteraxis) {}
    variable_list apply(variable_list&& grads) override;
    std::string name() const override {
        return std::string("MPIAlltoallBackward");
    }

    void release_variables() override {
        return;
    }

    int64_t gatheraxis;
    int64_t scatteraxis;
};

variable_list MPIAlltoallBackward::apply (variable_list&& grads)
{
    variable_list grad_inputs(1);
    // TODO: for these simple functions the should_compute_output check is superfluous
    if (should_compute_output(0)) {
        auto next_node = next_edge(0).function;
        auto input_nr = next_edge(0).input_nr;
        const int64_t numelem = next_node->input_metadata(input_nr).shape()[(size_t) gatheraxis];
        grad_inputs[0] = comm.MPIAlltoall(grads[0], scatteraxis, gatheraxis, numelem);
    }
    return grad_inputs;
}

Tensor MPI_Comm_Wrapper::MPIAlltoall(const Tensor& input, int64_t gatheraxis, int64_t scatteraxis, int64_t numelem)
{
    // TODO: check for root being in int range
    std::shared_ptr<MPIAlltoallBackward> grad_fn;
    if (torch::autograd::compute_requires_grad(input)) {
        grad_fn = std::shared_ptr<MPIAlltoallBackward>
            (new MPIAlltoallBackward(gatheraxis, scatteraxis),
             torch::autograd::deleteNode);
        grad_fn->comm = *this;
        grad_fn->set_next_edges(torch::autograd::collect_next_edges(input));
    }
    auto result = ([&]() {
        at::AutoDispatchBelowADInplaceOrView guard;

        // 1. Make input contiguous
        auto input_cont = input.contiguous();


        // TODO: This is probably not the best solution latency-wise, but the total number
        //       of bytes sent should match roughly a solution that uses Alltoallw.
        //       If the latency turns out to become a problem, we should switch to Alltoallw.
        //       Memory usage could also become an issue, since this solution requires
        //       temporarily twice the memory that an Alltoallw solution would require.
        std::vector<torch::Tensor> scattered_tensors;

        if (gatheraxis != scatteraxis) {
            // This is the easy case
            for(int64_t root = 0; root < GetSize(); ++root) {
                scattered_tensors.emplace_back(MPIScatter(input_cont, scatteraxis, numelem, root));
            }
        } else {
            // if gather- and scatteraxis coincide we first need to figure out who receives how many
            // elements from whom
            //
            // TODO: this could probably be solved more efficiently, with few more communication roundtrips

            const size_t npes = (size_t)GetSize();
            const size_t rank = (size_t)GetRank();
            std::vector<int> numelem_cur(npes+1);
            std::vector<int> numelem_new(npes+1);

            const int tmp1 = input_cont.sizes()[(size_t)gatheraxis];
            check_mpi_return_value(MPI_Allgather(&tmp1, 1, MPI_INT,
                                                 &numelem_cur[1], 1,
                                                 MPI_INT, comm));
            const int tmp2 = numelem;
            check_mpi_return_value(MPI_Allgather(&tmp2, 1, MPI_INT,
                                                 &numelem_new[1], 1,
                                                 MPI_INT, comm));
            for (size_t i = 1; i < npes; ++i) {
                numelem_cur[i+1] += numelem_cur[i];
                numelem_new[i+1] += numelem_new[i];
            }

            for(int64_t root = 0; root < npes; ++root) {
                int64_t localnumelem = std::min(numelem_new[rank+1],numelem_cur[root+1])
                                         - std::max(numelem_new[rank],numelem_cur[root]);
                if (localnumelem < 0) {
                    localnumelem = 0;
                }
                scattered_tensors.emplace_back(MPIScatter(input_cont, scatteraxis, localnumelem, root));
            }
        }

        return at::cat(scattered_tensors, gatheraxis);
    })();
    if (grad_fn) {
        set_history(torch::autograd::flatten_tensor_args(result), grad_fn);
    }
    return result;
}

struct JoinDummiesBackward : public torch::autograd::Node {
    variable_list apply(variable_list&& grads) override;
    std::string name() const override {
        return std::string("JoinDummiesBackward");
    }

    void handle_dummies_helper(const size_t first, variable_list& grad_inputs);

    // TODO: I am not sure whether it is wise here to circumvent the saved_variables list,
    // and do our own thing. It might be that we create a memory leak that way!
    Tensor loopthrough;
};

void JoinDummiesBackward::handle_dummies_helper(const size_t first, variable_list& grad_inputs)
{
    for (size_t i = first; i < num_outputs(); ++i) {
        if (should_compute_output(i)) {
            auto next_node = next_edge(i).function;
            auto input_nr = next_edge(i).input_nr;
            grad_inputs[i] = JoinDummies(next_node->input_metadata(input_nr).zeros_like(),{loopthrough});
        }
    }
}

variable_list JoinDummiesBackward::apply (variable_list&& grads)
{
    size_t numoutputs = num_outputs();
    variable_list grad_inputs(numoutputs);
    if (should_compute_output(0)) {
        grad_inputs[0] = JoinDummies(grads[0], {loopthrough});
    }
    handle_dummies_helper(1,grad_inputs);
    return grad_inputs;
}

Tensor JoinDummies(const Tensor& loopthrough, const variable_list& list)
{
    std::shared_ptr<JoinDummiesBackward> grad_fn;
    if (torch::autograd::compute_requires_grad(list)) {
        grad_fn = std::shared_ptr<JoinDummiesBackward> (new JoinDummiesBackward(), torch::autograd::deleteNode);
        grad_fn->set_next_edges(torch::autograd::collect_next_edges(loopthrough,list));
    } else {
        // if none of the dummy variables needs a gradient, we just return the loopthrough variable
        return loopthrough;
    }
    auto result = ([&]() {
        at::AutoDispatchBelowADInplaceOrView guard;

        auto res = loopthrough.variable_data();
        return res;
    })();
    // Checking for grad_fn is unneccessary
    //if (grad_fn) {
        grad_fn->loopthrough = result;
        set_history(torch::autograd::flatten_tensor_args(result), grad_fn);
    //}
    return result;
}

enum NonBlockingOp
{
    Isend_Op,
    Irecv_Op,
};

struct MPINonBlockingBackward : public MPIBackwardNode {
    variable_list apply(variable_list&& grads) override;
    std::string name() const override {
        return std::string("MPINonBlockingBackward");
    }
};

variable_list MPINonBlockingBackward::apply(variable_list&& grads)
{
    variable_list grad_inputs(1);
    // TODO: superfluous check??
    if (should_compute_output(0)) {
        grad_inputs[0] = comm.MPIWait(grads);
    }
    return grad_inputs;
}

variable_list MPI_Comm_Wrapper::MPIIsend(const Tensor& input, int64_t dest, int64_t tag)
{
    // TODO: check for dest and tag being in int's range
    std::shared_ptr<MPINonBlockingBackward> grad_fn;
    if (torch::autograd::compute_requires_grad(input)) {
        grad_fn = std::shared_ptr<MPINonBlockingBackward> (new MPINonBlockingBackward(), torch::autograd::deleteNode);
        grad_fn->comm = *this;
        grad_fn->set_next_edges(torch::autograd::collect_next_edges(input));
    }
    auto result = ([&]() {
        at::AutoDispatchBelowADInplaceOrView guard;

        MPIDeviceHelper devhelper(input);

        // make input contiguous
        // we call variable_data() since we also return the input buffer to ensure it stays in scope
        auto input_cont = devhelper.fromDeviceToMPI(input).contiguous().variable_data();

        MPI_Request req;
        check_mpi_return_value(MPI_Isend(input_cont.data_ptr(), input_cont.numel(),
                  torch2mpitype(input_cont.scalar_type()), static_cast<int>(dest),
                  static_cast<int>(tag), comm, &req));

        auto ret = torch::empty({7},at::kDouble);
        auto fortran_handle = MPI_Request_c2f(req);
        ret[0] = static_cast<double>(fortran_handle);
        ret[1] = static_cast<double>(Isend_Op);
        ret[2] = static_cast<double>(dest);
        ret[3] = static_cast<double>(tag);
        ret[4] = static_cast<double>(0xFFFFFFFF & std::hash<void*>()(input_cont.data_ptr()));
        ret[5] = static_cast<double>(devhelper.devicetype);
        ret[6] = static_cast<double>(devhelper.device.index());
        variable_list retlist;
        retlist.push_back(ret);
        retlist.push_back(input_cont); // make sure the buffer stays in scope!!!
        retlist.push_back(input.variable_data());
        return retlist;
    })();
    if (grad_fn) {
        set_history(torch::autograd::flatten_tensor_args(result), grad_fn);
    }
    return result;
}

variable_list MPI_Comm_Wrapper::MPIIrecv(const Tensor& input, int64_t source, int64_t tag)
{
    // TODO: check for dest and tag being in int's range
    std::shared_ptr<MPINonBlockingBackward> grad_fn;
    if (torch::autograd::compute_requires_grad(input)) {
        grad_fn = std::shared_ptr<MPINonBlockingBackward> (new MPINonBlockingBackward(), torch::autograd::deleteNode);
        grad_fn->comm = *this;
        grad_fn->set_next_edges(torch::autograd::collect_next_edges(input));
    }
    auto result = ([&]() {
        at::AutoDispatchBelowADInplaceOrView guard;

        MPIDeviceHelper devhelper(input);

        // TODO: check whether input is contiguous
        // TODO: Maybe add warning if input device is not mpi device
        auto input_cont = devhelper.fromDeviceToMPI(input).variable_data();

        MPI_Request req;
        check_mpi_return_value(MPI_Irecv(input_cont.data_ptr(), input_cont.numel(),
                  torch2mpitype(input_cont.scalar_type()), static_cast<int>(source),
                  static_cast<int>(tag), comm, &req));

        auto ret = torch::empty({7},at::kDouble);
        auto fortran_handle = MPI_Request_c2f(req);
        ret[0] = static_cast<double>(fortran_handle);
        ret[1] = static_cast<double>(Irecv_Op);
        ret[2] = static_cast<double>(source);
        ret[3] = static_cast<double>(tag);
        ret[4] = static_cast<double>(0xFFFFFFFF & std::hash<void*>()(input_cont.data_ptr()));
        ret[5] = static_cast<double>(devhelper.devicetype);
        ret[6] = static_cast<double>(devhelper.device.index());
        variable_list retlist;
        retlist.push_back(ret);
        retlist.push_back(input_cont); // We ensure that the buffer stays in scope and is not garbage collected
        retlist.push_back(input.variable_data()); // We do this for symmetry reasons, but it is more ISend which needs this
        return retlist;
    })();
    if (grad_fn) {
        set_history(torch::autograd::flatten_tensor_args(result), grad_fn);
    }
    return result;
}

struct MPIWaitBackward : public MPIBackwardNode {
    MPIWaitBackward(NonBlockingOp op, int64_t sourcedest_, int64_t tag_)
        : operation(op), sourcedest(sourcedest_), tag(tag_+10)
    {}
    variable_list apply(variable_list&& grads) override;
    std::string name() const override {
        return std::string("MPIWaitBackward");
    }

    NonBlockingOp operation;
    int64_t sourcedest;
    int64_t tag;
};

variable_list MPIWaitBackward::apply(variable_list&& grads)
{
    // TODO: maybe we should add some offset to the tag, since who knows how intertwined the forward and backward
    // operations can become. Beware of the special value MPI_ANY_TAG

    // TODO: superfluous check??
    if (should_compute_output(0)) {
        auto next_node = next_edge(1).function;
        auto input_nr = next_edge(1).input_nr;

        // Some rationale for this error checking:
        // Everytime when in forward mode a variables is used multiple times, i.e. where the forward graph
        // bifurcates, the backward DAG, as generated by pytorch, has to assume that in the reverse operation
        // the two incoming gradients need to add up. This is however unfortunate for the second part of our
        // wait handle, since this is where we store the send/receive buffer. If one would now add e.g. a zero
        // to this receive buffer a completely new buffer would be created from pytorch, and the original
        // receive buffer would go out of scope, would then be garbage-collected and any actual receive operation
        // potentially yields a seg-fault. On top of that the buffer as returned by MPIWait quite likely
        // contains the wrong result, since MPIWait returns the from pytorch created buffer that stores the
        // result from the addition operation.
        //
        // NOTE: This check cannot capture all cases of misuse. The additional storage of parts/hashes
        //       of the respective data_ptr aims at catching the same type of errors.
        if (next_node->name() != "MPINonBlockingBackward") {
            std::ostringstream oss;
            oss << "mpi4torch: Detected bifurcation in MPIWait handle usage. Next node in DAG"
                   " should be MPINonBlockingBackward, but is "
                << next_node->name() << "!";
            throw std::runtime_error(oss.str());
        }
        switch(operation) {
        case Isend_Op:
            {
            auto buf = next_node->input_metadata(input_nr).zeros_like();
            return comm.MPIIrecv(JoinDummies(buf,grads), sourcedest, tag);
            }
        case Irecv_Op:
            {
            return comm.MPIIsend(grads[0], sourcedest, tag);
            }
        default:
            throw std::runtime_error("Unsupported NonBlockingOp!");
        }
    }
    return variable_list();
}

Tensor MPI_Comm_Wrapper::MPIWait(const variable_list& input)
{
    auto fortran_handle =  static_cast<MPI_Fint>(input[0][0].item<double>());
    MPI_Request req = MPI_Request_f2c(fortran_handle);
    NonBlockingOp operation = static_cast<NonBlockingOp>(input[0][1].item<double>());
    auto sourcedest = static_cast<int64_t>(input[0][2].item<double>());
    auto tag = static_cast<int64_t>(input[0][3].item<double>());
    auto hashvalue = static_cast<uint32_t>(input[0][4].item<double>());
    auto devicetype = static_cast<int16_t>(input[0][5].item<double>());
    auto deviceindex = static_cast<int16_t>(input[0][6].item<double>());

    if (hashvalue != (0xFFFFFFFF & std::hash<void*>()(input[1].data_ptr()))) {
        std::ostringstream oss;
        oss << "mpi4torch: Detected bifurcation in MPIWait handle usage. "
               "Modifying or consuming the handle by other functions than functions from the "
               "MPIWait class is prohibited!";
        throw std::runtime_error(oss.str());
    }

    std::shared_ptr<MPIWaitBackward> grad_fn;
    if(torch::autograd::compute_requires_grad(input)) {
        grad_fn = std::shared_ptr<MPIWaitBackward>(new MPIWaitBackward(operation, sourcedest, tag),torch::autograd::deleteNode);
        grad_fn->comm = *this;
        grad_fn->set_next_edges(torch::autograd::collect_next_edges(input));
    }
    auto result = ([&]() {
        at::AutoDispatchBelowADInplaceOrView guard;

        MPI_Status status; // TODO: Handle use cases for MPI_Status
        check_mpi_return_value(MPI_Wait(&req, & status));

        if (operation == Isend_Op) {
            // We do not do any device conversion for Isend, we then simply return the initial tensor
            return input[2].variable_data();
        }

        MPIDeviceHelper devhelper(c10::Device((c10::DeviceType)devicetype, deviceindex));

        // return a shallow copy of the second input tensor without the autograd strings attached
        return devhelper.fromMPIToDevice(input[1]).variable_data();
    })();
    if (grad_fn) {
        set_history(torch::autograd::flatten_tensor_args(result), grad_fn);
    }
    return result;
}

}

// New-style API is more similar to pybind11, and will in the future be:
static auto mpi_comm_wrapper_registry = torch::class_<::MPI_Comm_Wrapper>("mpi4torch", "MPI_Comm_Wrapper")
    .def("GetRank", &MPI_Comm_Wrapper::GetRank)
    .def("GetSize", &MPI_Comm_Wrapper::GetSize)
    .def("Allreduce", &MPI_Comm_Wrapper::MPIAllreduce)
    .def("Bcast_", &MPI_Comm_Wrapper::MPIBcast_)
    .def("Reduce_", &MPI_Comm_Wrapper::MPIReduce_)
    .def("Gather", &MPI_Comm_Wrapper::MPIGather)
    .def("Allgather", &MPI_Comm_Wrapper::MPIAllgather)
    .def("Scatter", &MPI_Comm_Wrapper::MPIScatter)
    .def("Alltoall", &MPI_Comm_Wrapper::MPIAlltoall)
    .def("Isend", &MPI_Comm_Wrapper::MPIIsend)
    .def("Irecv", &MPI_Comm_Wrapper::MPIIrecv)
    .def("Wait", &MPI_Comm_Wrapper::MPIWait)
    .def_pickle([](const c10::intrusive_ptr<MPI_Comm_Wrapper>& self) -> std::string
        {
            if (self->comm != MPI_COMM_WORLD) {
                throw std::runtime_error("MPI communicators other than MPI_COMM_WORLD are not serializable!");
            }
            return std::string("MPI_COMM_WORLD");
        },
        [](std::string input) -> c10::intrusive_ptr<MPI_Comm_Wrapper>
        {
            if (input == std::string("MPI_COMM_WORLD")) {
                throw std::runtime_error("Unknown MPI communicator");
            }
            return c10::make_intrusive<MPI_Comm_Wrapper>(MPI_COMM_WORLD);
        }
    )
;

// Old-style registration API until pytorch 1.4.0 is
static auto registry = torch::RegisterOperators()
    .op("mpi4torch::COMM_WORLD", &comm_world)
    .op("mpi4torch::comm_from_fortran", &comm_from_fortran)
    .op("mpi4torch::JoinDummies", &JoinDummies);

#if defined(OPEN_MPI)
#if OMPI_MAJOR_VERSION < 3
#define _GNU_SOURCE
#include <dlfcn.h>
#endif
#endif

struct __MPI_Finalizer
{
    ~__MPI_Finalizer()
    {
        check_mpi_return_value(MPI_Finalize());
    }
};

static std::unique_ptr<__MPI_Finalizer> __finalizer;

static void __mpi4torch_mpi_init()
{
#if defined(OPEN_MPI)
#if OMPI_MAJOR_VERSION < 3
    // There are some issues with older versions of OpenMPI which have difficulties with dlopen-ing the MPI
    // libraries the way it is done within pythons extension system (i.e. with RTLD_LOCAL) [1].
    // In detail:
    // - cpython loads this extension as a shared libarry with RTLD_LOCAL
    // - this implies that several openmpi functions are not globally visible
    // - during initalization openmpi itself dynamically loads shared objects,
    //   which backreference to some of the previously loaded (but not visible) symbols
    // - openmpi crashes during the MPI_Init* calls since these backreferenced symbols could
    //   not be resolved
    //
    // [1] https://github.com/open-mpi/ompi/issues/3705

    // In principle we just want to reload libmpi with RTLD_GLOBAL specified before calling MPI_Init_thread
    // to make the respective symbols visible.
    // But we first need to find out by which path name the library goes we try to reload.

    // To do so, we query for the address of a known symbol in libmpi, e.g. MPI_Init_thread
    const void* mpi_init_thread_symbol = dlsym(RTLD_DEFAULT, "MPI_Init_thread");

    if (mpi_init_thread_symbol == nullptr) {
        // in principle we should never be able to reach this point, but better safe than sorry
        throw std::runtime_error(std::string("mpi4torch failed with: ")+std::string(dlerror()));
    }

    // Now we query the library info in which the symbol resides
    Dl_info dlinfo;
    if (!dladdr(mpi_init_thread_symbol, &dlinfo)) {
        throw std::runtime_error(std::string("mpi4torch failed with: ")+std::string(dlerror()));
    }

    // dlinfo.dli_fname should contain the pathname of the mpi library

    // As the man page of dlopen suggests, to promote the already loaded symbols from RTLD_LOCAL to RTLD_GLOBAL,
    // we just need to reopen the library with RTLD_NOLOAD | RTLD_GLOBAL flags.
    const void* mpi_lib_handle = dlopen(dlinfo.dli_fname, RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
    if (!mpi_lib_handle) {
        throw std::runtime_error(std::string("mpi4torch failed with: ")+std::string(dlerror()));
    }
#endif
#endif

    int mpi_initialized = -1;
    check_mpi_return_value(MPI_Initialized(&mpi_initialized));
    if (!mpi_initialized) {
        int provided = 0;
        // We play safe and initialize for multithreaded use cases
        check_mpi_return_value(MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided));

	// TODO: maybe in the future we actually may depend on multithreading,
	//       so we leave this code here
#if 0
	if (provided != MPI_THREAD_MULTIPLE) {
            std::cerr << "mpi4torch WARNING: MPI version does not provide full multithreading support!\n";
            std::cerr << "mpi4torch WARNING: This may crash mpi4torch.\n";
        }
#endif
        __finalizer = std::make_unique<__MPI_Finalizer>();
    } else {
#if 0
        int provided = 0;
	check_mpi_return_value(MPI_Query_thread(&provided));
        if (provided != MPI_THREAD_MULTIPLE) {
            std::cerr << "mpi4torch WARNING: MPI is already initialized but not with MPI_THREAD_MULTIPLE!\n";
            std::cerr << "mpi4torch WARNING: This may crash mpi4torch.\n";
        }
#endif
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Initialize MPI
    __mpi4torch_mpi_init();

#if MPI4TORCH_BUILT_WITH_CUDA_AWARENESS
    __setup_have_cuda_aware_mpi_support();
#endif

    m.def("deactivate_cuda_aware_mpi_support",&deactivate_cuda_aware_mpi_support,
          "Deactivates the CUDA-aware MPI support.\n"
          "\n"
          "Calling this function forces mpi4torch to first move any tensor into main memory before\n"
          "calling a MPI function on it, and then to move the result back into device memory after\n"
          "the MPI call has finished.\n"
          "\n"
          "Note\n"
          "----\n"
          "    This function is useful in situations in which MPI advertises CUDA-awareness but the\n"
          "    functionality is not really supported by the backend.\n");

    // Torchscript does not like the pybind11 enum_ solution
    //py::enum_<Mpi4torchCollectiveOps>(m, "MPI_Op")
    //    .value("MPI_MAX", Mpi4torchCollectiveOps::mpi4torch_op_max)
    //    .value("MPI_MIN", Mpi4torchCollectiveOps::mpi4torch_op_min)
    //    .value("MPI_SUM", Mpi4torchCollectiveOps::mpi4torch_op_sum)
    //    .value("MPI_PROD", Mpi4torchCollectiveOps::mpi4torch_op_prod)
    //    .export_values();

    m.attr("MPI_MAX") = py::int_((int64_t)mpi4torch_op_max);
    m.attr("MPI_MIN") = py::int_((int64_t)mpi4torch_op_min);
    m.attr("MPI_SUM") = py::int_((int64_t)mpi4torch_op_sum);
    m.attr("MPI_PROD") = py::int_((int64_t)mpi4torch_op_prod);
    m.attr("MPI_LAND") = py::int_((int64_t)mpi4torch_op_land);
    m.attr("MPI_BAND") = py::int_((int64_t)mpi4torch_op_band);
    m.attr("MPI_LOR") = py::int_((int64_t)mpi4torch_op_lor);
    m.attr("MPI_BOR") = py::int_((int64_t)mpi4torch_op_bor);
    m.attr("MPI_LXOR") = py::int_((int64_t)mpi4torch_op_lxor);
    m.attr("MPI_BXOR") = py::int_((int64_t)mpi4torch_op_bxor);
    m.attr("MPI_MINLOC") = py::int_((int64_t)mpi4torch_op_minloc);
    m.attr("MPI_MAXLOC") = py::int_((int64_t)mpi4torch_op_maxloc);
}

