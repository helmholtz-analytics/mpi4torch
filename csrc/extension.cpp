
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/utils.h>
//#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/script.h>
#include <torch/extension.h>

#include <mpi.h>

#include <iostream>
#include <stdexcept>
#include <sstream>

using torch::Tensor;
using torch::ScalarType;
using torch::autograd::variable_list;

namespace
{

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

    Tensor MPIAllreduce(const Tensor& input);
    Tensor MPIBcast_(const Tensor& input, int64_t root);
    Tensor MPIReduce_(const Tensor& input, int64_t root);
    variable_list MPIIsend(const Tensor& input, int64_t dest, int64_t tag);
    variable_list MPIIrecv(const Tensor& input, int64_t source, int64_t tag);
    Tensor MPIWait(const variable_list& input);
};

c10::intrusive_ptr<MPI_Comm_Wrapper> comm_world()
{
    return c10::make_intrusive<MPI_Comm_Wrapper>(MPI_COMM_WORLD);
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

struct MPIAllreduceBackward : public MPIBackwardNode {
    variable_list apply(variable_list&& grads) override;
    std::string name() const override {
        return std::string("MPIAllreduceBackward");
    }

    void release_variables() override {
        return;
    }
};

variable_list MPIAllreduceBackward::apply (variable_list&& grads)
{
    variable_list grad_inputs(1);
    if (should_compute_output(0)) {
        grad_inputs[0] = comm.MPIAllreduce(grads[0]);
    }
    return grad_inputs;
}

Tensor MPI_Comm_Wrapper::MPIAllreduce(const Tensor& input)
{
    std::shared_ptr<MPIAllreduceBackward> grad_fn;
    if (torch::autograd::compute_requires_grad(input)) {
        grad_fn = std::shared_ptr<MPIAllreduceBackward> (new MPIAllreduceBackward(), torch::autograd::deleteNode);
        grad_fn->comm = *this;
        grad_fn->set_next_edges(torch::autograd::collect_next_edges(input));
    }
    auto result = ([&]() {
        at::AutoNonVariableTypeMode non_var_type_mode(true);

        // make input contiguous
        auto input_cont = input.contiguous();

        auto recv = torch::empty_like(input_cont);

        check_mpi_return_value(
            MPI_Allreduce(input_cont.data_ptr(), recv.data_ptr(), input_cont.numel(),
                      torch2mpitype(input_cont.scalar_type()), MPI_SUM, comm)
        );

        return recv;
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
        grad_inputs[0] = comm.MPIReduce_(grads[0],root);
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
        at::AutoNonVariableTypeMode non_var_type_mode(true);

        // 1. Make input contiguous
        // 2. Call variable_data() to make a shallow copy of the input tensor without the autograd history,
        //    such that it can be savely returned from this function.
        auto input_cont = input.contiguous().variable_data();

        check_mpi_return_value(
            MPI_Bcast(input_cont.data_ptr(), input_cont.numel(),
                  torch2mpitype(input_cont.scalar_type()),
                  static_cast<int>(root), comm)
        );

        return input_cont;
    })();
    if (grad_fn) {
        set_history(torch::autograd::flatten_tensor_args(result), grad_fn);
    }
    return result;
}

struct MPIReduceInPlaceBackward : public MPIBackwardNode {
    MPIReduceInPlaceBackward(int _root) : root(_root) {}
    variable_list apply(variable_list&& grads) override;
    std::string name() const override {
        return std::string("MPIReduceInPlaceBackward");
    }

    void release_variables() override {
        return;
    }

    int root;
};

variable_list MPIReduceInPlaceBackward::apply (variable_list&& grads)
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

Tensor MPI_Comm_Wrapper::MPIReduce_(const Tensor& input, int64_t root)
{
    // TODO: check for root being in int range
    std::shared_ptr<MPIReduceInPlaceBackward> grad_fn;
    if (torch::autograd::compute_requires_grad(input)) {
        grad_fn = std::shared_ptr<MPIReduceInPlaceBackward> (new MPIReduceInPlaceBackward(static_cast<int>(root)),
                                                      torch::autograd::deleteNode);
        grad_fn->comm = *this;
        grad_fn->set_next_edges(torch::autograd::collect_next_edges(input));
    }
    auto result = ([&]() {
        at::AutoNonVariableTypeMode non_var_type_mode(true);

        // 1. Make input contiguous
        // 2. Call variable_data() to make a shallow copy of the input tensor without the autograd history,
        //    such that it can be savely returned from this function.
        auto input_cont = input.contiguous().variable_data();

        const int rank = GetRank();

        void* sendbuf = input_cont.data_ptr();
        if (rank == root) {
            // One is only allowed to pass MPI_IN_PLACE for the root process
            // cf. https://stackoverflow.com/a/17744793
            sendbuf = MPI_IN_PLACE;
        }

        check_mpi_return_value(MPI_Reduce(sendbuf, input_cont.data_ptr(), input_cont.numel(),
                                          torch2mpitype(input_cont.scalar_type()),
                                          MPI_SUM, static_cast<int>(root), comm));

        if (rank != root) {
            // We fill the non-root results with zeros to make the function properly behaved.
            // TODO: We could potentially let the return-value be undefined and save some ops?
            input_cont.zero_();
        }

        return input_cont;
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
        at::AutoNonVariableTypeMode non_var_type_mode(true);

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
        at::AutoNonVariableTypeMode non_var_type_mode(true);

        // make input contiguous
        // we call variable_data() since we also return the input buffer to ensure it stays in scope
        auto input_cont = input.contiguous().variable_data();

        MPI_Request req;
        check_mpi_return_value(MPI_Isend(input_cont.data_ptr(), input_cont.numel(),
                  torch2mpitype(input_cont.scalar_type()), static_cast<int>(dest),
                  static_cast<int>(tag), comm, &req));

        auto ret = torch::empty({5},at::kDouble);
        auto fortran_handle = MPI_Request_c2f(req);
        ret[0] = static_cast<double>(fortran_handle);
        ret[1] = static_cast<double>(Isend_Op);
        ret[2] = static_cast<double>(dest);
        ret[3] = static_cast<double>(tag);
        ret[4] = static_cast<double>(0xFFFFFFFF & std::hash<void*>()(input_cont.data_ptr()));
        variable_list retlist;
        retlist.push_back(ret);
        retlist.push_back(input_cont); // make sure the buffer stays in scope!!!
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
        at::AutoNonVariableTypeMode non_var_type_mode(true);

        // TODO: check whether input is contiguous
        auto input_cont = input.variable_data();

        MPI_Request req;
        check_mpi_return_value(MPI_Irecv(input_cont.data_ptr(), input_cont.numel(),
                  torch2mpitype(input_cont.scalar_type()), static_cast<int>(source),
                  static_cast<int>(tag), comm, &req));

        auto ret = torch::empty({5},at::kDouble);
        auto fortran_handle = MPI_Request_c2f(req);
        ret[0] = static_cast<double>(fortran_handle);
        ret[1] = static_cast<double>(Irecv_Op);
        ret[2] = static_cast<double>(source);
        ret[3] = static_cast<double>(tag);
        ret[4] = static_cast<double>(0xFFFFFFFF & std::hash<void*>()(input_cont.data_ptr()));
        variable_list retlist;
        retlist.push_back(ret);
        retlist.push_back(input_cont); // We ensure that the buffer stays in scope and is not garbage collected
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
            oss << "torchmpi: Detected bifurcation in MPIWait handle usage. Next node in DAG"
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

    if (hashvalue != (0xFFFFFFFF & std::hash<void*>()(input[1].data_ptr()))) {
        std::ostringstream oss;
        oss << "torchmpi: Detected bifurcation in MPIWait handle usage. "
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
        at::AutoNonVariableTypeMode non_var_type_mode(true);

        MPI_Status status; // TODO: Handle use cases for MPI_Status
        check_mpi_return_value(MPI_Wait(&req, & status));
        // return a shallow copy of the second input tensor without the autograd strings attached
        return input[1].variable_data();
    })();
    if (grad_fn) {
        set_history(torch::autograd::flatten_tensor_args(result), grad_fn);
    }
    return result;
}

}

// New-style API is more similar to pybind11, and will in the future be:
static auto mpi_comm_wrapper_registry = torch::class_<::MPI_Comm_Wrapper>("torchmpi", "MPI_Comm_Wrapper")
    .def("GetRank", &MPI_Comm_Wrapper::GetRank)
    .def("GetSize", &MPI_Comm_Wrapper::GetSize)
    .def("Allreduce", &MPI_Comm_Wrapper::MPIAllreduce)
    .def("Bcast_", &MPI_Comm_Wrapper::MPIBcast_)
    .def("Reduce_", &MPI_Comm_Wrapper::MPIReduce_)
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
    .op("torchmpi::COMM_WORLD", &comm_world)
    .op("torchmpi::JoinDummies", &JoinDummies);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // There are some issues with older versions of OpenMPI which have difficulties with dlopen-ing these
    // libraries the way it is done within pythons extension system (i.e. with RTLD_LOCAL). This leads to the fact
    // that loading this extension leaves some relocations dangling (?), which then
    // causes problems down the road once MPI_* functions are actually called.
    // mpi4py has fixes for this, so we just circumvent this issue by importing mpi4py first.
    // We could force the user to import mpi4py manually in his/her python application,
    // but it is of course more convenient to do it here.
    // Despite this we do not have any use for mpi4py in this extension.
    //
    // [1] https://github.com/open-mpi/ompi/issues/3705
    m.import("mpi4py.MPI");


    m.def("comm_world", []() { return MPI_Comm_Wrapper(MPI_COMM_WORLD); }, "Get MPI_COMM_WORLD");
    //m.def("GetRank", &GetRank, "Return current rank");
    //m.def("GetSize", &GetSize, "Return world communicator size");

    //m.def("MPIAllreduce", &MPIAllreduce, "AD-able variant of MPIAllreduce");
}

