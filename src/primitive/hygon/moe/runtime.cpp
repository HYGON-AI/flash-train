#include "moe_core/runtime.h"
#include "hip_kernels/helper_kernels.h"
#include <algorithm>
#include <cstring>
#include <limits>

namespace ftrain::moe
{
namespace
{

void check_hip(hipError_t status)
{
    if (status != hipSuccess)
        throw std::runtime_error(hipGetErrorString(status));
}

int64_t checked_numel(const std::vector<int64_t> &shape)
{
    int64_t elements = 1;
    for (int64_t extent : shape)
    {
        require(extent >= 0, "Negative tensor extent");
        require(extent == 0 || elements <= std::numeric_limits<int64_t>::max() / extent,
                "Tensor element count overflow");
        elements *= extent;
    }
    return elements;
}

struct DeviceScope
{
    int previous = 0;
    explicit DeviceScope(int device)
    {
        check_hip(hipGetDevice(&previous));
        if (previous != device)
            check_hip(hipSetDevice(device));
    }
    ~DeviceScope() { hipSetDevice(previous); }
};

struct IpcAllocation
{
    int device = 0;
    void *local = nullptr;
    std::vector<void *> peers;
    ~IpcAllocation()
    {
        int previous = 0;
        hipGetDevice(&previous);
        hipSetDevice(device);
        for (void *peer : peers)
            hipIpcCloseMemHandle(peer);
        if (local != nullptr)
            hipFree(local);
        hipSetDevice(previous);
    }
};

}

size_t element_size(ScalarType dtype)
{
    switch (dtype)
    {
    case ScalarType::Byte: return 1;
    case ScalarType::Half:
    case ScalarType::BFloat16: return 2;
    case ScalarType::Int:
    case ScalarType::Float: return 4;
    case ScalarType::Long: return 8;
    default: throw std::invalid_argument("Undefined tensor dtype");
    }
}

std::ostream &types::operator<<(std::ostream &output, ScalarType dtype)
{
    return output << static_cast<int>(dtype);
}

namespace core
{

TensorOptions TensorOptions::dtype(ScalarType value) const
{
    auto result = *this;
    result.scalar_type = value;
    return result;
}

TensorOptions TensorOptions::device_index(int value) const
{
    auto result = *this;
    result.device_id = value;
    return result;
}

hipStream_t TensorOptions::stream() const
{
    return current_stream ? current_stream() : nullptr;
}

Tensor Tensor::from_blob(void *data, std::vector<int64_t> shape,
                         TensorOptions options, std::shared_ptr<void> owner,
                         const void *identity, uint64_t version,
                         bool requires_grad)
{
    require(data != nullptr || checked_numel(shape) == 0, "Null tensor data");
    checked_numel(shape);
    const auto item_size = ftrain::moe::element_size(options.scalar_type);
    require(static_cast<uint64_t>(checked_numel(shape)) <=
                std::numeric_limits<size_t>::max() / item_size, "Tensor byte count overflow");
    Tensor result;
    result.data_ = data;
    result.shape_ = std::move(shape);
    result.options_ = std::move(options);
    result.owner_ = std::move(owner);
    result.identity_ = identity != nullptr ? identity : result.owner_.get();
    result.version_ = version;
    result.requires_grad_ = requires_grad;
    return result;
}

bool Tensor::defined() const { return options_.scalar_type != ScalarType::Undefined; }
bool Tensor::is_cuda() const { return defined(); }
bool Tensor::is_contiguous() const { return defined(); }
bool Tensor::requires_grad() const { return requires_grad_; }
int Tensor::get_device() const { return options_.device_id; }
int64_t Tensor::dim() const { return static_cast<int64_t>(shape_.size()); }
int64_t Tensor::size(int64_t dimension) const
{
    require(dimension >= 0 && dimension < dim(), "Invalid tensor dimension");
    return shape_[dimension];
}
int64_t Tensor::numel() const { return defined() ? checked_numel(shape_) : 0; }
size_t Tensor::element_size() const { return ftrain::moe::element_size(scalar_type()); }
ScalarType Tensor::scalar_type() const { return options_.scalar_type; }
const std::vector<int64_t> &Tensor::shape() const { return shape_; }
TensorOptions Tensor::options() const { return options_; }
uint64_t Tensor::version() const { return version_; }
void Tensor::set_version(uint64_t version) { version_ = version; }
void *Tensor::data_ptr() const { require(defined(), "Undefined tensor"); return data_; }
bool Tensor::is_same(const Tensor &other) const
{
    return defined() && other.defined() && identity_ == other.identity_ &&
           data_ == other.data_ && shape_ == other.shape_ &&
           scalar_type() == other.scalar_type() && get_device() == other.get_device();
}

std::string Tensor::sizes() const
{
    std::ostringstream output;
    output << '[';
    for (size_t dimension = 0; dimension < shape_.size(); ++dimension)
        output << (dimension == 0 ? "" : ", ") << shape_[dimension];
    output << ']';
    return output.str();
}

Tensor Tensor::narrow(int64_t dimension, int64_t start, int64_t length) const
{
    require(dimension == 0, "Only contiguous leading-dimension views are supported");
    require(start >= 0 && length >= 0 && start <= size(0) && length <= size(0) - start,
            "Tensor view exceeds capacity");
    Tensor result = *this;
    int64_t row_elements = 1;
    for (size_t index = 1; index < shape_.size(); ++index)
        row_elements *= shape_[index];
    if (start != 0)
        result.data_ = static_cast<unsigned char *>(data_) + start * row_elements * element_size();
    result.shape_[0] = length;
    return result;
}

Tensor Tensor::slice(int64_t dimension, int64_t start, int64_t end) const
{
    return narrow(dimension, start, std::min(end, size(dimension)) - start);
}

Tensor Tensor::clone() const
{
    Tensor result = empty(shape_, options_);
    if (numel() != 0)
        check_hip(hipMemcpyAsync(result.data_ptr(), data_ptr(), numel() * element_size(),
                                 hipMemcpyDeviceToDevice, options_.stream()));
    return result;
}

void Tensor::zero_() const
{
    if (numel() != 0)
        check_hip(hipMemsetAsync(data_ptr(), 0, numel() * element_size(), options_.stream()));
}

Tensor empty(const std::vector<int64_t> &shape, TensorOptions options)
{
    const auto elements = checked_numel(shape);
    const auto item_size = element_size(options.scalar_type);
    require(static_cast<uint64_t>(elements) <= std::numeric_limits<size_t>::max() / item_size,
            "Tensor byte count overflow");
    Allocation allocation;
    if (options.allocator)
        allocation = options.allocator(shape, options.scalar_type, options.device_id);
    else
    {
        DeviceScope guard(options.device_id);
        check_hip(hipMalloc(&allocation.data, std::max<size_t>(1, elements * item_size)));
        allocation.owner = std::shared_ptr<void>(allocation.data, [device = options.device_id](void *data) {
            int previous = 0;
            hipGetDevice(&previous);
            hipSetDevice(device);
            hipFree(data);
            hipSetDevice(previous);
        });
    }
    require(allocation.data != nullptr || elements == 0, "Allocator returned null");
    return Tensor::from_blob(allocation.data, shape, std::move(options), allocation.owner);
}

Tensor zeros(const std::vector<int64_t> &shape, TensorOptions options)
{
    auto result = empty(shape, std::move(options));
    result.zero_();
    return result;
}

Tensor ones(const std::vector<int64_t> &shape, TensorOptions options)
{
    require(options.scalar_type == ScalarType::Int, "ones requires int32");
    auto result = empty(shape, std::move(options));
    std::vector<int32_t> values(result.numel(), 1);
    if (!values.empty())
        check_hip(hipMemcpy(result.data_ptr(), values.data(), values.size() * sizeof(int32_t),
                            hipMemcpyHostToDevice));
    return result;
}

Context::Context(int rank, int world_size, int device, AllGather allgather, Allocator allocator)
    : rank_(rank), world_size_(world_size), device_(device),
      allgather_(std::move(allgather)), allocator_(std::move(allocator))
{
    require(world_size > 0 && world_size <= 8 && rank >= 0 && rank < world_size && device >= 0,
            "Invalid single-node rank configuration");
    require(world_size == 1 || static_cast<bool>(allgather_), "Missing bootstrap allgather");
}

int Context::rank() const { return rank_; }
int Context::world_size() const { return world_size_; }
int Context::device() const { return device_; }
hipStream_t Context::stream() const { return stream_; }
void Context::set_stream(hipStream_t stream) { stream_ = stream; }

TensorOptions Context::tensor_options()
{
    TensorOptions options;
    options.device_id = device_;
    options.allocator = allocator_;
    options.current_stream = [weak = weak_from_this()] {
        auto context = weak.lock();
        require(context != nullptr, "Tensor stream context has expired");
        return context->stream();
    };
    return options;
}

void Context::allgather(const void *send, void *receive, size_t bytes) const
{
    if (world_size_ == 1)
        std::memcpy(receive, send, bytes);
    else
        allgather_(send, receive, bytes);
}

void Context::barrier() const
{
    unsigned char value = 0;
    std::vector<unsigned char> values(world_size_);
    allgather(&value, values.data(), 1);
}

std::vector<Tensor> create_tensor_list(const std::vector<int64_t> &shape, ScalarType dtype,
                                      const std::shared_ptr<Context> &context, bool uncached)
{
    const auto elements = checked_numel(shape);
    require(elements > 0 && static_cast<uint64_t>(elements) <=
                std::numeric_limits<size_t>::max() / element_size(dtype),
            "Invalid IPC allocation size");
    DeviceScope guard(context->device());
    auto allocation = std::make_shared<IpcAllocation>();
    allocation->device = context->device();
    const size_t bytes = elements * element_size(dtype);
    if (uncached)
        check_hip(hipExtMallocWithFlags(&allocation->local, bytes, hipDeviceMallocUncached));
    else
        check_hip(hipMalloc(&allocation->local, bytes));
    check_hip(hipMemset(allocation->local, 0, bytes));
    hipIpcMemHandle_t handle;
    check_hip(hipIpcGetMemHandle(&handle, allocation->local));
    std::vector<hipIpcMemHandle_t> handles(context->world_size());
    context->allgather(&handle, handles.data(), sizeof(handle));
    std::vector<Tensor> tensors;
    for (int rank = 0; rank < context->world_size(); ++rank)
    {
        void *pointer = allocation->local;
        if (rank != context->rank())
        {
            check_hip(hipIpcOpenMemHandle(&pointer, handles[rank], hipIpcMemLazyEnablePeerAccess));
            allocation->peers.push_back(pointer);
        }
        tensors.push_back(Tensor::from_blob(pointer, shape, context->tensor_options().dtype(dtype), allocation));
    }
    context->barrier();
    return tensors;
}

void barrier_all_on_stream(hipStream_t stream, std::optional<std::vector<Tensor>> buffers,
                           std::optional<int> rank)
{
    require(buffers.has_value() && rank.has_value(), "Missing IPC barrier context");
    require(!buffers->empty() && buffers->size() <= 8 && *rank >= 0 &&
                *rank < static_cast<int>(buffers->size()), "Invalid IPC barrier context");
    std::vector<int32_t *> pointers;
    for (const auto &buffer : *buffers)
        pointers.push_back(buffer.data_ptr<int32_t>());
    cudaipc_barrier_all_on_stream_impl(stream, pointers.data(), *rank, pointers.size());
}

}
}
