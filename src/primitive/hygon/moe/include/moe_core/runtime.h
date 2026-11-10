#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ftrain::moe
{

namespace types
{
enum class ScalarType
{
    Undefined,
    Byte,
    Int,
    Long,
    Half,
    BFloat16,
    Float,
};

std::ostream &operator<<(std::ostream &output, ScalarType dtype);
}

using ScalarType = types::ScalarType;
size_t element_size(ScalarType dtype);

template <typename... Arguments>
void require(bool condition, const Arguments &...arguments)
{
    if (!condition)
    {
        std::ostringstream message;
        (message << ... << arguments);
        throw std::runtime_error(message.str());
    }
}

namespace core
{

struct Allocation
{
    void *data = nullptr;
    std::shared_ptr<void> owner;
};

using Allocator = std::function<Allocation(
    const std::vector<int64_t> &, ScalarType, int)>;

struct TensorOptions
{
    ScalarType scalar_type = ScalarType::Undefined;
    int device_id = 0;
    Allocator allocator;
    std::function<hipStream_t()> current_stream;

    TensorOptions dtype(ScalarType value) const;
    TensorOptions device_index(int value) const;
    hipStream_t stream() const;
};

class Tensor
{
public:
    Tensor() = default;
    static Tensor from_blob(void *data,
                            std::vector<int64_t> shape,
                            TensorOptions options,
                            std::shared_ptr<void> owner = {},
                            const void *identity = nullptr,
                            uint64_t version = 0,
                            bool requires_grad = false);
    bool defined() const;
    bool is_cuda() const;
    bool is_contiguous() const;
    bool requires_grad() const;
    int get_device() const;
    int64_t dim() const;
    int64_t size(int64_t dimension) const;
    int64_t numel() const;
    size_t element_size() const;
    ScalarType scalar_type() const;
    std::string sizes() const;
    const std::vector<int64_t> &shape() const;
    TensorOptions options() const;
    bool is_same(const Tensor &other) const;
    uint64_t version() const;
    void set_version(uint64_t version);
    void *data_ptr() const;
    template <typename Value> Value *data_ptr() const
    {
        return static_cast<Value *>(data_ptr());
    }
    Tensor narrow(int64_t dimension, int64_t start, int64_t length) const;
    Tensor slice(int64_t dimension, int64_t start, int64_t end) const;
    Tensor clone() const;
    void zero_() const;

private:
    void *data_ = nullptr;
    std::vector<int64_t> shape_;
    TensorOptions options_;
    std::shared_ptr<void> owner_;
    const void *identity_ = nullptr;
    uint64_t version_ = 0;
    bool requires_grad_ = false;
};

Tensor empty(const std::vector<int64_t> &shape, TensorOptions options);
Tensor zeros(const std::vector<int64_t> &shape, TensorOptions options);
Tensor ones(const std::vector<int64_t> &shape, TensorOptions options);

class Context : public std::enable_shared_from_this<Context>
{
public:
    using AllGather = std::function<void(const void *, void *, size_t)>;
    Context(int rank, int world_size, int device, AllGather allgather,
            Allocator allocator = {});
    int rank() const;
    int world_size() const;
    int device() const;
    hipStream_t stream() const;
    void set_stream(hipStream_t stream);
    TensorOptions tensor_options();
    void allgather(const void *send, void *receive, size_t bytes) const;
    void barrier() const;

private:
    int rank_;
    int world_size_;
    int device_;
    AllGather allgather_;
    Allocator allocator_;
    hipStream_t stream_ = nullptr;
};

std::vector<Tensor> create_tensor_list(
    const std::vector<int64_t> &shape, ScalarType dtype,
    const std::shared_ptr<Context> &context, bool uncached = false);
void barrier_all_on_stream(hipStream_t stream,
                           std::optional<std::vector<Tensor>> buffers,
                           std::optional<int> rank);

}
}

#define FTRAIN_REQUIRE(...) ::ftrain::moe::require(__VA_ARGS__)
