#ifndef FTRAIN_H
#define FTRAIN_H

#include "moe_core/runtime.h"
#include <iostream>
#include <cstring>

namespace ftrain
{
namespace moe
{
#define MAX_WORLD_SIZE 32
#define MAX_LOCAL_WORLD_SIZE 8
#define MAX_NUM_SIGNAL 64

#ifndef FTRAIN_LIKELY
#define FTRAIN_LIKELY(x) (__builtin_expect(!!(x), 1))
#endif

#ifndef FTRAIN_UNLIKELY
#define FTRAIN_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

#define __FILENAME__                                                           \
    (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define FTRAIN_LOG_INFO(fmt, ...)                                                \
    printf("INFO: [%s:%d] " fmt "\n", __FILENAME__, __LINE__, ##__VA_ARGS__)
#define FTRAIN_LOG_ERROR(fmt, ...)                                               \
    printf("ERROR: [%s:%d] " fmt "\n", __FILENAME__, __LINE__, ##__VA_ARGS__)

namespace detail
{
class CheckFail
{
public:
    CheckFail() = default;

    // Overload the stream insertion operator.
    template <typename T>
    CheckFail &operator<<(const T &value)
    {
        message_ << value;
        return *this;
    }

    // Destructor that throws an exception with the accumulated error message.
    ~CheckFail() noexcept(false)
    {
        std::cerr << message_.str() << std::endl;
        throw std::runtime_error(message_.str());
    }

private:
    std::ostringstream message_;
};
} // namespace detail

#define FTRAIN_CHECK(condition)                                                  \
    if (!(condition))                                                          \
    ::ftrain::moe::detail::CheckFail()                                     \
        << __FILE__ << ":" << __LINE__ << " Check failed: " #condition ". "

#define CUDA_CHECK(status)                                                     \
    do                                                                         \
    {                                                                          \
        hipError_t error = status;                                             \
        FTRAIN_CHECK(error == hipSuccess)                                        \
            << "Got bad cuda status: " << hipGetErrorString(error) << "("      \
            << error << ") at: " << #status << "\n";                           \
    } while (0)

#define FTRAIN_CHECK_BINOP(lhs, rhs, op)                                         \
    if (auto x = (lhs), y = (decltype(x))(rhs); FTRAIN_UNLIKELY(!(x op y)))      \
    ::ftrain::moe::detail::CheckFail()                                     \
        << __FILE__ << ":" << __LINE__ << " Check failed: " << x               \
        << "(" #lhs ") " #op " " << y << "(" #rhs ")"

#define FTRAIN_CHECK_EQ(lhs, rhs) FTRAIN_CHECK_BINOP((lhs), (rhs), ==)
#define FTRAIN_CHECK_NE(lhs, rhs) FTRAIN_CHECK_BINOP((lhs), (rhs), !=)
#define FTRAIN_CHECK_LT(lhs, rhs) FTRAIN_CHECK_BINOP((lhs), (rhs), <)
#define FTRAIN_CHECK_GT(lhs, rhs) FTRAIN_CHECK_BINOP((lhs), (rhs), >)
#define FTRAIN_CHECK_LE(lhs, rhs) FTRAIN_CHECK_BINOP((lhs), (rhs), <=)
#define FTRAIN_CHECK_GE(lhs, rhs) FTRAIN_CHECK_BINOP((lhs), (rhs), >=)
#define FTRAIN_CHECK_DIV(lhs, rhs)                                               \
    if (auto x = (lhs), y = (decltype(x))(rhs); FTRAIN_UNLIKELY(x % y != 0))     \
    ::ftrain::moe::detail::CheckFail()                                     \
        << __FILE__ << ":" << __LINE__ << " Check failed: " << x               \
        << "(" #lhs ") % " << y << "(" #rhs ") != 0"


#define CHECK_TYPE(x, st)                                                      \
    FTRAIN_REQUIRE(x.scalar_type() == st,                                         \
                "Inconsistency of Tensor type: " #x,                           \
                " expect ",                                                    \
                st,                                                            \
                " got ",                                                       \
                x.scalar_type())
#define CHECK_CPU(x) FTRAIN_REQUIRE(x.is_cpu(), #x, " must be a cpu tensor")
#define CHECK_CUDA(x) FTRAIN_REQUIRE(x.is_cuda(), #x, " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    FTRAIN_REQUIRE(x.is_contiguous(), #x, " must be contiguous")
#define CHECK_INPUT(x, st)                                                     \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x);                                                       \
    CHECK_TYPE(x, st)
#define CHECK_INPUT_LOOSE(x)                                                   \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)
#define CHECK_NDIM(x, ndim)                                                    \
    FTRAIN_REQUIRE(x.dim() == ndim, #x, ".dim()", " != ", ndim)
#define CHECK_DIM(x, dim, sz)                                                  \
    FTRAIN_REQUIRE(x.size(dim) == sz,                                             \
                #x,                                                            \
                ".size(",                                                      \
                dim,                                                           \
                "):",                                                          \
                x.size(dim),                                                   \
                " != ",                                                        \
                sz)
#define CHECK_1D(x, dim0)                                                      \
    CHECK_NDIM(x, 1);                                                          \
    CHECK_DIM(x, 0, dim0)
#define CHECK_2D(x, dim0, dim1)                                                \
    CHECK_NDIM(x, 2);                                                          \
    CHECK_DIM(x, 0, dim0);                                                     \
    CHECK_DIM(x, 1, dim1)
#define CHECK_3D(x, dim0, dim1, dim2)                                          \
    CHECK_NDIM(x, 3);                                                          \
    CHECK_DIM(x, 0, dim0);                                                     \
    CHECK_DIM(x, 1, dim1);                                                     \
    CHECK_DIM(x, 2, dim2)
#define CHECK_DIV(x, y) FTRAIN_REQUIRE(x % y == 0, #x, " % ", #y, " != 0")

enum class DataTypeEnum : int8_t
{
    Void,
    FP16,
    BF16,
    FP32,
    E4M3,
    E5M2
};

} // namespace moe
} // namespace ftrain

#endif // FTRAIN_H
