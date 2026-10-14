#ifndef FTRAIN_HYGON_VENDOR_H_
#define FTRAIN_HYGON_VENDOR_H_

#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace bytedance
{
namespace hygon_vendor
{
#define MAX_WORLD_SIZE 32
#define MAX_LOCAL_WORLD_SIZE 8
#define MAX_NUM_SIGNAL 64

#ifndef FTRAIN_HYGON_LIKELY
#define FTRAIN_HYGON_LIKELY(x) (__builtin_expect(!!(x), 1))
#endif

#ifndef FTRAIN_HYGON_UNLIKELY
#define FTRAIN_HYGON_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

#define __FILENAME__                                                           \
    (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define FTRAIN_HYGON_LOG_INFO(fmt, ...)                                                \
    printf("INFO: [%s:%d] " fmt "\n", __FILENAME__, __LINE__, ##__VA_ARGS__)
#define FTRAIN_HYGON_LOG_ERROR(fmt, ...)                                               \
    printf("ERROR: [%s:%d] " fmt "\n", __FILENAME__, __LINE__, ##__VA_ARGS__)

namespace detail
{
inline const char *get_vendor_env(const char *name)
{
    if (name == nullptr)
    {
        return nullptr;
    }
    if (const char *primary_value = std::getenv(name);
        primary_value != nullptr)
    {
        return primary_value;
    }
    constexpr const char *kPrimaryPrefix = "FTRAIN_HYGON_";
    constexpr std::size_t kPrimaryPrefixSize = 13;
    if (std::strncmp(name, kPrimaryPrefix, kPrimaryPrefixSize) != 0)
    {
        return nullptr;
    }
    const std::string legacyName =
        "FL" "UX_" + std::string(name + kPrimaryPrefixSize);
    return std::getenv(legacyName.c_str());
}

inline bool get_vendor_bool_env(const char *name, bool default_value)
{
    // Keep the historical variable authoritative while both spellings are
    // supported.  This preserves old A/B scripts and route-scoped overrides.
    const char *value = get_vendor_env(name);
    if (value == nullptr)
    {
        return default_value;
    }
    const std::string parsed(value);
    return parsed == "1" || parsed == "ON";
}

inline std::string legacy_vendor_env_name(const char *name)
{
    if (name == nullptr)
    {
        return {};
    }
    constexpr const char *kPrimaryPrefix = "FTRAIN_HYGON_";
    constexpr std::size_t kPrimaryPrefixSize = 13;
    if (std::strncmp(name, kPrimaryPrefix, kPrimaryPrefixSize) != 0)
    {
        return name;
    }
    return "FL" "UX_" + std::string(name + kPrimaryPrefixSize);
}

inline std::string legacy_kernel_function_name(const std::string &name)
{
    constexpr const char *kPrimaryPrefix = "hygon_";
    constexpr std::size_t kPrimaryPrefixSize = 6;
    if (name.rfind(kPrimaryPrefix, 0) != 0)
    {
        return name;
    }
    return "fl" "ux_" + name.substr(kPrimaryPrefixSize);
}

class ScopedVendorEnvironment
{
  public:
    ScopedVendorEnvironment() = default;
    ScopedVendorEnvironment(const ScopedVendorEnvironment &) = delete;
    ScopedVendorEnvironment &operator=(const ScopedVendorEnvironment &) = delete;

    ~ScopedVendorEnvironment()
    {
        for (auto it = saved_.rbegin(); it != saved_.rend(); ++it)
        {
            if (it->present)
            {
                setenv(it->name, it->value.c_str(), 1);
            }
            else
            {
                unsetenv(it->name);
            }
        }
    }

    void set(const char *name, const char *value)
    {
        // Route-owned values use the public Hygon name.  get_vendor_env gives
        // this active value precedence, so writing the legacy alias too would
        // only add host-side setenv overhead.
        save_and_set(name, value);
    }

    void set_if_absent(const char *name, const char *value)
    {
        if (get_vendor_env(name) == nullptr)
        {
            set(name, value);
        }
    }

    void unset(const char *name)
    {
        save_and_unset(name);
        const std::string legacy_name = legacy_vendor_env_name(name);
        if (legacy_name != name)
        {
            dynamic_names_.emplace_back(legacy_name);
            save_and_unset(dynamic_names_.back().c_str());
        }
    }

    void reserve(std::size_t count)
    {
        saved_.reserve(count);
    }

  private:
    struct SavedValue
    {
        const char *name = nullptr;
        bool present = false;
        std::string value;
    };

    void save_and_set(const char *name, const char *value)
    {
        const char *old_value = std::getenv(name);
        saved_.emplace_back(SavedValue{name, old_value != nullptr});
        if (old_value != nullptr)
        {
            saved_.back().value = old_value;
        }
        setenv(name, value, 1);
    }

    void save_and_unset(const char *name)
    {
        const char *old_value = std::getenv(name);
        saved_.emplace_back(SavedValue{name, old_value != nullptr});
        if (old_value != nullptr)
        {
            saved_.back().value = old_value;
        }
        unsetenv(name);
    }

    std::vector<SavedValue> saved_;
    std::deque<std::string> dynamic_names_;
};

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

#define FTRAIN_HYGON_CHECK(condition)                                                  \
    if (!(condition))                                                          \
    ::bytedance::hygon_vendor::detail::CheckFail()                                     \
        << __FILE__ << ":" << __LINE__ << " Check failed: " #condition ". "

#define CUDA_CHECK(status)                                                     \
    do                                                                         \
    {                                                                          \
        hipError_t error = status;                                             \
        FTRAIN_HYGON_CHECK(error == hipSuccess)                                        \
            << "Got bad cuda status: " << hipGetErrorString(error) << "("      \
            << error << ") at: " << #status << "\n";                           \
    } while (0)

#define FTRAIN_HYGON_CHECK_BINOP(lhs, rhs, op)                                         \
    if (auto x = (lhs), y = (decltype(x))(rhs); FTRAIN_HYGON_UNLIKELY(!(x op y)))      \
    ::bytedance::hygon_vendor::detail::CheckFail()                                     \
        << __FILE__ << ":" << __LINE__ << " Check failed: " << x               \
        << "(" #lhs ") " #op " " << y << "(" #rhs ")"

#define FTRAIN_HYGON_CHECK_EQ(lhs, rhs) FTRAIN_HYGON_CHECK_BINOP((lhs), (rhs), ==)
#define FTRAIN_HYGON_CHECK_NE(lhs, rhs) FTRAIN_HYGON_CHECK_BINOP((lhs), (rhs), !=)
#define FTRAIN_HYGON_CHECK_LT(lhs, rhs) FTRAIN_HYGON_CHECK_BINOP((lhs), (rhs), <)
#define FTRAIN_HYGON_CHECK_GT(lhs, rhs) FTRAIN_HYGON_CHECK_BINOP((lhs), (rhs), >)
#define FTRAIN_HYGON_CHECK_LE(lhs, rhs) FTRAIN_HYGON_CHECK_BINOP((lhs), (rhs), <=)
#define FTRAIN_HYGON_CHECK_GE(lhs, rhs) FTRAIN_HYGON_CHECK_BINOP((lhs), (rhs), >=)
#define FTRAIN_HYGON_CHECK_DIV(lhs, rhs)                                               \
    if (auto x = (lhs), y = (decltype(x))(rhs); FTRAIN_HYGON_UNLIKELY(x % y != 0))     \
    ::bytedance::hygon_vendor::detail::CheckFail()                                     \
        << __FILE__ << ":" << __LINE__ << " Check failed: " << x               \
        << "(" #lhs ") % " << y << "(" #rhs ") != 0"


#define CHECK_TYPE(x, st)                                                      \
    TORCH_CHECK(x.scalar_type() == st,                                         \
                "Inconsistency of Tensor type: " #x,                           \
                " expect ",                                                    \
                st,                                                            \
                " got ",                                                       \
                x.scalar_type())
#define CHECK_CPU(x) TORCH_CHECK(x.is_cpu(), #x, " must be a cpu tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x, " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")
#define CHECK_INPUT(x, st)                                                     \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x);                                                       \
    CHECK_TYPE(x, st)
#define CHECK_INPUT_LOOSE(x)                                                   \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)
#define CHECK_NDIM(x, ndim)                                                    \
    TORCH_CHECK(x.dim() == ndim, #x, ".dim()", " != ", ndim)
#define CHECK_DIM(x, dim, sz)                                                  \
    TORCH_CHECK(x.size(dim) == sz,                                             \
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
#define CHECK_DIV(x, y) TORCH_CHECK(x % y == 0, #x, " % ", #y, " != 0")

enum class DataTypeEnum : int8_t
{
    Void,
    FP16,
    BF16,
    FP32,
    E4M3,
    E5M2
};

} // namespace hygon_vendor
} // namespace bytedance

#endif // FTRAIN_HYGON_VENDOR_H_
