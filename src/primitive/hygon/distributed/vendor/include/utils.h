#ifndef FTRAIN_HYGON_UTILS_H
#define FTRAIN_HYGON_UTILS_H

#include "hygon_vendor.h"
#include <c10/core/ScalarType.h>
#include <string>

namespace bytedance
{
namespace hygon_vendor
{

inline bool is_fp8_torch_dtype(at::ScalarType torch_dtype)
{
    return (torch_dtype == at::ScalarType::Float8_e4m3fn) ||
           (torch_dtype == at::ScalarType::Float8_e5m2);
}

inline bool is_s8_torch_dtype(at::ScalarType torch_dtype)
{
    return torch_dtype == at::ScalarType::Char;
}

inline std::string get_device_name()
{
    hipDeviceProp_t props{};
    int device;
    auto status = hipGetDevice(&device);
    if (status != hipSuccess)
    {
        return std::string();
    }

    status = hipGetDeviceProperties(&props, device);
    if (status != hipSuccess)
    {
        return std::string();
    }
    const std::string raw_name(props.gcnArchName);
    return raw_name.substr(
        0,
        raw_name.find(':')); // str.substr(0, npos) returns str.
}

inline bool arch_check()
{
    std::string arch_flag = get_device_name();
    if("gfx938" == arch_flag || "gfx936" == arch_flag){
        return true;
    }else{
        return false;
    }
}

inline bool get_bool_from_env(const char *env, bool default_value)
{
    auto *env_str = detail::get_vendor_env(env);
    if (env_str == nullptr)
    {
#if defined(FTRAIN_HYGON_DEBUG)
        fprintf(stderr,
                "env [%s] not set, using default_value: %d\n",
                env,
                default_value);
#endif
        return default_value;
    }
    bool value = std::string(env_str) == "1" || std::string(env_str) == "ON";
#if defined(FTRAIN_HYGON_DEBUG)
    fprintf(stderr, "env [%s] set to %d\n", env, value);
#endif
    return value;
}

inline std::string get_hygon_asm_dir()
{
    std::string hygon_asm_dir = "";
    std::string arch_flag = get_device_name();
    if ("gfx938" == arch_flag)
    {
        const char *env = detail::get_vendor_env("FTRAIN_HYGON_ASM_DIR_GFX938");
        FTRAIN_HYGON_CHECK(env != nullptr) << "FTRAIN_HYGON_ASM_DIR_GFX938 not set!";
        hygon_asm_dir = env;
    }
    else if ("gfx936" == arch_flag)
    {
        const char *env = detail::get_vendor_env("FTRAIN_HYGON_ASM_DIR_GFX936");
        FTRAIN_HYGON_CHECK(env != nullptr) << "FTRAIN_HYGON_ASM_DIR_GFX936 not set!";
        hygon_asm_dir = env;
    }
    else
    {
        FTRAIN_HYGON_LOG_ERROR("unsupport arch [%s].", arch_flag.c_str());
    }
    return hygon_asm_dir;
}

} // namespace hygon_vendor
} // namespace bytedance

#endif // FTRAIN_HYGON_UTILS_H
