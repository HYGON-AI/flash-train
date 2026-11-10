#pragma once
#include "moe_core.h"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <string>

namespace ftrain::moe
{
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
    auto *env_str = getenv(env);
    if (env_str == nullptr)
    {
#if defined(FTRAIN_DEBUG)
        fprintf(stderr,
                "env [%s] not set, using default_value: %d\n",
                env,
                default_value);
#endif
        return default_value;
    }
    bool value = std::string(env_str) == "1" || std::string(env_str) == "ON";
#if defined(FTRAIN_DEBUG)
    fprintf(stderr, "env [%s] set to %d\n", env, value);
#endif
    return value;
}

inline std::string get_ftrain_asm_dir()
{
    std::string ftrain_ams_dir = "";
    std::string arch_flag = get_device_name();
    if ("gfx938" == arch_flag)
    {
        const char *env = std::getenv("FTRAIN_ASM_DIR_GFX938");
        FTRAIN_CHECK(env != nullptr) << "FTRAIN_ASM_DIR_GFX938 not set!";
        ftrain_ams_dir = env;
    }
    else if ("gfx936" == arch_flag)
    {
        const char *env = std::getenv("FTRAIN_ASM_DIR_GFX936");
        FTRAIN_CHECK(env != nullptr) << "FTRAIN_ASM_DIR_GFX936 not set!";
        ftrain_ams_dir = env;
    }
    else
    {
        FTRAIN_LOG_ERROR("unsupport arch [%s].", arch_flag.c_str());
    }
    return ftrain_ams_dir;
}

}
