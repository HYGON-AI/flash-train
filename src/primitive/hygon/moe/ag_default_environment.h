#pragma once

#include <cstdlib>
#include <string>
#include <vector>

namespace ftrain::moe::core
{
class ScopedAGDefaultEnvironment
{
public:
    explicit ScopedAGDefaultEnvironment(bool enabled)
    {
        if (!enabled)
            return;
        set_default("FTRAIN_AG_GROUP_GEMM_ASM_MT256", "1");
        const std::string mt256 = std::getenv("FTRAIN_AG_GROUP_GEMM_ASM_MT256");
        if (mt256 != "1" && mt256 != "ON")
            return;
        set_default("FTRAIN_AG_GROUP_GEMM_RANK_TILE_ENABLE", "1");
        set_default("FTRAIN_AG_GROUP_GEMM_ASM_MT256_WGM8", "1");
        set_default("FTRAIN_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT", "0");
        set_default("FTRAIN_AG_GROUP_GEMM_ASM_MT256_VARIANT", "kn_direct");
        set_default("FTRAIN_AG_GROUP_GEMM_ASM_MT256_NATIVE_CONTRACT", "1");
        set_default("FTRAIN_AG_GROUP_GEMM_ASM_MT256_DIRECT_INPUT", "1");
    }

    ~ScopedAGDefaultEnvironment()
    {
        for (auto iterator = added_.rbegin(); iterator != added_.rend(); ++iterator)
            unsetenv(iterator->c_str());
    }

    ScopedAGDefaultEnvironment(const ScopedAGDefaultEnvironment &) = delete;
    ScopedAGDefaultEnvironment &operator=(const ScopedAGDefaultEnvironment &) = delete;

private:
    void set_default(const char *name, const char *value)
    {
        if (std::getenv(name) == nullptr)
        {
            added_.emplace_back(name);
            setenv(name, value, 1);
        }
    }

    std::vector<std::string> added_;
};
}
