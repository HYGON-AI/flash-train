#pragma once

#include <cerrno>
#include <cstdlib>
#include <string>
#include <system_error>
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
        try
        {
            const std::string mt256 = set_default("FTRAIN_AG_GROUP_GEMM_ASM_MT256", "1");
            if (mt256 != "1" && mt256 != "ON")
                return;
            set_default("FTRAIN_AG_GROUP_GEMM_RANK_TILE_ENABLE", "1");
            set_default("FTRAIN_AG_GROUP_GEMM_ASM_MT256_WGM8", "1");
            set_default("FTRAIN_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT", "0");
            set_default("FTRAIN_AG_GROUP_GEMM_ASM_MT256_VARIANT", "kn_direct");
            set_default("FTRAIN_AG_GROUP_GEMM_ASM_MT256_NATIVE_CONTRACT", "1");
            set_default("FTRAIN_AG_GROUP_GEMM_ASM_MT256_DIRECT_INPUT", "1");
        }
        catch (...)
        {
            restore();
            throw;
        }
    }

    ~ScopedAGDefaultEnvironment()
    {
        restore();
    }

    ScopedAGDefaultEnvironment(const ScopedAGDefaultEnvironment &) = delete;
    ScopedAGDefaultEnvironment &operator=(const ScopedAGDefaultEnvironment &) = delete;

private:
    const char *set_default(const char *name, const char *value)
    {
        if (const char *existing = std::getenv(name))
            return existing;
        added_.emplace_back(name);
        if (setenv(name, value, 1) != 0)
        {
            const int error = errno;
            added_.pop_back();
            throw std::system_error(error, std::generic_category(),
                                    std::string("setenv(") + name + ")");
        }
        return value;
    }

    void restore() noexcept
    {
        for (auto iterator = added_.rbegin(); iterator != added_.rend(); ++iterator)
            unsetenv(iterator->c_str());
        added_.clear();
    }

    std::vector<std::string> added_;
};
}
