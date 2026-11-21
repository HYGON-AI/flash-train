#ifndef FTRAIN_BENCH_OPS_MOE_CONTEXT_HPP_
#define FTRAIN_BENCH_OPS_MOE_CONTEXT_HPP_

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <flash_train/flash_train.h>

namespace ftrain_bench {

inline int environmentInt(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr) { return fallback; }
    char* end         = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (*value == '\0' || *end != '\0' || parsed < 0 || parsed > INT32_MAX) {
        throw std::runtime_error(std::string("invalid benchmark environment: ") + name);
    }
    return static_cast<int>(parsed);
}

struct MoeContext final {
    std::filesystem::path directory;
    int rank     = environmentInt("FTRAIN_BENCH_RANK", 0);
    int world    = environmentInt("FTRAIN_BENCH_WORLD_SIZE", 0);
    int sequence = 0;

    MoeContext() {
        const char* path = std::getenv("FTRAIN_BENCH_RENDEZVOUS");
        if (path == nullptr || world < 2 || world > 8 || rank >= world) {
            throw std::runtime_error("MoE C harness must be launched through ftrain_bench cbench");
        }
        directory = path;
    }

    void gather(const void* send, void* receive, uint64_t bytes) {
        const auto prefix  = std::to_string(sequence++) + "-";
        const auto target  = directory / (prefix + std::to_string(rank));
        const auto pending = target.string() + ".pending";
        {
            std::ofstream output(pending, std::ios::binary);
            output.write(static_cast<const char*>(send), bytes);
            if (!output.good()) { throw std::runtime_error("benchmark bootstrap write failed"); }
        }
        std::filesystem::rename(pending, target);
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(environmentInt("FTRAIN_BENCH_TIMEOUT", 600));
        for (int peer = 0; peer < world; ++peer) {
            const auto path = directory / (prefix + std::to_string(peer));
            while (!std::filesystem::exists(path)) {
                if (std::chrono::steady_clock::now() > deadline) {
                    throw std::runtime_error("benchmark bootstrap timeout");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::ifstream input(path, std::ios::binary);
            input.read(static_cast<char*>(receive) + peer * bytes, bytes);
            if (!input.good() || input.peek() != std::ifstream::traits_type::eof()) {
                throw std::runtime_error("benchmark bootstrap byte count mismatch");
            }
        }
    }

    void barrier() {
        int value = 1;
        std::vector<int> values(world);
        gather(&value, values.data(), sizeof(value));
        return;
    }

    static FTrainStatus callback(void* user, const void* send, void* receive, uint64_t bytes) noexcept {
        try {
            static_cast<MoeContext*>(user)->gather(send, receive, bytes);
            return FTRAIN_STATUS_SUCCESS;
        }
        catch (...) {
            return FTRAIN_STATUS_INTERNAL_ERROR;
        }
    }
};

}  // namespace ftrain_bench

#endif
