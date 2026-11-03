// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_H_POST_BDA_BWD_FINDER_HPP_
#define FTRAIN_FAMILY_H_POST_BDA_BWD_FINDER_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "flash_train/family/h_post_bda_bwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct HPostBdaBwdRegistrationOrderFinder final {
    static const char* getName() { return "HPostBdaBwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const HPostBdaBwdProblem& problem, const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16HPostBdaBwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16HPostBdaBwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP32: base = "Fp32HPostBdaBwd"; break;
            default: return {};
        }
        const auto preferred =
            (problem.getComputeType() == FTRAIN_NUMERIC_TYPE_FP32 && problem.getStreams() == 4 &&
             problem.getChannels() >= 4096 && problem.getBatchStride() >= 1024)
                ? std::string("Tiled4")
            : ((problem.getStreams() == 2 || problem.getStreams() == 4) && problem.getChannels() >= 512 &&
               problem.getChannels() % (problem.getComputeType() == FTRAIN_NUMERIC_TYPE_FP32 ? 256 : 512) == 0 &&
               ((reinterpret_cast<std::uintptr_t>(problem.getGradOutputAddress()) |
                 reinterpret_cast<std::uintptr_t>(problem.getResidualAddress()) |
                 reinterpret_cast<std::uintptr_t>(problem.getXAddress())) &
                15) == 0)
                ? (problem.getBatchStride() >= 256 ? std::string("Coalesced4") : std::string("Coalesced1"))
            : (problem.getStreams() == 2 || problem.getStreams() == 4 || problem.getStreams() == 8 ||
               problem.getStreams() == 16)
                ? (problem.getChannels() % 8 == 0
                       ? (problem.getBatchStride() >= 256 ? std::string("Vectorized4") : std::string("Vectorized1"))
                       : (problem.getBatchStride() >= 256 ? std::string("Tiled4") : std::string("Tiled1")))
                : std::string("");
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"Coalesced1", "Coalesced4", "", "Tiled1", "Tiled4", "Vectorized1", "Vectorized4"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        // Preserve the established algorithm order, resolving its former internal schedule here.
        const auto algorithms = candidates;
        candidates.clear();
        const auto append = [&](const std::string& name) {
            if (std::find(candidates.begin(), candidates.end(), name) == candidates.end()) candidates.push_back(name);
        };
        const auto streams = problem.getStreams();
        const auto batch   = problem.getBatchStride();
        auto chunks        = batch > 0 ? 1 + 127 / batch : 1;
        if (chunks > streams) chunks = streams;
        if (streams == 16 && chunks == 1) chunks = 2;
        while (chunks > 1 && streams % chunks != 0) --chunks;
        const auto chunk  = chunks > 0 ? streams / chunks : 1;
        const bool bias16 = problem.hasBiasGradient() && problem.getChannels() >= 128 && batch >= 256;
        int tile_b        = 64;
        while (streams <= 32 && tile_b * streams * (streams + 1) * sizeof(float) > 65536) tile_b /= 2;
        const auto resolve = [&](std::string name) {
            if (name == base) {
                if (tile_b != 64) name += "TileB" + std::to_string(tile_b);
            } else if (chunk != 1) {
                name += "Chunk" + std::to_string(chunk);
            }
            if (bias16) name += "Bias16";
            return name;
        };
        for (const auto& algorithm : algorithms) append(resolve(algorithm));
        for (const auto& algorithm : algorithms) {
            for (int chunk_size : {1, 2, 4, 8}) {
                if (algorithm == base && chunk_size != 1) continue;
                if (algorithm.find("Coalesced") != std::string::npos && chunk_size == 8) continue;
                const auto fixed = algorithm + (chunk_size == 1 ? "" : "Chunk" + std::to_string(chunk_size));
                append(fixed);
                append(fixed + "Bias16");
            }
        }

        for (int tile : {1, 2, 4, 8, 16, 32}) {
            const auto name = std::string(base) + "TileB" + std::to_string(tile);
            append(name);
            append(name + "Bias16");
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
