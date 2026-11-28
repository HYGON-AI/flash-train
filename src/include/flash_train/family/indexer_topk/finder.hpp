// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_TOPK_FINDER_HPP_
#define FTRAIN_FAMILY_INDEXER_TOPK_FINDER_HPP_

#include <cstdint>
#include <algorithm>
#include <string>
#include <vector>

#include "flash_train/family/indexer_topk/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct IndexerTopKRegistrationOrderFinder final {
    static const char* getName() { return "IndexerTopKRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const IndexerTopKProblem& problem, const Constraints&) {
        const char* base = "IndexerTopK";
        const auto preferred =
            !problem.getOrderValues() && problem.getOutValuesAddress() == nullptr && problem.getNumColumns() >= 1024 &&
                    problem.getTopK() >= static_cast<std::int64_t>(problem.getNumColumns()) &&
                    problem.getTopK() % 4 == 0
                ? std::string("GroupedAll")
            : (problem.getOrderValues() && problem.getNumColumns() == 4096 && problem.getTopK() > 1024 &&
               problem.getTopK() <= 2048)
                ? std::string("OrderedUInt")
            : (!problem.getOrderValues() && problem.getNumColumns() == 4096 && problem.getTopK() == 2048 &&
               problem.getNextN() == 1)
                ? std::string("GroupedRadix")
            : (problem.getNumColumns() > 1024 || (problem.getNumColumns() == 1024 && problem.getTopK() <= 512) ||
               (problem.getNumColumns() > 512 && problem.getTopK() > 1024))
                ? std::string("WideRadix")
                : (problem.getNumColumns() > 512 && problem.getTopK() < 1024 ? std::string("Radix1024")
                                                                             : std::string(""));
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"OrderedUInt", "GroupedAll", "GroupedRadix", "", "WideRadix", "Radix1024"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        // Preserve the established algorithm order, resolving its former internal schedule here.
        const auto algorithms = candidates;
        candidates.clear();
        const auto append = [&](const std::string& name) {
            if (std::find(candidates.begin(), candidates.end(), name) == candidates.end()) candidates.push_back(name);
        };
        const auto columns = problem.getNumColumns();
        const auto resolve = [&](std::string name) {
            if (name == base) {
                int size = 4;
                while (size < columns && size < 1024) size <<= 1;
                if (size != 1024) name += "Size" + std::to_string(size);
            } else if (name == std::string(base) + "WideRadix") {
                const int capacity = columns <= 2048 ? 8 : columns <= 4096 ? 16 : columns <= 8192 ? 32 : 64;
                if (capacity != 8) name += "Capacity" + std::to_string(capacity);
            }
            return name;
        };
        for (const auto& algorithm : algorithms) append(resolve(algorithm));
        append(base);
        for (const char* suffix : {"Size4", "Size8", "Size16", "Size32", "Size64", "Size128", "Size256", "Size512",
                                   "WideRadix", "WideRadixCapacity16", "WideRadixCapacity32", "WideRadixCapacity64"})
            append(std::string(base) + suffix);

        return candidates;
    }
};

}  // namespace ftrain

#endif
