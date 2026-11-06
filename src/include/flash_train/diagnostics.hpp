// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_DIAGNOSTICS_HPP_
#define FTRAIN_DIAGNOSTICS_HPP_

#include <type_traits>
#include <utility>

#include "flash_train/logging.hpp"

namespace ftrain {
namespace detail {

template<typename Family, typename = void>
struct HasDiagnosticName : std::false_type {};
template<typename Family>
struct HasDiagnosticName<Family, std::void_t<decltype(Family::diagnosticName())>> : std::true_type {};

template<typename Family, typename Problem, typename = void>
struct HasDiagnosticSummary : std::false_type {};
template<typename Family, typename Problem>
struct HasDiagnosticSummary<
    Family, Problem,
    std::void_t<decltype(Family::describeProblem(std::declval<const Problem&>(), std::declval<logging::Buffer&>()))>>
    : std::true_type {};

}  // namespace detail

// Optional Family hooks. Read metadata only: no business API, selection, device
// query, tensor read, mutation or logging reentry. Failures discard this record.
// A diagnostic name, if provided, must return a stable null-terminated string.
template<typename Family, typename Problem>
void logProblem(const Problem& problem) noexcept {
    logging::log(logging::Level::kDebug, "engine", [&](logging::Buffer& record) {
        record.append("event=problem family=");
        if constexpr (detail::HasDiagnosticName<Family>::value) {
            const char* name = Family::diagnosticName();
            record.append(name == nullptr ? "unknown" : name);
        } else {
            record.append("unknown");
        }
        if constexpr (detail::HasDiagnosticSummary<Family, Problem>::value) {
            record.append(" ");
            Family::describeProblem(problem, record);
        }
    });
}

}  // namespace ftrain
#endif
