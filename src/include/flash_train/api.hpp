#ifndef FTRAIN_API_HPP_
#    define FTRAIN_API_HPP_

#    include <exception>
#    include <memory>
#    include <new>
#    include <type_traits>
#    include <utility>
#    include <vector>

#    include "flash_train/ops_args.hpp"
#    include "flash_train/error.hpp"
#    include "flash_train/pattern.hpp"
#    include "flash_train/plan.hpp"

namespace ftrain {

// Returns the calling thread's API result. Initially success with an empty
// message. The reference stays valid until thread exit.
const Result& getLastResult() noexcept;

// Resets the calling thread's API result to success with an empty message.
void clearLastResult() noexcept;

// Stores status and message into the calling thread's API result, using
// Result::set() semantics.
void setLastResult(FTrainStatus status, const char* message) noexcept;

// Runs a void callable at a C API exception boundary and converts exceptions
// to a status: ftrain::Exception maps to its own Result, std::bad_alloc maps
// to FTRAIN_STATUS_OUT_OF_MEMORY, other std::exception map to
// FTRAIN_STATUS_INTERNAL_ERROR with what(), and anything else maps to
// FTRAIN_STATUS_INTERNAL_ERROR with a fixed message. Normal completion resets
// the thread's API result. Never throws.
template<typename Function>
FTrainStatus invokeApi(Function&& function) noexcept {
    static_assert(std::is_void_v<std::invoke_result_t<Function&&>>, "invokeApi callable must return void");

    try {
        std::forward<Function>(function)();
        clearLastResult();
        return FTRAIN_STATUS_SUCCESS;
    }
    catch (const Exception& exception) {
        const Result& result = exception.getResult();
        setLastResult(result.getStatus(), result.getMessage());
        return result.getStatus();
    }
    catch (const std::bad_alloc& exception) {
        setLastResult(FTRAIN_STATUS_OUT_OF_MEMORY, exception.what());
        return FTRAIN_STATUS_OUT_OF_MEMORY;
    }
    catch (const std::exception& exception) {
        setLastResult(FTRAIN_STATUS_INTERNAL_ERROR, exception.what());
        return FTRAIN_STATUS_INTERNAL_ERROR;
    }
    catch (...) {
        setLastResult(FTRAIN_STATUS_INTERNAL_ERROR, "Unknown exception caught at C API boundary");
        return FTRAIN_STATUS_INTERNAL_ERROR;
    }
}

}  // namespace ftrain

#endif

// Opaque C API handle backs. The structs live in the global namespace to
// match the C header declarations.
struct FTrainPatternStruct final {
    ftrain::PatternBuilder builder;
};

struct FTrainOpsStruct final {
    FTrainOpsStruct(const ftrain::Pattern& user_pattern, const ftrain::Pattern& supported_pattern)
        : ops(user_pattern, supported_pattern) {}

    ftrain::Ops ops;
};

struct FTrainArgsStruct final {
    explicit FTrainArgsStruct(ftrain::Args&& args_operand) : args(std::move(args_operand)) {}

    ftrain::Args args;
};

struct FTrainPlanStruct final {
    explicit FTrainPlanStruct(ftrain::Plan&& plan_operand) : plan(std::move(plan_operand)) {}

    ftrain::Plan plan;
};
