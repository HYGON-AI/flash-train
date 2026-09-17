#ifndef FTRAIN_PRIMITIVE_BASE_HPP_
#define FTRAIN_PRIMITIVE_BASE_HPP_

#include <cstdint>
#include <memory>
#include <vector>

#include "flash_train/common.h"

#include "flash_train/constraints.hpp"
#include "flash_train/error.hpp"
#include "flash_train/resources.hpp"

namespace ftrain {

// One operator implementation. Engines hold shared prototypes; each plan
// creation clones a prototype and configures the clone for one call's
// problem. A configured clone must keep everything it needs after the source
// Ops and Args are destroyed; Tensor memory addresses stay non-owning.
// execute() enqueues asynchronous work on the supplied resources without
// synchronizing it: the Primitive, the referenced memory, and the workspace
// must stay valid until that work completes, and concurrent execute() calls
// on one Primitive require external synchronization.
struct PrimitiveBase {
    virtual ~PrimitiveBase();

    // Returns this implementation's name: a stable null-terminated string with
    // static storage duration. The name appears in engine diagnostics and in
    // the FTRAIN_ENABLED_PRIMITIVES / FTRAIN_DISABLED_PRIMITIVES lists.
    virtual const char* getName() const noexcept = 0;

    // Returns a copy of this object. Members must be values or non-owning
    // pointers so the default copy produces a valid independent clone.
    // Engines clone unconfigured prototypes and configure each copy for one
    // call.
    virtual std::unique_ptr<PrimitiveBase> clone() const = 0;

    // Returns the workspace bytes one execution needs. Queried after
    // configure(), so the value may depend on configured parameters.
    virtual std::uint64_t getRequiredWorkspaceBytes() const noexcept = 0;

    // Enqueues this Primitive's work on resources' stream. Throws Exception
    // with FTRAIN_STATUS_INVALID_ARGUMENT when resources' workspace size is
    // below getRequiredWorkspaceBytes() or the requirement is nonzero while
    // the workspace is null. Asynchronous; see the class comment.
    void execute(const Resources& resources);

  protected:
    // Constructs an unconfigured prototype.
    PrimitiveBase() noexcept = default;

    // Copying and moving stay available to concrete clone() implementations.
    PrimitiveBase(const PrimitiveBase&)            = default;
    PrimitiveBase& operator=(const PrimitiveBase&) = default;
    PrimitiveBase(PrimitiveBase&&)                 = default;
    PrimitiveBase& operator=(PrimitiveBase&&)      = default;

    virtual void executeImpl(const Resources& resources) = 0;
};

// A list of shared immutable Primitive records.
using PrimitiveList = std::vector<std::shared_ptr<const PrimitiveBase>>;

// Primitive bound to one operator family's Problem type.
template<typename Problem>
struct Primitive : PrimitiveBase {
    // Returns success when this implementation can execute problem under
    // constraints, or FTRAIN_STATUS_UNSUPPORTED with a human-readable reason. The
    // engine reports the collected reasons when no candidate applies. Must be
    // safe for concurrent const calls: concurrent selections share one
    // prototype.
    virtual Result isApplicable(const Problem& problem, const Constraints& constraints) const = 0;

    // Stores everything executeImpl() and getRequiredWorkspaceBytes() need
    // from problem; the problem and its source Ops and Args may be destroyed
    // afterwards.
    virtual void configure(const Problem& problem) = 0;
};

}  // namespace ftrain

#endif
