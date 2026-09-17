#ifndef FTRAIN_FINDER_HPP_
#define FTRAIN_FINDER_HPP_

#include "flash_train/ops_args.hpp"
#include "flash_train/context.hpp"
#include "flash_train/primitive/base.hpp"

namespace ftrain {

// A candidate-selection policy consulted by OpsEngine, which visits Finders in
// registration order. Implementations must be safe for concurrent const calls.
class Finder {
  public:
    virtual ~Finder() = default;

    // Returns whether this Finder participates for args and context; when
    // false, OpsEngine skips its other methods.
    virtual bool isEnabled(const Args& args, const SelectionContext& context) const = 0;

    // Returns the records to consider, usually a subset of records. Every
    // element must be non-null; a null candidate makes OpsEngine throw
    // Exception with FTRAIN_STATUS_INTERNAL_ERROR.
    virtual PrimitiveList findCandidates(const PrimitiveList& records, const Args& args,
                                         const SelectionContext& context) const = 0;

    // Reorders candidates from most to least preferred.
    virtual void sortCandidates(const Args& args, const SelectionContext& context, PrimitiveList& candidates) const = 0;
};

}  // namespace ftrain

#endif
