// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_TRACE_HPP_
#define FTRAIN_TRACE_HPP_

namespace ftrain {

// A nested trace range on the constructing thread. Destroy it on the same
// thread, after every range opened after this one has been closed. The
// constructor and destructor never throw.
class ScopedTraceRange final {
  public:
    // Opens a range named message (a null-terminated string owned by the
    // caller). If the platform backend rejects it, the object becomes
    // inactive and the destructor does nothing.
    explicit ScopedTraceRange(const char* message) noexcept;

    // Closes the range opened by this object, or does nothing when inactive.
    ~ScopedTraceRange() noexcept;

    ScopedTraceRange(const ScopedTraceRange&)            = delete;
    ScopedTraceRange& operator=(const ScopedTraceRange&) = delete;
    ScopedTraceRange(ScopedTraceRange&&)                 = delete;
    ScopedTraceRange& operator=(ScopedTraceRange&&)      = delete;

  private:
    bool active_{false};
};

// Emits an instantaneous event named message (a caller-owned null-terminated
// string) on the calling thread. Never throws.
void markTraceEvent(const char* message) noexcept;

}  // namespace ftrain

#endif
