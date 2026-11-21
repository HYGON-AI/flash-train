// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_KDA_G2_RUNNER_HPP_
#define FTRAIN_PRIMITIVE_HYGON_KDA_G2_RUNNER_HPP_

#include <cstdint>

#include "flash_train/common.h"

namespace ftrain {
namespace kda_g2 {

// The G2 KDA pipeline on gfx936: the G2 kernel sequence (chunk 64,
// K = V = 128, bf16, every head its own q/k/v), driven entirely from the host without
// synchronization. Layout: packed THD rows [T, H, D]; cu_seqlens (int32 [N + 1], device)
// splits them into independent sequences, null means one sequence of T rows.
struct Shape {
    std::int64_t tokens = 0;  // T: packed rows
    std::int64_t heads  = 0;  // H
    std::int64_t seqs   = 1;  // N: sequences (cu_seqlens has N + 1 entries; 1 when null)
    bool has_cu_seqlens = false;
    bool l2norm_qk      = true;   // l2-normalize q and k inside the operator
    bool has_state      = false;  // backward only: the forward handed over its recurrent state
                                  //   (BackwardArgs::state_*), so the bN recompute buffers are not needed
    bool has_saved_h = false;     // backward only: the forward saved h (BackwardArgs::saved_h); wu forms
                                  //   v_new from it, so neither the recompute nor its bN buffer runs
    bool g_in_dqdk = false;       // backward only: qn / kn are given and dk directly follows dq in
                                  //   memory, so the fp32 gate scratch (T * H * 512 B) is staged in
                                  //   dq | dk (idle until the tail) instead of the workspace
    bool gate_in_kernel = false;  // true: g is the raw K3 gate (bf16), activated in kernel as
                                  //   lower_bound * sigmoid(exp(A_log) * (g + dt_bias));
                                  // false: g is the per-token log decay (fp32), as given

    // Dense: no cu_seqlens and T % 64 == 0 (identity chunk map). Otherwise the varlen path:
    // chunk tables are built on the device with capacity ceil(T / 64) + N chunks.
    bool isDense() const noexcept { return !has_cu_seqlens && tokens % 64 == 0; }
    std::int64_t chunks() const noexcept {
        return isDense() ? tokens / 64 : (tokens + 63) / 64 + (has_cu_seqlens ? seqs : 1);
    }
};

struct ForwardArgs {
    const void* q;        // bf16 [T, H, 128]
    const void* k;        // bf16 [T, H, 128]
    const void* v;        // bf16 [T, H, 128]
    const void* g;        // gate_in_kernel ? bf16 : fp32, [T, H, 128]
    const void* beta;     // fp32 [T, H], post-activation
    const void* a_log;    // fp32 [H] (gate_in_kernel only)
    const void* dt_bias;  // fp32 [H, 128] (gate_in_kernel only)
    const int* cu_seqlens;
    void* o;       // bf16 [T, H, 128]; doubles as scratch until the recurrence writes it (must not alias inputs)
    void* q_rstd;  // fp32 [T, H] (l2norm_qk only): saved for the backward
    void* k_rstd;  // fp32 [T, H] (l2norm_qk only)
    void* aqk;     // bf16 [T, H, 64]: saved for the backward
    void* akk;     // bf16 [T, H, 64]: saved for the backward
    // Optional (both or none, l2norm_qk only): the forward prep also writes the l2-normalized
    // q / k (bf16 [T, H, 128]) so the backward reads them instead of rebuilding them (kra P-77b).
    void* qn = nullptr;
    void* kn = nullptr;
    // Optional, saved-h (kra P-82): the recurrence also writes its state h (bN, bf16 16384 per
    // chunk-head); the backward then forms v_new in wu and runs no recompute recurrence.
    // Exclusive with the keep-state ports below.
    void* saved_h = nullptr;
    // Optional (all three or none), keep-state: the recurrence's B2 state handed to the backward
    // so it skips its recompute (for activation-checkpoint recompute, where the forward runs
    // right before the backward). bf16 natives, per (chunk, head): hbn 16384, vn / anv 8192.
    void* state_hbn = nullptr;
    void* state_vn  = nullptr;
    void* state_anv = nullptr;
    float scale;
    float lower_bound;
};

struct BackwardArgs {
    const void* d_o;  // bf16 [T, H, 128]
    const void* q;
    const void* k;
    const void* v;
    const void* g;
    const void* beta;
    const void* a_log;
    const void* dt_bias;
    const int* cu_seqlens;
    const void* q_rstd;  // the forward's saved tensors
    const void* k_rstd;
    const void* aqk;
    const void* akk;
    const void* qn = nullptr;  // the forward's optional qn / kn (ForwardArgs::qn / kn): q / k unused then
    const void* kn = nullptr;
    // dq / dk / dg double as scratch until the final kernel writes them: they must not alias any input.
    void* dq;                       // bf16 [T, H, 128]
    void* dk;                       // bf16 [T, H, 128]
    void* dv;                       // bf16 [T, H, 128]
    void* dg;                       // g's dtype, [T, H, 128]
    void* dbeta;                    // fp32 [T, H]
    void* da_log;                   // fp32 [H] (gate_in_kernel only)
    void* ddt_bias;                 // fp32 [H, 128] (gate_in_kernel only)
    const void* saved_h = nullptr;  // optional: the forward's saved h (ForwardArgs::saved_h) == shape.has_saved_h
    // Optional (all three or none): the forward's keep-state outputs (ForwardArgs::state_*).
    const void* state_hbn = nullptr;
    const void* state_vn  = nullptr;
    const void* state_anv = nullptr;
    float scale;
    float lower_bound;
};

std::uint64_t forwardWorkspaceBytes(const Shape& shape) noexcept;
std::uint64_t backwardWorkspaceBytes(const Shape& shape) noexcept;

// Enqueue on stream; throw Exception(FTRAIN_STATUS_INTERNAL_ERROR) when a launch fails.
void forward(const Shape& shape, const ForwardArgs& args, void* workspace, FTrainStream stream);
void backward(const Shape& shape, const BackwardArgs& args, void* workspace, FTrainStream stream);

// True when the current HIP device is a gfx936 (the only architecture the kernels target).
bool deviceSupported(FTrainDeviceId device) noexcept;

}  // namespace kda_g2
}  // namespace ftrain

#endif
