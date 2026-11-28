// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Per-architecture symbol renaming for the GDN2 gfx938 device sources.
//
// The top-level build may package gfx936 and gfx938 code objects into one
// library (FTRAIN_HYGON_HIP_ARCHS=gfx936;gfx938). Both architectures export
// the same extern "C" ABI names, so every ABI symbol of this directory is
// suffixed with _gfx938 at compile time. The macro must be defined before the
// ABI headers are included, which is why every source and header in this
// directory includes this file first. Kernel names are not listed here: the
// __global__ kernels live in anonymous namespaces and never collide.
//
// The host-side wrappers (src/primitive/hygon/gdn2/*_gfx938.cpp) include the
// headers of this directory and therefore call the suffixed symbols too.
#ifndef FTRAIN_GDN2_GFX938_DEVICE_SYMBOLS_H_
#define FTRAIN_GDN2_GFX938_DEVICE_SYMBOLS_H_

#define flash_train_gdn2_intra_bf16                       flash_train_gdn2_intra_bf16_gfx938
#define flash_train_gdn2_state_h_bf16                     flash_train_gdn2_state_h_bf16_gfx938
#define flash_train_gdn2_chunk_o_bf16                     flash_train_gdn2_chunk_o_bf16_gfx938
#define flash_train_gdn2_chunk_shape_supported            flash_train_gdn2_chunk_shape_supported_gfx938
#define flash_train_gdn2_chunk_fwd_shape_supported        flash_train_gdn2_chunk_fwd_shape_supported_gfx938
#define flash_train_gdn2_chunk_fwd_needs_pad              flash_train_gdn2_chunk_fwd_needs_pad_gfx938
#define flash_train_gdn2_chunk_workspace_bytes            flash_train_gdn2_chunk_workspace_bytes_gfx938
#define flash_train_gdn2_chunk_fwd_workspace_bytes        flash_train_gdn2_chunk_fwd_workspace_bytes_gfx938
#define flash_train_gdn2_chunk_forward_bf16               flash_train_gdn2_chunk_forward_bf16_gfx938
#define flash_train_gdn2_bwd_dav_bf16                     flash_train_gdn2_bwd_dav_bf16_gfx938
#define flash_train_gdn2_bwd_dhu_bf16                     flash_train_gdn2_bwd_dhu_bf16_gfx938
#define flash_train_gdn2_bwd_wy_dqkg_bf16                 flash_train_gdn2_bwd_wy_dqkg_bf16_gfx938
#define flash_train_gdn2_bwd_intra_bf16                   flash_train_gdn2_bwd_intra_bf16_gfx938
#define flash_train_gdn2_chunk_backward_shape_supported   flash_train_gdn2_chunk_backward_shape_supported_gfx938
#define flash_train_gdn2_chunk_backward_needs_pad         flash_train_gdn2_chunk_backward_needs_pad_gfx938
#define flash_train_gdn2_chunk_backward_workspace_bytes   flash_train_gdn2_chunk_backward_workspace_bytes_gfx938
#define flash_train_gdn2_chunk_backward_bf16              flash_train_gdn2_chunk_backward_bf16_gfx938

#endif  // FTRAIN_GDN2_GFX938_DEVICE_SYMBOLS_H_
