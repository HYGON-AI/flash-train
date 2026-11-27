// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Per-architecture symbol renaming for the GDN gfx938 device sources.
//
// The top-level build may package gfx936 and gfx938 code objects into one
// library (FTRAIN_HYGON_HIP_ARCHS=gfx936;gfx938). Both architectures export
// the same extern "C" ABI names, so every ABI symbol of this directory is
// suffixed with _gfx938 at compile time. The macro must be defined before the
// ABI headers are included, which is why every source and header in this
// directory includes this file first. Kernel names are not listed here: the
// __global__ kernels live in anonymous namespaces and never collide.
//
// The host-side wrappers (src/primitive/hygon/gdn/*_gfx938.cpp) include the
// headers of this directory and therefore call the suffixed symbols too.
#ifndef FTRAIN_GDN_GFX938_DEVICE_SYMBOLS_H_
#define FTRAIN_GDN_GFX938_DEVICE_SYMBOLS_H_

#define flash_train_gdn_chunk_supported                   flash_train_gdn_chunk_supported_gfx938
#define flash_train_gdn_chunk_forward_config              flash_train_gdn_chunk_forward_config_gfx938
#define flash_train_gdn_chunk_forward_route_string        flash_train_gdn_chunk_forward_route_string_gfx938
#define flash_train_gdn_chunk_value_tile                  flash_train_gdn_chunk_value_tile_gfx938
#define flash_train_gdn_chunk_states_bytes                flash_train_gdn_chunk_states_bytes_gfx938
#define flash_train_gdn_chunk_set_phase_mask              flash_train_gdn_chunk_set_phase_mask_gfx938
#define flash_train_gdn_chunk_fp32_forward_launch         flash_train_gdn_chunk_fp32_forward_launch_gfx938
#define flash_train_gdn_chunk_bf16_forward_launch         flash_train_gdn_chunk_bf16_forward_launch_gfx938
#define flash_train_gdn_intra_bf16                        flash_train_gdn_intra_bf16_gfx938
#define flash_train_gdn_chunk_o_bf16                      flash_train_gdn_chunk_o_bf16_gfx938
#define flash_train_gdn_state_h_bf16                      flash_train_gdn_state_h_bf16_gfx938
#define flash_train_gdn_bwd_dv_local_bf16                 flash_train_gdn_bwd_dv_local_bf16_gfx938
#define flash_train_gdn_bwd_dhu_bf16                      flash_train_gdn_bwd_dhu_bf16_gfx938
#define flash_train_gdn_bwd_dqkwg_bf16                    flash_train_gdn_bwd_dqkwg_bf16_gfx938
#define flash_train_gdn_bwd_dqkwg_dg_planes               flash_train_gdn_bwd_dqkwg_dg_planes_gfx938
#define flash_train_gdn_bwd_wy_bf16                       flash_train_gdn_bwd_wy_bf16_gfx938
#define flash_train_gdn_bwd_groupsum_bf16                 flash_train_gdn_bwd_groupsum_bf16_gfx938
#define flash_train_gdn_chunk_backward_supported          flash_train_gdn_chunk_backward_supported_gfx938
#define flash_train_gdn_chunk_backward_workspace_bytes    flash_train_gdn_chunk_backward_workspace_bytes_gfx938
#define flash_train_gdn_chunk_bf16_backward_launch        flash_train_gdn_chunk_bf16_backward_launch_gfx938
#define flash_train_gdn_bwd_pad_copy                      flash_train_gdn_bwd_pad_copy_gfx938
#define flash_train_gdn_split_supported                   flash_train_gdn_split_supported_gfx938
#define flash_train_gdn_split_workspace                   flash_train_gdn_split_workspace_gfx938
#define flash_train_gdn_split_bf16_forward_launch         flash_train_gdn_split_bf16_forward_launch_gfx938
#define flash_train_gdn_split_fp32_forward_launch         flash_train_gdn_split_fp32_forward_launch_gfx938

#endif  // FTRAIN_GDN_GFX938_DEVICE_SYMBOLS_H_
