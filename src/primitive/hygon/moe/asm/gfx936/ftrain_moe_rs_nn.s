.amdgcn_target "amdgcn-amd-amdhsa--gfx936"
.ifndef FTRAIN_FC2_TILE_SWIZZLE_LOG2
.set FTRAIN_FC2_TILE_SWIZZLE_LOG2, 3
.endif
.text
	.protected ftrain_moe_rs_nn
	.globl ftrain_moe_rs_nn
	.p2align 8
	.type ftrain_moe_rs_nn,@function
	.section .rodata,#alloc
	.p2align 6
	.amdhsa_kernel ftrain_moe_rs_nn
	  .amdhsa_user_sgpr_kernarg_segment_ptr 1
	  .amdhsa_next_free_vgpr 255
	  .amdhsa_next_free_sgpr 101
	  .amdhsa_group_segment_fixed_size 65536
	  .amdhsa_private_segment_fixed_size 0
	  .amdhsa_system_sgpr_workgroup_id_x 1
	  .amdhsa_system_sgpr_workgroup_id_y 1
	  .amdhsa_system_sgpr_workgroup_id_z 1
	  .amdhsa_system_vgpr_workitem_id 0
	.end_amdhsa_kernel
.text

.amdgpu_metadata
---
amdhsa.version: [1, 0]
amdhsa.kernels:
  - .name: ftrain_moe_rs_nn
    .symbol: ftrain_moe_rs_nn.kd
    .args:
      - .name: input
        .offset: 0
        .size: 8
        .value_kind: by_value
      - .name: weight
        .offset: 8
        .size: 8
        .value_kind: by_value
      - .name: gather_index
        .offset: 16
        .size: 8
        .value_kind: by_value
      - .name: output
        .offset: 24
        .size: 8
        .value_kind: by_value
      - .name: rank_signals
        .offset: 32
        .size: 8
        .value_kind: by_value
      - .name: tile_info
        .offset: 40
        .size: 8
        .value_kind: by_value
      - .name: problem_count
        .offset: 48
        .size: 4
        .value_kind: by_value
      - .name: n_dim
        .offset: 52
        .size: 4
        .value_kind: by_value
      - .name: k_dim
        .offset: 56
        .size: 4
        .value_kind: by_value
      - .name: topk
        .offset: 60
        .size: 4
        .value_kind: by_value
      - .name: rank
        .offset: 64
        .size: 4
        .value_kind: by_value
      - .name: signal_value
        .offset: 68
        .size: 4
        .value_kind: by_value
      - .name: epilogue_mode
        .offset: 72
        .size: 4
        .value_kind: by_value
      - .name: chunk_size_n
        .offset: 76
        .size: 4
        .value_kind: by_value
      - .name: num_done_groups
        .offset: 80
        .size: 4
        .value_kind: by_value
      - .name: group_major
        .offset: 84
        .size: 4
        .value_kind: by_value
      - .name: sorted_token_ids
        .offset: 88
        .size: 8
        .value_kind: by_value
      - .name: chunk_done_counts
        .offset: 96
        .size: 8
        .value_kind: by_value
      - .name: chunk_done_flags
        .offset: 104
        .size: 8
        .value_kind: by_value
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 112
    .max_flat_workgroup_size: 512
    .private_segment_fixed_size: 0
    .sgpr_count: 101
    .sgpr_spill_count: 0
    .vgpr_count: 255
    .vgpr_spill_count: 0
    .wavefront_size: 64
...
.end_amdgpu_metadata
ftrain_moe_rs_nn:

.macro _v_add_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_add_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_add_u32 dst:req, src0:req, src1:req, dpp=
   v_add_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_add_i32 dst:req, src0:req, src1:req, dpp=
   v_add_i32 \dst, \src0, \src1 \dpp
.endm

.macro _v_addc_co_u32 dst:req, ccOut:req, src0:req, ccIn:req, src1:req, dpp=
   v_addc_co_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp
.endm

.macro _v_sub_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_sub_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_sub_u32 dst:req, src0:req, src1:req, dpp=
   v_sub_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_sub_i32 dst:req, src0:req, src1:req, dpp=
   v_sub_i32 \dst, \src0, \src1 \dpp
.endm

.macro _v_add_lshl_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_add_lshl_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_lshl_add_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_lshl_add_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_lshl_or_b32 dst:req, src0:req, shiftCnt:req, src1:req
    v_lshl_or_b32 \dst, \src0, \shiftCnt, \src1
.endm

.macro _v_cmpx_lt_i16 dst, src0, src1=
   v_cmpx_lt_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_lt_i32 dst, src0, src1=
   v_cmpx_lt_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_lt_i64 dst, src0, src1=
   v_cmpx_lt_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_lt_u16 dst, src0, src1=
   v_cmpx_lt_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_lt_u32 dst, src0, src1=
   v_cmpx_lt_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_lt_u64 dst, src0, src1=
   v_cmpx_lt_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_eq_i16 dst, src0, src1=
   v_cmpx_eq_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_eq_i32 dst, src0, src1=
   v_cmpx_eq_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_eq_i64 dst, src0, src1=
   v_cmpx_eq_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_eq_u16 dst, src0, src1=
   v_cmpx_eq_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_eq_u32 dst, src0, src1=
   v_cmpx_eq_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_eq_u64 dst, src0, src1=
   v_cmpx_eq_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_le_i16 dst, src0, src1=
   v_cmpx_le_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_le_i32 dst, src0, src1=
   v_cmpx_le_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_le_i64 dst, src0, src1=
   v_cmpx_le_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_le_u16 dst, src0, src1=
   v_cmpx_le_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_le_u32 dst, src0, src1=
   v_cmpx_le_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_le_u64 dst, src0, src1=
   v_cmpx_le_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_gt_i16 dst, src0, src1=
   v_cmpx_gt_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_gt_i32 dst, src0, src1=
   v_cmpx_gt_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_gt_i64 dst, src0, src1=
   v_cmpx_gt_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_gt_u16 dst, src0, src1=
   v_cmpx_gt_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_gt_u32 dst, src0, src1=
   v_cmpx_gt_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_gt_u64 dst, src0, src1=
   v_cmpx_gt_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_ne_i16 dst, src0, src1=
   v_cmpx_ne_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_ne_i32 dst, src0, src1=
   v_cmpx_ne_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_ne_i64 dst, src0, src1=
   v_cmpx_ne_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_ne_u16 dst, src0, src1=
   v_cmpx_ne_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_ne_u32 dst, src0, src1=
   v_cmpx_ne_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_ne_u64 dst, src0, src1=
   v_cmpx_ne_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_lg_i16 dst, src0, src1=
   v_cmpx_lg_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_lg_i32 dst, src0, src1=
   v_cmpx_lg_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_lg_i64 dst, src0, src1=
   v_cmpx_lg_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_lg_u16 dst, src0, src1=
   v_cmpx_lg_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_lg_u32 dst, src0, src1=
   v_cmpx_lg_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_lg_u64 dst, src0, src1=
   v_cmpx_lg_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_ge_i16 dst, src0, src1=
   v_cmpx_ge_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_ge_i32 dst, src0, src1=
   v_cmpx_ge_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_ge_i64 dst, src0, src1=
   v_cmpx_ge_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_ge_u16 dst, src0, src1=
   v_cmpx_ge_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_ge_u32 dst, src0, src1=
   v_cmpx_ge_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_ge_u64 dst, src0, src1=
   v_cmpx_ge_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_o_i16 dst, src0, src1=
   v_cmpx_o_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_o_i32 dst, src0, src1=
   v_cmpx_o_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_o_i64 dst, src0, src1=
   v_cmpx_o_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_o_u16 dst, src0, src1=
   v_cmpx_o_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_o_u32 dst, src0, src1=
   v_cmpx_o_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_o_u64 dst, src0, src1=
   v_cmpx_o_u64 \dst, \src0, \src1
.endm

.macro _v_cmpx_u_i16 dst, src0, src1=
   v_cmpx_u_i16 \dst, \src0, \src1
.endm

.macro _v_cmpx_u_i32 dst, src0, src1=
   v_cmpx_u_i32 \dst, \src0, \src1
.endm

.macro _v_cmpx_u_i64 dst, src0, src1=
   v_cmpx_u_i64 \dst, \src0, \src1
.endm

.macro _v_cmpx_u_u16 dst, src0, src1=
   v_cmpx_u_u16 \dst, \src0, \src1
.endm

.macro _v_cmpx_u_u32 dst, src0, src1=
   v_cmpx_u_u32 \dst, \src0, \src1
.endm

.macro _v_cmpx_u_u64 dst, src0, src1=
   v_cmpx_u_u64 \dst, \src0, \src1
.endm
.macro _v_mac_f32 c:req, a:req, b:req
    v_mac_f32 \c, \a, \b
.endmacro

.macro V_MAGIC_DIV dstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    v_mul_hi_u32 v[\dstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\dstIdx+0], \dividend, \magicA
    _v_add_u32 v[\dstIdx+0], v[\dstIdx+0], v[\dstIdx+1]
    v_lshrrev_b32 v[\dstIdx+0], \magicShift, v[\dstIdx+0]
.endm

.set vgprValuC, 0

.set vgprValuA_X0_I0, 128
.set vgprValuA_X1_I0, 144
.set vgprG2LA, 180
.set vgprValuB_X0_I0, 160
.set vgprValuB_X1_I0, 168
.set vgprG2LB, 184
.set vgprLocalWriteAddrA, 176
.set vgprLocalWriteAddrB, 177
.set vgprGlobalReadOffsetA, 178
.set vgprGlobalReadOffsetB, 179
.set vgprLocalReadAddrA, 188
.set vgprLocalReadAddrB, 189
.set vgprLocalReadAddrA_ori, 190
.set vgprLocalReadAddrB_ori, 191
.set vgprLocalReadAddrASwap, 192
.set vgprLocalReadAddrASwap_ori, 193
.set vgprLocalReadAddrBSwap, 194
.set vgprLocalReadAddrBSwap_ori, 195
.set vgprSerial, 196
.set vgprGlobalReadOffsetB1, 197
.set vgprSerial_forLdsWrapOptA, 198
.set vgprSerial_forLdsWrapOptB, 199
.set vgprLocalReadAddrB_forLdsWrapTailLoop, 200

.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprLoopCounterL, 5
.set sgprOrigLoopCounter, 6
.set sgprSrdA, 8
.set sgprSrdB, 12
.set sgprSrdD, 16
.set sgprSrdC, 20
.set sgprTensor2dSizeA, 24
.set sgprTensor2dSizeB, 26
.set sgprAddressD, 28
.set sgprAddressC, 30
.set sgprAddressA, 32
.set sgprAddressB, 34
.set sgprAlpha, 36
.set sgprBeta, 37
.set sgprStridesD, 38
.set sgprStridesC, 40
.set sgprStridesA, 42
.set sgprStridesB, 44
.set sgprSizesFree, 46
.set sgprSizesSum, 49
.set sgprOrigStaggerUIter, 50
.set sgprNumWorkGroups0, 51
.set sgprNumWorkGroups1, 52
.set sgprNumFullBlocks, 53
.set sgprWgmRemainder1, 54
.set sgprMagicNumberWgmRemainder1, 55
.set sgprOffsetD, 56
.set sgprOffsetC, 57
.set sgprOffsetA, 58
.set sgprOffsetB, 59
.set sgprShadowLimitA, 56
.set sgprShadowLimitB, 58
.set sgprGlobalReadIncsA, 7
.set sgprGlobalReadIncsB, 50
.set sgprLocalWriteAddrA, 60
.set sgprLocalWriteAddrB, 61
.set sgprMask, 72
.set sgprWaveiD, 73
.set sgprStrideSwizzle, 74
.set sgprStructNum, 75
.set sgprStructBit, 76
.set sgprStructBit1, 77
.set sgprStrideStruct, 78
.set sgprOffsetStruct, 79
.set sgprShadowLimitB_forTailLoop, 82
.set sgprBarrierPtrLow, 90
.set sgprBarrierPtrHigh, 91
.set sgprRank, 92
.set sgprWorldSize, 93

.set sgprSizeI, sgprSizesFree+0
.set sgprSizeJ, sgprSizesFree+1
.set sgprSizeK, sgprSizesFree+2
.set sgprSizeL, sgprSizesSum+0

.set constStrideD0I, 1
.set sgprStrideD1J, sgprStridesD+0
.set sgprStrideDK, sgprStridesD+1
.set constStrideC0I, 1
.set sgprStrideC1J, sgprStridesC+0
.set sgprStrideCK, sgprStridesC+1
.set constStrideA0I, 1
.set sgprStrideAL, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1
.set sgprStrideoffC, sgprSrdA
.set sgprStrideoffD, sgprAddressD
.set sgprEdgeCheck, sgprAddressB

.set MT0, 256
.set MT1, 256
.set DepthU, 16
.set GSU, 1
.set BpeA, 2
.set BpeALog2, 1
.set BpeB, 2
.set BpeBLog2, 1

.set SrdShiftLeftA, 8
.set SrdShiftLeftB, 8

.set BufferLimit, 0xfffffffe
.set BufferOOB, 0xfffffffe

.set Srd127_96, 0x00020000

.macro GLOBAL_OFFSET_A vgprAddr:req vgprOffset0I:req vgprOffsetL:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideAL], v[\vgprOffsetL]
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffset0I], v[\vgprTmp+0]
_v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]
v_lshlrev_b32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]
.endm

.macro GLOBAL_OFFSET_B_1 vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J]
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0]
_v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]
v_lshlrev_b32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]
.endm

.macro GLOBAL_OFFSET_B1 vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideStruct], v[\vgprOffset1J]
v_mov_b32 v[\vgprAddr+0], v[\vgprOffset1J]
v_add_u32 v[\vgprAddr+1], v[\vgprOffsetL], v[\vgprTmp+0]
v_add_co_u32  v[\vgprAddr+1],vcc, 0x8, v[\vgprAddr+1]
v_lshlrev_b32 v[\vgprAddr+1], 0x1, v[\vgprAddr+1]
.endm

.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
v_lshlrev_b32 v[\vgprAddr+0], s[sgprStrideStruct], v[\vgprOffset1J]
v_mov_b32 v[\vgprAddr+1], v[\vgprOffsetL]
v_add_co_u32  v[\vgprAddr+1],vcc, 0x8, v[\vgprAddr+1]
v_lshlrev_b32 v[\vgprAddr+1], 0x1, v[\vgprAddr+1]
.endm

.macro GLOBAL_STRUCT_OFFSET_B vgprAddr:req vgprOffset:req vgprIndex:req
v_mul_lo_u32 v[\vgprAddr], s[sgprOffsetStruct], v[\vgprIndex]
_v_add_u32 v[\vgprAddr], v[\vgprAddr], v[\vgprOffset]
_v_add_u32 v[\vgprAddr], 8, v[\vgprAddr]
v_lshlrev_b32 v[\vgprAddr], 1, v[\vgprAddr]
.endm
.macro GLOBAL_STRUCT_INDEX_B vgprAddr:req vgprIndex:req
v_mul_lo_u32 v[\vgprAddr], s[sgprStrideStruct], v[\vgprIndex]
.endm

.macro GLOBAL_ROWMAP_D_ADDR vgprAddr:req, vgprCoord0:req, vgprCoord1:req, sgprMask:req
v_cndmask_b32 v253, 0, v[\vgprCoord1], s[\sgprMask:\sgprMask+1]
_v_add_u32 v253, s7, v253
v_lshlrev_b32 v253, 0x2, v253
buffer_load_dword v252, v253, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 glc
s_waitcnt vmcnt(0)
v_cmp_gt_i32 vcc, v252, -1
s_and_b64 s[\sgprMask:\sgprMask+1], s[\sgprMask:\sgprMask+1], vcc
v_mul_lo_u32 v[\vgprAddr], s[sgprStrideD1J], v252
v_mul_hi_u32 v[\vgprAddr+20], s[sgprStrideD1J], v252
_v_add_co_u32 v[\vgprAddr], vcc, v[\vgprAddr], v[\vgprCoord0]
_v_addc_co_u32 v[\vgprAddr+20], vcc, v[\vgprAddr+20], 0, vcc
v_lshrrev_b32 v254, 31, v[\vgprAddr]
v_lshlrev_b32 v253, 1, v[\vgprAddr]
v_lshlrev_b32 v[\vgprAddr+20], 1, v[\vgprAddr+20]
v_or_b32 v[\vgprAddr+20], v[\vgprAddr+20], v254
v_mov_b32 v[\vgprAddr], v253
v_mov_b32 v253, s[sgprSrdD+0]
v_mov_b32 v254, s[sgprSrdD+1]
_v_add_co_u32 v[\vgprAddr], vcc, v[\vgprAddr], v253
_v_addc_co_u32 v[\vgprAddr+20], vcc, v[\vgprAddr+20], v254, vcc
.endm

.macro BUFFER_STORE_SHORT_D_CLIPPED vgprVal:req, vgprAddr:req, sgprMask:req
s_and_saveexec_b64 s[54:55], s[\sgprMask:\sgprMask+1]
v_mov_b32 v252, v[\vgprAddr]
v_mov_b32 v253, v[\vgprAddr+20]
global_store_short v[252:253], v[\vgprVal], off
s_or_b64 exec, exec, s[54:55]
.endm

.macro PREFETCH_ROWMAP_D rowToken:req, rowOffset:req
s_mov_b32 s56, \rowOffset
_v_add_co_u32 v133, vcc, v129, s56
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]
v_cndmask_b32 v253, 0, v133, s[60:61]
_v_add_u32 v253, s7, v253
v_lshlrev_b32 v253, 0x2, v253
buffer_load_dword v[\rowToken], v253, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 glc
.endm

.macro GLOBAL_ROWMAP_D_ROW_BASE rowToken:req, rowOffset:req
s_mov_b32 s56, \rowOffset
_v_add_co_u32 v133, vcc, v129, s56
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]
v_mov_b32 v252, v[\rowToken]
v_cmp_gt_i32 vcc, v252, -1
s_and_b64 s[60:61], s[60:61], vcc
v_mul_lo_u32 v134, s[sgprStrideD1J], v252
v_mul_hi_u32 v154, s[sgprStrideD1J], v252
v_lshrrev_b32 v254, 31, v134
v_lshlrev_b32 v134, 1, v134
v_lshlrev_b32 v154, 1, v154
v_or_b32 v154, v154, v254
v_mov_b32 v253, s[sgprSrdD+0]
v_mov_b32 v254, s[sgprSrdD+1]
_v_add_co_u32 v134, vcc, v134, v253
_v_addc_co_u32 v154, vcc, v154, v254, vcc
.endm

.macro DIRECT_ROWMAP_D_STORE valuIndex:req, nOffset:req
.if \nOffset == 0
v_mov_b32 v132, v128
.elseif \nOffset == 16
_v_add_co_u32 v132, vcc, v128, 16
.elseif \nOffset == 64
_v_add_co_u32 v132, vcc, v128, 64
.elseif \nOffset == 80
_v_add_co_u32 v132, vcc, v128, s66
.elseif \nOffset == 128
_v_add_co_u32 v132, vcc, v128, s67
.elseif \nOffset == 144
_v_add_co_u32 v132, vcc, v128, s68
.elseif \nOffset == 192
_v_add_co_u32 v132, vcc, v128, s69
.elseif \nOffset == 208
_v_add_co_u32 v132, vcc, v128, s70
.else
.error "unsupported native Ailk direct-output N offset"
.endif
v_cmp_lt_u32 s[62:63], v132, s[sgprSizeI]
s_and_b64 s[62:63], s[60:61], s[62:63]
v_lshlrev_b32 v252, 1, v132
_v_add_co_u32 v252, vcc, v134, v252
_v_addc_co_u32 v253, vcc, v154, 0, vcc
FP32_TO_BF16 vgprValuC+\valuIndex
s_and_saveexec_b64 s[54:55], s[62:63]
global_store_short v[252:253], v[vgprValuC+\valuIndex], off
s_or_b64 exec, exec, s[54:55]
.endm

.macro BUFFER_ROWMAP_D_ROW_BASE rowToken:req, rowOffset:req
s_mov_b32 s56, \rowOffset
_v_add_co_u32 v133, vcc, v129, s56
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]
v_mov_b32 v252, v[\rowToken]
v_cmp_gt_i32 vcc, v252, -1
s_and_b64 s[60:61], s[60:61], vcc
v_mul_lo_u32 v134, s[sgprStrideD1J], v252
v_lshlrev_b32 v134, 1, v134
.endm

.macro DIRECT_ROWMAP_D_BUFFER_STORE valuIndex:req, nOffset:req
.if \nOffset == 0
v_mov_b32 v132, v128
.elseif \nOffset == 16
_v_add_co_u32 v132, vcc, v128, 16
.elseif \nOffset == 64
_v_add_co_u32 v132, vcc, v128, 64
.elseif \nOffset == 80
_v_add_co_u32 v132, vcc, v128, s66
.elseif \nOffset == 128
_v_add_co_u32 v132, vcc, v128, s67
.elseif \nOffset == 144
_v_add_co_u32 v132, vcc, v128, s68
.elseif \nOffset == 192
_v_add_co_u32 v132, vcc, v128, s69
.elseif \nOffset == 208
_v_add_co_u32 v132, vcc, v128, s70
.else
.error "unsupported native Ailk direct-output N offset"
.endif
v_cmp_lt_u32 s[62:63], v132, s[sgprSizeI]
s_and_b64 s[62:63], s[60:61], s[62:63]
v_lshlrev_b32 v252, 1, v132
_v_add_u32 v252, v134, v252
FP32_TO_BF16 vgprValuC+\valuIndex
s_and_saveexec_b64 s[54:55], s[62:63]
buffer_store_short v[vgprValuC+\valuIndex], v252, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0
s_or_b64 exec, exec, s[54:55]
.endm

.macro DIRECT_ROWMAP_D_BUFFER_ROW rowToken:req, rowOffset:req, valuBase:req
BUFFER_ROWMAP_D_ROW_BASE \rowToken, \rowOffset
DIRECT_ROWMAP_D_BUFFER_STORE \valuBase+0, 0
DIRECT_ROWMAP_D_BUFFER_STORE \valuBase+4, 16
DIRECT_ROWMAP_D_BUFFER_STORE \valuBase+8, 64
DIRECT_ROWMAP_D_BUFFER_STORE \valuBase+12, 80
DIRECT_ROWMAP_D_BUFFER_STORE \valuBase+16, 128
DIRECT_ROWMAP_D_BUFFER_STORE \valuBase+20, 144
DIRECT_ROWMAP_D_BUFFER_STORE \valuBase+24, 192
DIRECT_ROWMAP_D_BUFFER_STORE \valuBase+28, 208
.endm

.macro GLOBAL_OFFSET_B_TailLoop vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J]
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0]
_v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]
v_lshlrev_b32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]
.endm

.macro DYNAMIC_VECTOR_DIVIDE vQuotient vRemainder vDividend vDivisor vTmp0 vTmp1 sTmp
v_cvt_f32_u32 v[\vQuotient], v[\vDivisor]
v_rcp_f32 v[\vQuotient], v[\vQuotient]
v_mul_f32 v[\vQuotient], 0x4f800000, v[\vQuotient]
v_cvt_u32_f32 v[\vQuotient], v[\vQuotient]
v_mul_lo_u32 v[\vRemainder], v[\vDivisor], v[\vQuotient]
v_mul_hi_u32 v[\vTmp0], v[\vDivisor], v[\vQuotient]
_v_sub_co_u32 v[\vTmp1], vcc, 0x0, v[\vRemainder]
v_cmp_ne_i32 s[\sTmp:\sTmp+1], 0x0, v[\vTmp0]
v_cndmask_b32 v[\vRemainder], v[\vTmp1], v[\vRemainder], s[\sTmp:\sTmp+1]
v_mul_hi_u32 v[\vRemainder], v[\vRemainder], v[\vQuotient]
_v_sub_co_u32 v[\vTmp0], vcc, v[\vQuotient], v[\vRemainder]
_v_add_co_u32 v[\vQuotient], vcc, v[\vQuotient], v[\vRemainder]
v_cndmask_b32 v[\vQuotient], v[\vQuotient], v[\vTmp0], s[\sTmp:\sTmp+1]
v_mul_hi_u32 v[\vQuotient], v[\vQuotient], v[\vDividend]
v_mul_lo_u32 v[\vRemainder], v[\vQuotient], v[\vDivisor]
_v_sub_co_u32 v[\vTmp0], vcc, v[\vDividend], v[\vRemainder]
v_cmp_ge_u32 s[\sTmp:\sTmp+1], v[\vDividend], v[\vRemainder]
_v_add_co_u32 v[\vRemainder], vcc, 0x1, v[\vQuotient]
_v_add_co_u32 v[\vTmp1], vcc, -1, v[\vQuotient]
v_cmp_le_u32 vcc, v[\vDivisor], v[\vTmp0]
s_and_b64 vcc, s[\sTmp:\sTmp+1], vcc
v_cndmask_b32 v[\vQuotient], v[\vQuotient], v[\vRemainder], vcc
v_cndmask_b32 v[\vQuotient], v[\vTmp1], v[\vQuotient], s[\sTmp:\sTmp+1]
v_cmp_ne_i32 vcc, 0x0, v[\vDivisor]
v_cndmask_b32 v[\vQuotient], -1, v[\vQuotient], vcc
v_mul_lo_u32 v[\vRemainder], v[\vQuotient], v[\vDivisor]
_v_sub_co_u32 v[\vRemainder], vcc, v[\vDividend], v[\vRemainder]
.endm

.macro MAC_16x8_X0

v_mmac_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+0:vgprValuC+3]
s_setprio 1
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4:vgprValuC+7], v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+4:vgprValuC+7]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+11], v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+8:vgprValuC+11]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12:vgprValuC+15], v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+12:vgprValuC+15]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+19], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+16:vgprValuC+19]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20:vgprValuC+23], v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+20:vgprValuC+23]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+27], v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+24:vgprValuC+27]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28:vgprValuC+31], v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+28:vgprValuC+31]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+35], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+32:vgprValuC+35]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36:vgprValuC+39], v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+36:vgprValuC+39]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+43], v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+40:vgprValuC+43]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44:vgprValuC+47], v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+44:vgprValuC+47]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+51], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+48:vgprValuC+51]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52:vgprValuC+55], v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+52:vgprValuC+55]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+59], v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+56:vgprValuC+59]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60:vgprValuC+63], v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+60:vgprValuC+63]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+67], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+64:vgprValuC+67]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68:vgprValuC+71], v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+68:vgprValuC+71]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+75], v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+72:vgprValuC+75]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76:vgprValuC+79], v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+76:vgprValuC+79]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80:vgprValuC+83], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+80:vgprValuC+83]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84:vgprValuC+87], v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+84:vgprValuC+87]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88:vgprValuC+91], v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+88:vgprValuC+91]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92:vgprValuC+95], v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+92:vgprValuC+95]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96:vgprValuC+99], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+96:vgprValuC+99]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100:vgprValuC+103], v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+100:vgprValuC+103]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104:vgprValuC+107], v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+104:vgprValuC+107]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108:vgprValuC+111], v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+108:vgprValuC+111]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112:vgprValuC+115], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+112:vgprValuC+115]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116:vgprValuC+119], v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+116:vgprValuC+119]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120:vgprValuC+123], v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+120:vgprValuC+123]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124:vgprValuC+127], v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+124:vgprValuC+127]
s_setprio 0
.endm
.macro MAC_16x8_X1

v_mmac_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+1], v[vgprValuC+0:vgprValuC+3]
s_setprio 1
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4:vgprValuC+7], v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+1], v[vgprValuC+4:vgprValuC+7]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8:vgprValuC+11], v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+5], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+1], v[vgprValuC+8:vgprValuC+11]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12:vgprValuC+15], v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+7], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+1], v[vgprValuC+12:vgprValuC+15]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16:vgprValuC+19], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+9], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+1], v[vgprValuC+16:vgprValuC+19]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20:vgprValuC+23], v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+11], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+1], v[vgprValuC+20:vgprValuC+23]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24:vgprValuC+27], v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+13], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+1], v[vgprValuC+24:vgprValuC+27]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28:vgprValuC+31], v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+15], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+1], v[vgprValuC+28:vgprValuC+31]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32:vgprValuC+35], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+3], v[vgprValuC+32:vgprValuC+35]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36:vgprValuC+39], v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+3], v[vgprValuC+36:vgprValuC+39]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40:vgprValuC+43], v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+5], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+3], v[vgprValuC+40:vgprValuC+43]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44:vgprValuC+47], v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+7], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+3], v[vgprValuC+44:vgprValuC+47]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48:vgprValuC+51], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+9], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+3], v[vgprValuC+48:vgprValuC+51]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52:vgprValuC+55], v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+11], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+3], v[vgprValuC+52:vgprValuC+55]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56:vgprValuC+59], v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+13], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+3], v[vgprValuC+56:vgprValuC+59]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60:vgprValuC+63], v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+15], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+3], v[vgprValuC+60:vgprValuC+63]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64:vgprValuC+67], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+5], v[vgprValuC+64:vgprValuC+67]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68:vgprValuC+71], v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+5], v[vgprValuC+68:vgprValuC+71]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72:vgprValuC+75], v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+5], v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+5], v[vgprValuC+72:vgprValuC+75]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76:vgprValuC+79], v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+7], v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+5], v[vgprValuC+76:vgprValuC+79]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80:vgprValuC+83], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+9], v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+5], v[vgprValuC+80:vgprValuC+83]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84:vgprValuC+87], v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+11], v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+5], v[vgprValuC+84:vgprValuC+87]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88:vgprValuC+91], v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+13], v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+5], v[vgprValuC+88:vgprValuC+91]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92:vgprValuC+95], v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+15], v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+5], v[vgprValuC+92:vgprValuC+95]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96:vgprValuC+99], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+7], v[vgprValuC+96:vgprValuC+99]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100:vgprValuC+103], v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+7], v[vgprValuC+100:vgprValuC+103]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104:vgprValuC+107], v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+5], v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+7], v[vgprValuC+104:vgprValuC+107]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108:vgprValuC+111], v[vgprValuA_X1_I0+6:vgprValuA_X1_I0+7], v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+7], v[vgprValuC+108:vgprValuC+111]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112:vgprValuC+115], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+9], v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+7], v[vgprValuC+112:vgprValuC+115]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116:vgprValuC+119], v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+11], v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+7], v[vgprValuC+116:vgprValuC+119]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120:vgprValuC+123], v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+13], v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+7], v[vgprValuC+120:vgprValuC+123]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124:vgprValuC+127], v[vgprValuA_X1_I0+14:vgprValuA_X1_I0+15], v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+7], v[vgprValuC+124:vgprValuC+127]
s_setprio 0
.endm

.macro FP32_TO_BF16 vgprValue:req
	v_cmp_u_f32 s[98:99], v[\vgprValue], v[\vgprValue]
	v_bfe_u32 v208, v[\vgprValue], 16, 1
	v_add3_u32 v208, v[\vgprValue], v208, v207
	v_cndmask_b32 v[\vgprValue], v208, v206, s[98:99]
	v_lshrrev_b32 v[\vgprValue], 16, v[\vgprValue]
.endm

s_mov_b32 m0, 0x10000
v_mov_b32 v[vgprSerial], v0

s_load_dwordx2 s[sgprAddressB:sgprAddressB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0
s_load_dwordx2 s[sgprAddressA:sgprAddressA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x8
s_load_dwordx2 s[68:69], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
s_load_dwordx2 s[sgprAddressD:sgprAddressD+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x18
s_load_dwordx2 s[sgprBarrierPtrLow:sgprBarrierPtrHigh], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x20
s_load_dwordx2 s[sgprAddressC:sgprAddressC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x28
s_load_dword s[sgprSizeK], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x30
s_load_dword s[sgprStrideD1J], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x34
s_load_dword s[sgprSizeL], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x38
s_load_dword s94, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x3c
s_load_dword s[sgprRank], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40
s_load_dword s95, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x44
s_load_dword s96, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x48
s_load_dword s80, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x4c
s_load_dword s82, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_load_dword s77, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x54
s_load_dwordx2 s[86:87], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60
s_waitcnt lgkmcnt(0)

s_cmp_eq_u32 s94, 0
s_cbranch_scc1 .L_moe_problem_end
s_cmp_eq_u32 s96, 1
s_cbranch_scc0 .L_moe_problem_end

s_mov_b32 s[sgprAlpha], 0x3f800000
s_mov_b32 s[sgprBeta], 0
s_mov_b32 s[sgprStrideDK], s[sgprStrideD1J]
s_mov_b32 s[sgprStrideC1J], s[sgprStrideD1J]
s_mov_b32 s[sgprStrideCK], s[sgprStrideD1J]
s_mov_b32 s[sgprStrideAL], s[sgprStrideD1J]
s_mov_b32 s[sgprStrideAK], s[sgprStrideD1J]
s_mov_b32 s[sgprStrideB1J], s[sgprSizeL]
s_mov_b32 s[sgprStrideBK], s[sgprSizeL]
s_mov_b32 s[sgprSizeI], s[sgprStrideD1J]
s_mov_b32 s[sgprSizeJ], 0
s_mov_b32 s[sgprOrigStaggerUIter], 0
s_mov_b32 s[sgprNumWorkGroups0], s[sgprSizeK]
s_mov_b32 s[sgprNumWorkGroups1], 1
s_mov_b32 s[sgprNumFullBlocks], 0
s_mov_b32 s[sgprWgmRemainder1], 1
s_mov_b32 s[sgprMagicNumberWgmRemainder1], 0x80000001
s_mov_b32 s[sgprOffsetD], 0
s_mov_b32 s[sgprOffsetC], 0
s_mov_b32 s[sgprOffsetA], 0
s_mov_b32 s[sgprOffsetB], 0
s_mov_b32 s[sgprStructBit], 0
s_mov_b32 s[sgprWorldSize], 0

.set OffsetD, UNDEF
.set OffsetC, UNDEF
.set OffsetA, UNDEF
.set OffsetB, UNDEF

label_AlphaNonZero:

v_lshrrev_b32 v[vgprValuA_X0_I0], 6, v[vgprSerial]
v_readfirstlane_b32 s[sgprWaveiD], v[vgprValuA_X0_I0]

v_and_b32 v0, 63, v[vgprSerial]
v_lshrrev_b32 v4, 6, v[vgprSerial]
v_and_b32 v5, 3, v0
v_lshrrev_b32 v3, 2, v0

v_and_b32 v8, 1, v4
s_mov_b32 s62, 32
v_mul_lo_u32 v8, s62, v8

v_and_b32 v1, 7, v0
v_lshrrev_b32 v2, 3, v0

v_and_b32 v6, 3, v1
v_lshrrev_b32 v7, 2, v1

v_lshlrev_b32 v6, 3, v6
v_lshlrev_b32 v10, 9, v7
v_lshlrev_b32 v11, 5, v7

v_and_b32 v12, 15, v0
v_lshrrev_b32 v13, 4, v0
v_lshrrev_b32 v12, 3, v12

v_lshlrev_b32 v12, 8, v12
v_lshlrev_b32 v13, 10, v13

v_add_co_u32 v6, vcc, v11, v6
v_add_co_u32 v6, vcc, v6, v10
v_add_co_u32 v6, vcc, v6, v12
v_add_co_u32 v6, vcc, v6, v13
v_add_lshl_u32 v[vgprLocalReadAddrA], v6, v8, 0x1

v_lshrrev_b32 v7, 2, v0
v_and_b32 v7, 3, v7
v_cmp_eq_u32 s[64:65], v7, 3

v_and_b32 v2, 1, v4
v_cmp_eq_u32 s[62:63], v2, 1

s_and_b64 s[62:63], s[62:63], s[64:65]
v_mov_b32 v7,0x200

v_cndmask_b32 v7, 0, v7, s[62:63]

v_sub_i32 v6, v6, v7
v_add_lshl_u32 v[vgprLocalReadAddrASwap], v6, v8, 0x1

v_and_b32 v5, 15, v0
v_lshrrev_b32 v6, 4, v0

v_and_b32 v7, 3, v0
v_lshrrev_b32 v8, 2, v5

v_lshlrev_b32 v7, 4, v7
v_lshlrev_b32 v9, 9, v8
v_lshlrev_b32 v8, 3, v8

v_lshlrev_b32 v11, 2, v6

v_lshrrev_b32 v12, 1, v4
v_lshlrev_b32 v12, 6, v12

v_add_co_u32 v7, vcc, v7, v8
v_add_co_u32 v7, vcc, v7, v9
v_add_co_u32 v7, vcc, v7, v11

v_sub_co_u32 v[vgprLocalReadAddrB_forLdsWrapTailLoop], vcc, v7, v8
v_add_lshl_u32 v[vgprLocalReadAddrB], v7, v12, 0x1
v_add_lshl_u32 v[vgprLocalReadAddrB_forLdsWrapTailLoop], v[vgprLocalReadAddrB_forLdsWrapTailLoop], v12, 0x1

v_add_co_u32 v[vgprLocalReadAddrBSwap], vcc, 512, v[vgprLocalReadAddrB]
v_and_b32 v6, 15, v[vgprSerial]
v_lshrrev_b32 v6, 2, v6
v_lshlrev_b32 v6, 10, v6
v_add_co_u32 v6, vcc, 1024, v6

v_cmp_ge_u32 s[64:65], v[vgprLocalReadAddrBSwap], v6
v_mov_b32 v8, 1024
v_cndmask_b32 v8, 0, v8, s[64:65]
v_sub_u32 v[vgprLocalReadAddrBSwap], v[vgprLocalReadAddrBSwap], v8

_v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x2000, v[vgprLocalReadAddrB+0]
_v_add_co_u32 v[vgprLocalReadAddrBSwap+0], vcc, 0x2000, v[vgprLocalReadAddrBSwap+0]
_v_add_co_u32 v[vgprLocalReadAddrB_forLdsWrapTailLoop], vcc, 0x2000, v[vgprLocalReadAddrB_forLdsWrapTailLoop]

v_mov_b32 v[vgprLocalReadAddrA_ori], v[vgprLocalReadAddrA]
v_mov_b32 v[vgprLocalReadAddrB_ori], v[vgprLocalReadAddrB]
v_mov_b32 v[vgprLocalReadAddrASwap_ori], v[vgprLocalReadAddrASwap]
v_mov_b32 v[vgprLocalReadAddrBSwap_ori], v[vgprLocalReadAddrBSwap]

s_load_dwordx2 s[68:69], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
s_add_u32 s85, s[sgprStrideD1J], 255
s_lshr_b32 s85, s85, 8
s_cmp_eq_u32 s85, 0
s_cbranch_scc1 .L_moe_problem_end
s_and_b32 s79, s77, 1
s_cmp_eq_u32 s79, 1
s_cbranch_scc1 .L_v36_direct_group_major_decode

s_cmp_ge_u32 s[sgprWorkGroup0], s85
s_cbranch_scc1 .L_moe_problem_end
s_mov_b32 s94, s[sgprWorkGroup0]
s_mov_b32 s100, s94
s_branch .L_v36_direct_launch_decode_done

.L_v36_direct_group_major_decode:
s_cmp_eq_u32 s[sgprSizeK], 0
s_cbranch_scc1 .L_moe_problem_end
s_lshr_b32 s83, s80, 8
s_cmp_eq_u32 s83, 0
s_cbranch_scc1 .L_moe_problem_end
s_cmp_ge_u32 s[sgprWorkGroup0], s83
s_cbranch_scc1 .L_moe_problem_end
s_cmp_eq_u32 s82, 0
s_cbranch_scc1 .L_moe_problem_end

s_mov_b32 s84, s[sgprWorkGroup1]
s_mov_b32 s88, 0
.L_v36_direct_launch_div:
s_cmp_lt_u32 s84, s[sgprSizeK]
s_cbranch_scc1 .L_v36_direct_launch_div_done
s_sub_u32 s84, s84, s[sgprSizeK]
s_add_u32 s88, s88, 1
s_branch .L_v36_direct_launch_div
.L_v36_direct_launch_div_done:
s_cmp_ge_u32 s88, s82
s_cbranch_scc1 .L_moe_problem_end
s_mul_i32 s89, s88, s83
s_add_u32 s94, s89, s[sgprWorkGroup0]
s_cmp_ge_u32 s94, s85
s_cbranch_scc1 .L_moe_problem_end
s_mov_b32 s[sgprWorkGroup1], s84
s_mov_b32 s100, s94

.L_v36_direct_launch_decode_done:
.if FTRAIN_FC2_TILE_SWIZZLE_LOG2 > 0
s_bitcmp1_b32 s77, 0
s_cbranch_scc1 .L_fc2_swizzle_done
s_lshr_b32 s85, s[sgprStrideD1J], 8
s_and_b32 s84, s[sgprWorkGroup1], (1 << FTRAIN_FC2_TILE_SWIZZLE_LOG2) - 1
s_mul_i32 s84, s84, s85
s_add_u32 s84, s84, s[sgprWorkGroup0]
s_and_b32 s88, s[sgprWorkGroup1], -(1 << FTRAIN_FC2_TILE_SWIZZLE_LOG2)
s_and_b32 s89, s84, (1 << FTRAIN_FC2_TILE_SWIZZLE_LOG2) - 1
s_add_u32 s[sgprWorkGroup1], s88, s89
s_lshr_b32 s94, s84, FTRAIN_FC2_TILE_SWIZZLE_LOG2
s_mov_b32 s100, s94
.L_fc2_swizzle_done:
.endif
s_cmp_ge_u32 s[sgprWorkGroup1], s[sgprSizeK]
s_cbranch_scc1 .L_moe_problem_end

s_mov_b32 s97, s[sgprWorkGroup1]
s_mov_b32 s[sgprWorkGroup0], s97
s_mov_b32 s[sgprWorkGroup1], s94

s_mul_i32 s64, s[sgprWorkGroup1], 256

s_lshl_b32 s66, s[sgprWorkGroup0], 2
v_mov_b32 v0, s66
global_load_dword v1, v0, s[sgprAddressC:sgprAddressC+1] glc

s_add_u32 s66, s[sgprSizeK], s[sgprWorkGroup0]
s_lshl_b32 s66, s66, 2
v_mov_b32 v0, s66
global_load_dword v2, v0, s[sgprAddressC:sgprAddressC+1] glc

s_mul_i32 s66, s[sgprSizeK], 2
s_add_u32 s66, s66, s[sgprWorkGroup0]
s_lshl_b32 s66, s66, 2
v_mov_b32 v0, s66
global_load_dword v3, v0, s[sgprAddressC:sgprAddressC+1] glc

s_mul_i32 s66, s[sgprSizeK], 3
s_add_u32 s66, s66, s[sgprWorkGroup0]
s_lshl_b32 s66, s66, 2
v_mov_b32 v0, s66
global_load_dword v4, v0, s[sgprAddressC:sgprAddressC+1] glc

s_mul_i32 s66, s[sgprSizeK], 4
s_add_u32 s66, s66, s[sgprWorkGroup0]
s_lshl_b32 s66, s66, 2
v_mov_b32 v0, s66
global_load_dword v5, v0, s[sgprAddressC:sgprAddressC+1] glc
s_waitcnt vmcnt(0)

v_readfirstlane_b32 s62, v1
v_readfirstlane_b32 s63, v2
v_mov_b32 v202, s63
s_mul_i32 s66, s63, s[sgprSizeL]
s_mul_hi_u32 s67, s63, s[sgprSizeL]
s_lshl_b64 s[66:67], s[66:67], 1
s_add_u32 s[sgprAddressB], s[sgprAddressB], s66
s_addc_u32 s[sgprAddressB+1], s[sgprAddressB+1], s67
v_readfirstlane_b32 s80, v3
v_readfirstlane_b32 s82, v4
v_readfirstlane_b32 s81, v5

s_cmp_le_i32 s81, 0
s_cbranch_scc1 .L_moe_problem_end
s_cmp_ge_u32 s64, s[sgprStrideD1J]
s_cbranch_scc1 .L_moe_problem_end
s_sub_u32 s84, s[sgprStrideD1J], s64
s_mov_b32 s85, 256
s_cmp_lt_u32 s84, s85
s_cselect_b32 s84, s84, s85
s_cmp_eq_u32 s84, 0
s_cbranch_scc1 .L_moe_problem_end

s_waitcnt lgkmcnt(0)
s_lshl_b32 s65, s[sgprWorkGroup0], 8

s_mov_b32 s66, s64
s_mov_b32 s67, 0
s_lshl_b64 s[66:67], s[66:67], 1
s_add_u32 s[sgprAddressD+0], s[sgprAddressD+0], s66
s_addc_u32 s[sgprAddressD+1], s[sgprAddressD+1], s67

s_load_dwordx2 s[sgprAddressC:sgprAddressC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_waitcnt lgkmcnt(0)

s_mul_i32 s66, s62, s[sgprSizeL]
s_mul_hi_u32 s67, s66, s[sgprStrideD1J]
s_mul_i32 s66, s66, s[sgprStrideD1J]
s_add_u32 s66, s66, s64
s_addc_u32 s67, s67, 0
s_lshl_b64 s[66:67], s[66:67], 1
s_add_u32 s[sgprAddressA+0], s[sgprAddressA+0], s66
s_addc_u32 s[sgprAddressA+1], s[sgprAddressA+1], s67

s_mov_b32 s[sgprSizeI], s84
s_mov_b32 s[sgprSizeJ], s81
s_mov_b32 s[sgprSizeK], 1
s_mov_b32 s[sgprWorkGroup0], 0
s_mov_b32 s[sgprWorkGroup1], 0
s_mov_b32 s[sgprWorkGroup2], 0
s_mov_b32 s[sgprNumWorkGroups0], 1
s_mov_b32 s[sgprNumWorkGroups1], 1
s_mov_b32 s[sgprNumFullBlocks], 0
s_mov_b32 s[sgprWgmRemainder1], 1
s_mov_b32 s[sgprMagicNumberWgmRemainder1], 0x80000001

s_mul_hi_u32 s[sgprTensor2dSizeA+1], s[sgprSizeL], s[sgprStrideD1J]
s_mul_i32 s[sgprTensor2dSizeA+0], s[sgprSizeL], s[sgprStrideD1J]
s_sub_u32 s[sgprTensor2dSizeA+0], s[sgprTensor2dSizeA+0], s64
s_subb_u32 s[sgprTensor2dSizeA+1], s[sgprTensor2dSizeA+1], 0
s_mul_hi_u32 s[sgprTensor2dSizeB+1], s81, s[sgprSizeL]
s_mul_i32 s[sgprTensor2dSizeB+0], s81, s[sgprSizeL]

s_barrier
s_branch .L_moe_problem_body
.L_moe_problem_end:
s_endpgm
.L_moe_problem_body:

v_lshrrev_b32 v1, 5, v[vgprSerial]
v_and_b32 v0, 31, v[vgprSerial]

v_lshlrev_b32 v0, 0x3, v0

v_and_b32 v2, 3, v1
v_lshrrev_b32 v3, 2, v1

v_and_b32  v4, 1, v2
v_lshrrev_b32 v5, 1, v2

v_lshlrev_b32 v4, 1, v4
v_lshlrev_b32 v3, 2, v3

v_add_co_u32 v4, vcc, v4, v5
v_add_co_u32 v1, vcc, v4, v3

v_lshlrev_b32 v[vgprSerial_forLdsWrapOptA], 5, v1
v_and_b32 v7, 31, v[vgprSerial]
v_add_co_u32 v[vgprSerial_forLdsWrapOptA], vcc, v[vgprSerial_forLdsWrapOptA], v7

v_and_b32 v3, 1, v[vgprSerial]
v_lshlrev_b32 v3, 3, v3

v_and_b32 v4, 7, v[vgprSerial]
v_lshrrev_b32 v4, 1, v4

v_and_b32 v5, 63, v[vgprSerial]
v_lshrrev_b32 v5, 3, v5
v_lshlrev_b32 v5, 5, v5

v_lshrrev_b32 v6, 6, v[vgprSerial]
v_lshlrev_b32 v6, 2, v6

v_add_co_u32 v2, vcc, v4, v5
v_add_co_u32 v2, vcc, v2, v6

v_lshlrev_b32 v[vgprSerial_forLdsWrapOptB], 1, v2
v_and_b32 v7, 1, v[vgprSerial]
v_add_co_u32 v[vgprSerial_forLdsWrapOptB], vcc, v[vgprSerial_forLdsWrapOptB], v7

v_mov_b32 v4, v0

v_mov_b32 v5, v2

v_mov_b32 v6, v1

v_mov_b32 v7, v3

s_mul_i32 s62, s[sgprWorkGroup0], 256
s_sub_u32 s62, s[sgprSizeI], s62
s_sub_u32 s62, s62, 8
v_mov_b32 v8, s62
_v_add_co_u32 v9, vcc, v8, 8
_v_add_co_u32 v10, vcc, v4, 8
v_cmp_lt_u32 s[62:63], v10, v9
v_cndmask_b32 v4, v8, v4, s[62:63]

GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  4,  6, 8

GLOBAL_OFFSET_B_TailLoop vgprGlobalReadOffsetB1, 7,  5, 8

s_cmp_ge_u32 s[sgprStrideB1J], 256
s_cmov_b32 s[sgprStructBit], 8

s_cmp_ge_u32 s[sgprStrideB1J], 512
s_cmov_b32 s[sgprStructBit], 9

s_cmp_ge_u32 s[sgprStrideB1J], 1024
s_cmov_b32 s[sgprStructBit], 10

s_cmp_ge_u32 s[sgprStrideB1J], 2048
s_cmov_b32 s[sgprStructBit], 11

s_cmp_ge_u32 s[sgprStrideB1J], 4096
s_cmov_b32 s[sgprStructBit], 12

s_lshl_b32 s[sgprStrideSwizzle], 0x1, s[sgprStructBit]

s_cmp_lt_u32 s[sgprStrideB1J], s[sgprStrideSwizzle]
s_cbranch_scc1 Skip_supply_offset1B
s_lshr_b32 s[sgprStrideStruct], s[sgprStrideB1J], s[sgprStructBit]
s_lshl_b32 s[sgprOffsetStruct], s[sgprStrideStruct], s[sgprStructBit]
s_sub_u32 s[sgprOffsetStruct], s[sgprStrideB1J], s[sgprOffsetStruct]
s_branch Skip_supply_offset2B
Skip_supply_offset1B:
s_mov_b32 s[sgprStrideStruct], 1
s_mov_b32 s[sgprOffsetStruct], 0
Skip_supply_offset2B:
GLOBAL_STRUCT_INDEX_B vgprGlobalReadOffsetB+0, 5
GLOBAL_STRUCT_OFFSET_B vgprGlobalReadOffsetB+1, 7, 5

s_mul_hi_u32 s65, s[sgprWorkGroup1], 256
s_mul_i32 s64, s[sgprWorkGroup1], 256
s_mul_hi_u32 s65, s64, s[sgprStrideB1J]
s_mul_i32 s64, s64, s[sgprStrideB1J]
s_sub_u32 s[sgprShadowLimitB_forTailLoop+0], s[sgprTensor2dSizeB], s64
s_subb_u32 s[sgprShadowLimitB_forTailLoop+1], s[sgprTensor2dSizeB+1], s65
s_lshl_b64 s[sgprShadowLimitB_forTailLoop:sgprShadowLimitB_forTailLoop+1], s[sgprShadowLimitB_forTailLoop:sgprShadowLimitB_forTailLoop+1], 0x1
s_add_u32 s[sgprShadowLimitB_forTailLoop+0], s[sgprShadowLimitB_forTailLoop+0], 16
s_addc_u32 s[sgprShadowLimitB_forTailLoop+1], s[sgprShadowLimitB_forTailLoop+1], 0

s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 4
s_mov_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeB
s_mul_i32 s[sgprGlobalReadIncsB+0], s[sgprGlobalReadIncsB+0], s[sgprLoopCounterL]

s_sub_u32 s[sgprShadowLimitB_forTailLoop+0], s[sgprShadowLimitB_forTailLoop+0], s[sgprGlobalReadIncsB+0]
s_subb_u32 s[sgprShadowLimitB_forTailLoop+1], s[sgprShadowLimitB_forTailLoop+1], 0

s_mul_hi_u32 s65, s[sgprWorkGroup0], 256
s_mul_i32 s64, s[sgprWorkGroup0], 256
s_sub_u32 s[sgprShadowLimitA+0], s[sgprTensor2dSizeA], s64
s_subb_u32 s[sgprShadowLimitA+1], s[sgprTensor2dSizeA+1], s65
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x1
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit
s_mul_hi_u32 s63, s[sgprStrideAK], s[sgprWorkGroup2]
s_mul_i32 s62, s[sgprStrideAK], s[sgprWorkGroup2]
s_add_u32 s64, s64, s62
s_addc_u32 s65, s65, s63
s_lshl_b64 s[64:65], s[64:65], 0x1
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s64
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s65
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 16
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0
s_mov_b32 s[sgprSrdA+3], Srd127_96

s_mul_hi_u32 s65, s[sgprWorkGroup1], 256
s_mul_i32 s64, s[sgprWorkGroup1], 256

s_cmp_lt_u32 s[sgprStrideB1J], s[sgprStrideSwizzle]
s_cbranch_scc1 Skip_fix_strideB1
s_lshr_b32 s[sgprStrideStruct], s[sgprStrideB1J], s[sgprStructBit]
s_sub_u32 s[sgprShadowLimitB+0], s[sgprSizeJ], s64
s_subb_u32 s[sgprShadowLimitB+1], 0, s65
s_mul_hi_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+0], s[sgprStrideStruct]
s_mul_i32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprStrideStruct]
s_branch Skip_fix_strideB2
Skip_fix_strideB1:
s_lshr_b32 s[sgprStrideStruct], s[sgprStrideB1J], 0xd
s_sub_u32 s[sgprShadowLimitB+0], s[sgprSizeJ], s64
s_subb_u32 s[sgprShadowLimitB+1], 0, s65
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprStrideStruct]
Skip_fix_strideB2:

s_mul_hi_u32 s65, s64, s[sgprStrideB1J]
s_mul_i32 s64, s64, s[sgprStrideB1J]
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_mov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]

s_mul_hi_u32 s63, s[sgprStrideBK], s[sgprWorkGroup2]
s_mul_i32 s62, s[sgprStrideBK], s[sgprWorkGroup2]
s_add_u32 s64, s64, s62
s_addc_u32 s65, s65, s63
s_lshl_b64 s[64:65], s[64:65], 0x1
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s64
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s65
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 16
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0
s_mov_b32 s[sgprSrdB+3], Srd127_96

s_add_u32 s[sgprStructBit1], 17, s[sgprStructBit]
s_cmp_lt_u32 s[sgprStrideB1J], s[sgprStrideSwizzle]
s_cbranch_scc1 skip_fix_set
s_mov_b32 s[sgprStructNum], 1
s_lshl_b32 s[sgprStructNum], s[sgprStructNum], s[sgprStructBit1]
s_branch skip_fix_set2
skip_fix_set:
s_lshr_b32 s[sgprStrideStruct], s[sgprStrideB1J], 0xd
s_lshl_b32 s[sgprStructNum], s[sgprStrideB1J], 0x1
s_lshr_b32 s[sgprStructNum],s[sgprStructNum],s[sgprStrideStruct]
s_lshl_b32 s[sgprStructNum], s[sgprStructNum], 0x10
skip_fix_set2:
s_or_b32 s[sgprStructNum], s[sgprStructNum], 0x40000000
s_mov_b32 s90, s[sgprStructNum]
s_or_b32 s[sgprSrdB+1], s[sgprSrdB+1], s90

s_mul_i32 s[sgprGlobalReadIncsA+0], DepthU*BpeA, s[sgprStrideAL]

s_mov_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeB

v_lshrrev_b32 v1, 5, v[vgprSerial]
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x100, v1
_v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x1
v_readfirstlane_b32 s[sgprLocalWriteAddrA], v[vgprLocalWriteAddrA]

s_and_b32 s64, s[sgprWaveiD], 1
s_lshl_b32 s64, s64, 18
s_or_b32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA], s64

v_and_b32 v3, 1, v[vgprSerial]
v_lshlrev_b32 v3, 3, v3
v_lshrrev_b32 v2, 1, v[vgprSerial]

v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x10, v2
_v_add_lshl_u32 v[vgprLocalWriteAddrB], v3, v[vgprLocalWriteAddrB], 0x1
_v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x2000, v[vgprLocalWriteAddrB]
v_readfirstlane_b32 s[sgprLocalWriteAddrB], v[vgprLocalWriteAddrB]

s_and_b32 s64, s[sgprWaveiD], 3
s_lshl_b32 s64, s64, 16
s_or_b32 s[sgprLocalWriteAddrB], s[sgprLocalWriteAddrB], s64

s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 4
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL]

s_cmp_eq_u32 s[sgprLoopCounterL], 0
s_cbranch_scc1 ShadowInitStart_9

s_barrier
s_mov_b32 m0, s[sgprLocalWriteAddrA]
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds
s_mov_b32 m0, s[sgprLocalWriteAddrB]
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds

s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0]
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0]
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0]
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_mov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]

s_cmp_eq_i32 s[sgprLoopCounterL], 0x1
s_cbranch_scc1 ShadowInitStart_9

s_barrier
s_add_u32 m0, s[sgprLocalWriteAddrA],0x4000
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds
s_add_u32 m0, s[sgprLocalWriteAddrB],0x4000
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds

s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0]
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0]
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0]
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_mov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]

s_cmp_eq_i32 s[sgprLoopCounterL], 0x2
s_cbranch_scc1 ShadowInitStart_9

s_barrier
s_add_u32 m0, s[sgprLocalWriteAddrA],0x8000
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds
s_add_u32 m0, s[sgprLocalWriteAddrB],0x8000
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds

s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0]
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0]
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0]
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_mov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]

s_cmp_eq_i32 s[sgprLoopCounterL], 0x3
s_cbranch_scc1 ShadowInitStart_9

s_barrier
s_add_u32 m0, s[sgprLocalWriteAddrA],0xc000
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds
s_add_u32 m0, s[sgprLocalWriteAddrB],0xc000
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds

s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0]
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0]
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0]
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_mov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]

ShadowInitStart_9:

s_mov_b32 s[sgprSrdD+0], s[sgprAddressD+0]
s_mov_b32 s[sgprSrdD+1], s[sgprAddressD+1]
s_mov_b32 s[sgprSrdD+2], 0xfffffffe
s_mov_b32 s[sgprSrdD+3], Srd127_96

s_mov_b32 s[sgprSrdC+0], s[sgprAddressC+0]
s_mov_b32 s[sgprSrdC+1], s[sgprAddressC+1]
s_mov_b32 s[sgprSrdC+2], 0xfffffffe
s_mov_b32 s[sgprSrdC+3], Srd127_96

s_mul_i32 s64, MT1, s[sgprWorkGroup1]
s_mul_hi_u32 s63, s64, s[sgprStrideC1J]
s_mul_i32 s62, s64, s[sgprStrideC1J]
s_lshl_b64 s[62:63], s[62:63], 1
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s62
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s63
s_mul_hi_u32 s63, s64, s[sgprStrideD1J]
s_mul_i32 s62, s64, s[sgprStrideD1J]
s_lshl_b64 s[62:63], s[62:63], 1
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s62
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s63

s_mul_hi_u32 s63, s[sgprWorkGroup2], s[sgprStrideCK]
s_mul_i32 s62, s[sgprWorkGroup2], s[sgprStrideCK]
s_lshl_b64 s[62:63], s[62:63], 1
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s62
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s63
s_mul_hi_u32 s63, s[sgprWorkGroup2], s[sgprStrideDK]
s_mul_i32 s62, s[sgprWorkGroup2], s[sgprStrideDK]
s_lshl_b64 s[62:63], s[62:63], 1
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s62
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s63

v_mov_b32 v[vgprValuC+0], 0x0
v_mov_b32 v[vgprValuC+1], 0x0
v_mov_b32 v[vgprValuC+2], 0x0
v_mov_b32 v[vgprValuC+3], 0x0
v_mov_b32 v[vgprValuC+4], 0x0
v_mov_b32 v[vgprValuC+5], 0x0
v_mov_b32 v[vgprValuC+6], 0x0
v_mov_b32 v[vgprValuC+7], 0x0
v_mov_b32 v[vgprValuC+8], 0x0
v_mov_b32 v[vgprValuC+9], 0x0
v_mov_b32 v[vgprValuC+10], 0x0
v_mov_b32 v[vgprValuC+11], 0x0
v_mov_b32 v[vgprValuC+12], 0x0
v_mov_b32 v[vgprValuC+13], 0x0
v_mov_b32 v[vgprValuC+14], 0x0
v_mov_b32 v[vgprValuC+15], 0x0
v_mov_b32 v[vgprValuC+16], 0x0
v_mov_b32 v[vgprValuC+17], 0x0
v_mov_b32 v[vgprValuC+18], 0x0
v_mov_b32 v[vgprValuC+19], 0x0
v_mov_b32 v[vgprValuC+20], 0x0
v_mov_b32 v[vgprValuC+21], 0x0
v_mov_b32 v[vgprValuC+22], 0x0
v_mov_b32 v[vgprValuC+23], 0x0
v_mov_b32 v[vgprValuC+24], 0x0
v_mov_b32 v[vgprValuC+25], 0x0
v_mov_b32 v[vgprValuC+26], 0x0
v_mov_b32 v[vgprValuC+27], 0x0
v_mov_b32 v[vgprValuC+28], 0x0
v_mov_b32 v[vgprValuC+29], 0x0
v_mov_b32 v[vgprValuC+30], 0x0
v_mov_b32 v[vgprValuC+31], 0x0
v_mov_b32 v[vgprValuC+32], 0x0
v_mov_b32 v[vgprValuC+33], 0x0
v_mov_b32 v[vgprValuC+34], 0x0
v_mov_b32 v[vgprValuC+35], 0x0
v_mov_b32 v[vgprValuC+36], 0x0
v_mov_b32 v[vgprValuC+37], 0x0
v_mov_b32 v[vgprValuC+38], 0x0
v_mov_b32 v[vgprValuC+39], 0x0
v_mov_b32 v[vgprValuC+40], 0x0
v_mov_b32 v[vgprValuC+41], 0x0
v_mov_b32 v[vgprValuC+42], 0x0
v_mov_b32 v[vgprValuC+43], 0x0
v_mov_b32 v[vgprValuC+44], 0x0
v_mov_b32 v[vgprValuC+45], 0x0
v_mov_b32 v[vgprValuC+46], 0x0
v_mov_b32 v[vgprValuC+47], 0x0
v_mov_b32 v[vgprValuC+48], 0x0
v_mov_b32 v[vgprValuC+49], 0x0
v_mov_b32 v[vgprValuC+50], 0x0
v_mov_b32 v[vgprValuC+51], 0x0
v_mov_b32 v[vgprValuC+52], 0x0
v_mov_b32 v[vgprValuC+53], 0x0
v_mov_b32 v[vgprValuC+54], 0x0
v_mov_b32 v[vgprValuC+55], 0x0
v_mov_b32 v[vgprValuC+56], 0x0
v_mov_b32 v[vgprValuC+57], 0x0
v_mov_b32 v[vgprValuC+58], 0x0
v_mov_b32 v[vgprValuC+59], 0x0
v_mov_b32 v[vgprValuC+60], 0x0
v_mov_b32 v[vgprValuC+61], 0x0
v_mov_b32 v[vgprValuC+62], 0x0
v_mov_b32 v[vgprValuC+63], 0x0
v_mov_b32 v[vgprValuC+64], 0x0
v_mov_b32 v[vgprValuC+65], 0x0
v_mov_b32 v[vgprValuC+66], 0x0
v_mov_b32 v[vgprValuC+67], 0x0
v_mov_b32 v[vgprValuC+68], 0x0
v_mov_b32 v[vgprValuC+69], 0x0
v_mov_b32 v[vgprValuC+70], 0x0
v_mov_b32 v[vgprValuC+71], 0x0
v_mov_b32 v[vgprValuC+72], 0x0
v_mov_b32 v[vgprValuC+73], 0x0
v_mov_b32 v[vgprValuC+74], 0x0
v_mov_b32 v[vgprValuC+75], 0x0
v_mov_b32 v[vgprValuC+76], 0x0
v_mov_b32 v[vgprValuC+77], 0x0
v_mov_b32 v[vgprValuC+78], 0x0
v_mov_b32 v[vgprValuC+79], 0x0
v_mov_b32 v[vgprValuC+80], 0x0
v_mov_b32 v[vgprValuC+81], 0x0
v_mov_b32 v[vgprValuC+82], 0x0
v_mov_b32 v[vgprValuC+83], 0x0
v_mov_b32 v[vgprValuC+84], 0x0
v_mov_b32 v[vgprValuC+85], 0x0
v_mov_b32 v[vgprValuC+86], 0x0
v_mov_b32 v[vgprValuC+87], 0x0
v_mov_b32 v[vgprValuC+88], 0x0
v_mov_b32 v[vgprValuC+89], 0x0
v_mov_b32 v[vgprValuC+90], 0x0
v_mov_b32 v[vgprValuC+91], 0x0
v_mov_b32 v[vgprValuC+92], 0x0
v_mov_b32 v[vgprValuC+93], 0x0
v_mov_b32 v[vgprValuC+94], 0x0
v_mov_b32 v[vgprValuC+95], 0x0
v_mov_b32 v[vgprValuC+96], 0x0
v_mov_b32 v[vgprValuC+97], 0x0
v_mov_b32 v[vgprValuC+98], 0x0
v_mov_b32 v[vgprValuC+99], 0x0
v_mov_b32 v[vgprValuC+100], 0x0
v_mov_b32 v[vgprValuC+101], 0x0
v_mov_b32 v[vgprValuC+102], 0x0
v_mov_b32 v[vgprValuC+103], 0x0
v_mov_b32 v[vgprValuC+104], 0x0
v_mov_b32 v[vgprValuC+105], 0x0
v_mov_b32 v[vgprValuC+106], 0x0
v_mov_b32 v[vgprValuC+107], 0x0
v_mov_b32 v[vgprValuC+108], 0x0
v_mov_b32 v[vgprValuC+109], 0x0
v_mov_b32 v[vgprValuC+110], 0x0
v_mov_b32 v[vgprValuC+111], 0x0
v_mov_b32 v[vgprValuC+112], 0x0
v_mov_b32 v[vgprValuC+113], 0x0
v_mov_b32 v[vgprValuC+114], 0x0
v_mov_b32 v[vgprValuC+115], 0x0
v_mov_b32 v[vgprValuC+116], 0x0
v_mov_b32 v[vgprValuC+117], 0x0
v_mov_b32 v[vgprValuC+118], 0x0
v_mov_b32 v[vgprValuC+119], 0x0
v_mov_b32 v[vgprValuC+120], 0x0
v_mov_b32 v[vgprValuC+121], 0x0
v_mov_b32 v[vgprValuC+122], 0x0
v_mov_b32 v[vgprValuC+123], 0x0
v_mov_b32 v[vgprValuC+124], 0x0
v_mov_b32 v[vgprValuC+125], 0x0
v_mov_b32 v[vgprValuC+126], 0x0
v_mov_b32 v[vgprValuC+127], 0x0

s_cmp_eq_u32 s[sgprLoopCounterL], 0

s_cbranch_scc0 label_NoBranch_10
s_getpc_B64 s[62:63]
s_add_u32 s64, PrefetchGlobalLastIterEnd_4, 0x4
s_add_u32 s62, s62, s64
s_addc_u32 s63, 0, s63
s_setpc_b64 s[62:63]
label_NoBranch_10:

s_mov_b32 s[sgprMask], 0x10000

s_cmp_gt_u32 s[sgprLoopCounterL], 3
s_cbranch_scc1 prefetchwait
s_waitcnt vmcnt(0)
prefetchwait:
s_waitcnt vmcnt(4)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x80
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x1000
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x00
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x1000
s_waitcnt lgkmcnt(0)
s_barrier

openLoopL_11:

s_cmp_le_u32 s[sgprLoopCounterL], 0x4
s_cbranch_scc1 LoopEndL_2

s_cmp_ge_i32 s[sgprWaveiD], 4
s_cbranch_scc1 LoopBeginL_2

LoopBeginL_1:

label_0012:

ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0x4000
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0x4080
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0x4100
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x4180

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0x4000
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0x5000
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x4000
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x5000

s_add_u32 m0, s[sgprLocalWriteAddrA], 0x000
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x000
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds

s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0]
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0]
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0]
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_mov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]

s_waitcnt lgkmcnt(8)

s_waitcnt vmcnt(4)
s_barrier

MAC_16x8_X0

s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4
s_cbranch_scc1 LoopEndL_oddexit_31

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x8000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x8080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x8100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x8180

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x8000
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x9000
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x8000
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x9000

s_add_u32 m0, s[sgprLocalWriteAddrA], 0x4000
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x4000
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds

s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0]
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0]
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0]
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_mov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]

s_waitcnt lgkmcnt(8)
s_waitcnt vmcnt(4)
s_barrier

MAC_16x8_X1

s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4
s_cbranch_scc1 LoopEndL_oddexit_32

ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0xc000
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0xc080
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0xc100
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0xc180

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0xc000
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0xd000
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0xc000
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0xd000

s_add_u32 m0, s[sgprLocalWriteAddrA], 0x8000
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x8000
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds

s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0]
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0]
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0]
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_mov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]

s_waitcnt lgkmcnt(8)
s_waitcnt vmcnt(4)
s_barrier

MAC_16x8_X0

s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4
s_cbranch_scc1 LoopEndL_oddexit_33

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x80
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x00
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x1000
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x00
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x1000

s_add_u32 m0, s[sgprLocalWriteAddrA], 0xc000
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds
s_add_u32 m0, s[sgprLocalWriteAddrB], 0xc000
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds

s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0]
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0]
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0]
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_mov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]

s_waitcnt lgkmcnt(8)
s_waitcnt vmcnt(4)
s_barrier

MAC_16x8_X1

s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4
s_cbranch_scc0 LoopBeginL_1
s_branch LoopEndL_oddexit_34

LoopBeginL_2:

label_0012_1:

ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0x4000
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0x4080
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0x4100
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x4180

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0x4000
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0x5000
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x4000
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x5000

s_add_u32 m0, s[sgprLocalWriteAddrA], 0x000
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x000
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds

s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0]
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0]
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0]
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_mov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt vmcnt(4)
s_barrier

s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4
s_cbranch_scc1 LoopEndL_oddexit_31

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x8000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x8080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x8100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x8180

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x8000
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x9000
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x8000
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x9000

s_add_u32 m0, s[sgprLocalWriteAddrA], 0x4000
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x4000
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds

s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0]
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0]
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0]
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_mov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt vmcnt(4)
s_barrier

s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4
s_cbranch_scc1 LoopEndL_oddexit_32

ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0xc000
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0xc080
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0xc100
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0xc180

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0xc000
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0xd000
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0xc000
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0xd000

s_add_u32 m0, s[sgprLocalWriteAddrA], 0x8000
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x8000
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds

s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0]
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0]
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0]
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_mov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt vmcnt(4)
s_barrier

s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4
s_cbranch_scc1 LoopEndL_oddexit_33

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x00
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x80
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x00
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x1000
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x00
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x1000

s_add_u32 m0, s[sgprLocalWriteAddrA], 0xc000
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds
s_add_u32 m0, s[sgprLocalWriteAddrB], 0xc000
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds

s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0]
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0]
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0]
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_mov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt vmcnt(4)
s_barrier

s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4
s_cbranch_scc0 LoopBeginL_2
s_branch LoopEndL_oddexit_34

LoopEndL_oddexit_31:

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x8000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x8080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x8100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x8180

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x8000
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x9000
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x8000
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x9000

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt vmcnt(2)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0xc000
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0xc080
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0xc100
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0xc180

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0xc000
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0xd000
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0xc000
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0xd000

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt vmcnt(0)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x00
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x80
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x00
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x1000
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x00
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x1000

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt lgkmcnt(0)
MAC_16x8_X0

s_branch end_store

LoopEndL_oddexit_32:

ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0xc000
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0xc080
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0xc100
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0xc180

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0xc000
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0xd000
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0xc000
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0xd000

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt vmcnt(2)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x00
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x80
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x00
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x1000
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x00
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x1000

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt vmcnt(0)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0x4000
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0x4080
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0x4100
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x4180

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0x4000
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0x5000
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x4000
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x5000

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt lgkmcnt(0)
MAC_16x8_X1

s_branch end_store

LoopEndL_oddexit_33:

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x00
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x80
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x00
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x1000
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x00
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x1000

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt vmcnt(2)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0x4000
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0x4080
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0x4100
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x4180

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0x4000
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0x5000
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x4000
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x5000

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt vmcnt(0)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x8000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x8080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x8100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x8180

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x8000
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x9000
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x8000
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x9000

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt lgkmcnt(0)
MAC_16x8_X0

s_branch end_store

LoopEndL_oddexit_34:
ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0x4000
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0x4080
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0x4100
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x4180

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0x4000
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0x5000
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x4000
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x5000

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt vmcnt(2)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x8000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x8080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x8100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x8180

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x8000
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x9000
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x8000
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x9000

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt vmcnt(0)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0xc000
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0xc080
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0xc100
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0xc180

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0xc000
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0xd000
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0xc000
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0xd000

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt lgkmcnt(0)
MAC_16x8_X1

s_branch end_store

LoopEndL_oddexit_3:
LoopEndL_2:

s_waitcnt vmcnt(0)
s_barrier
s_mov_b32 s64, 0

last_loop:

s_waitcnt lgkmcnt(0)
MAC_16x8_X0

v_add_co_u32 v[vgprLocalReadAddrASwap], 0x4000, v[vgprLocalReadAddrASwap]
v_cmp_lt_u32 s[62:63], v[vgprLocalReadAddrASwap], s[sgprMask]
v_cndmask_b32 v[vgprLocalReadAddrASwap], v[vgprLocalReadAddrASwap_ori], v[vgprLocalReadAddrASwap], s[62:63]

v_add_co_u32 v[vgprLocalReadAddrA], 0x4000, v[vgprLocalReadAddrA]
v_cmp_lt_u32 s[62:63], v[vgprLocalReadAddrA], s[sgprMask]
v_cndmask_b32 v[vgprLocalReadAddrA], v[vgprLocalReadAddrA_ori], v[vgprLocalReadAddrA], s[62:63]

v_add_co_u32 v[vgprLocalReadAddrB], 0x4000, v[vgprLocalReadAddrB]
v_cmp_lt_u32 s[62:63], v[vgprLocalReadAddrB], s[sgprMask]
v_cndmask_b32 v[vgprLocalReadAddrB], v[vgprLocalReadAddrB_ori], v[vgprLocalReadAddrB], s[62:63]

v_add_co_u32 v[vgprLocalReadAddrBSwap], 0x4000, v[vgprLocalReadAddrBSwap]
v_cmp_lt_u32 s[62:63], v[vgprLocalReadAddrBSwap], s[sgprMask]
v_cndmask_b32 v[vgprLocalReadAddrBSwap], v[vgprLocalReadAddrBSwap_ori], v[vgprLocalReadAddrBSwap], s[62:63]

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x000
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x1000
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x00
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x1000

s_add_u32 s64, s64, 1
s_cmp_lt_u32 s64, s[sgprLoopCounterL]
s_cbranch_scc1 last_loop

end_store:

s_cmpk_eq_u32 s[sgprBeta], 0x0
s_cbranch_scc0 OptNLL_End_14

s_cmp_eq_u32 s[sgprAlpha], 1.0
s_cbranch_scc0 OptNLL_End_14

s_and_b32 s62, 255, s[sgprSizeI]
s_add_u32 s64, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s64
s_cselect_b32 s62, s62, 0
s_cmpk_gt_u32 s62, 0x0
s_cbranch_scc1 OptNLL_End_14
s_and_b32 s62, 255, s[sgprSizeJ]
s_add_u32 s64, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s64
s_cselect_b32 s62, s62, 0
s_cmpk_gt_u32 s62, 0x0
s_cbranch_scc1 OptNLL_End_14

s_and_b32 s63, 15, s[sgprSizesSum+0]
s_cmp_eq_u32 s63, 0x0
s_cbranch_scc0 OptNLL_End_14

Summation_End_OptNLL_15:

.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
.set ShadowLimitA, UNDEF
.set ShadowLimitB, UNDEF
.set LocalWriteAddrA, UNDEF
.set LocalWriteAddrB, UNDEF

v_and_b32 v128, 63, v[vgprSerial]
v_lshrrev_b32 v129, 6, v[vgprSerial]
v_and_b32 v132, 15, v128
v_lshrrev_b32 v133, 4, v128
v_and_b32 v134, 1, v129
s_mov_b32 s54, 32
v_mul_lo_u32 v134, s54, v134
v_lshlrev_b32 v128, 0, v132
_v_add_co_u32 v128, vcc, v128, v134
v_lshrrev_b32 v134, 1, v129
s_mov_b32 s54, 32
v_mul_lo_u32 v134, s54, v134
v_lshlrev_b32 v129, 0, v133
_v_add_co_u32 v129, vcc, v129, v134
s_mul_i32 s[sgprStrideoffC+0], 8, s[sgprStridesC]
s_mul_i32 s[sgprStrideoffC+1], 16, s[sgprStridesC]
s_mul_i32 s[sgprStrideoffC+2], 24, s[sgprStridesC]
s_mul_i32 s[sgprStrideoffD+0], 8, s[sgprStridesD]
s_mul_i32 s[sgprStrideoffD+1], 16, s[sgprStridesD]
s_mul_i32 s[sgprStrideoffD+2], 24, s[sgprStridesD]
v_mul_lo_u32 v130, v129, s[sgprStrideC1J]
v_mul_lo_u32 v131, v129, s[sgprStrideD1J]

s_mul_i32 s54, 0x100, s[sgprWorkGroup0]
_v_add_co_u32 v128, vcc, s54, v128
s_mul_i32 s56, 0x100, s[sgprWorkGroup1]
_v_add_co_u32 v129, vcc, s56, v129

s_branch GW_B0_DIRECT_101

OptNLL_End_14:

PrefetchGlobalLastIterEnd_4:

s_cmp_eq_u32 s[sgprShadowLimitB_forTailLoop+1], 0
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB_forTailLoop+0]

s_and_b32 s[sgprLoopCounterL], 15, s[sgprSizesSum+0]
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0
s_mov_b32 s[sgprOrigLoopCounter], 0
s_cbranch_scc1 SkipTailLoopL_7
s_sub_i32 s66, s[sgprLoopCounterL], 0
s_cmp_gt_u32 s66, 16
s_cselect_b32 s66, 16, s66
s_mul_i32 s66, 32, s66
v_cmp_lt_u32 s[68:69], v[vgprSerial_forLdsWrapOptA], s66
v_cndmask_b32 v[vgprGlobalReadOffsetA+0], -0x1, v[vgprGlobalReadOffsetA+0], s[68:69]

v_mov_b32 v[vgprGlobalReadOffsetB+0], v[vgprGlobalReadOffsetB1]

v_and_b32 v[vgprG2LB+0], v[vgprSerial], 1
v_lshlrev_b32 v[vgprG2LB+0], 3, v[vgprG2LB+0]
v_add_co_u32 v[vgprG2LB+0], vcc, 0, v[vgprG2LB+0]
v_cmp_lt_u32 s[68:69], v[vgprG2LB+0], s[sgprLoopCounterL]
v_add_co_u32 v[vgprG2LB+0], vcc, 0, v[vgprGlobalReadOffsetB+0]
v_cndmask_b32 v[vgprG2LB+0], -0x1, v[vgprG2LB+0], s[68:69]
v_and_b32 v173, v[vgprSerial], 1
v_lshlrev_b32 v173, 3, v173
v_add_co_u32 v173, vcc, 1, v173
v_cmp_lt_u32 s[68:69], v173, s[sgprLoopCounterL]
v_add_co_u32 v173, vcc, 2, v[vgprGlobalReadOffsetB+0]
v_cndmask_b32 v173, -0x1, v173, s[68:69]
v_and_b32 v[vgprG2LB+1], v[vgprSerial], 1
v_lshlrev_b32 v[vgprG2LB+1], 3, v[vgprG2LB+1]
v_add_co_u32 v[vgprG2LB+1], vcc, 2, v[vgprG2LB+1]
v_cmp_lt_u32 s[68:69], v[vgprG2LB+1], s[sgprLoopCounterL]
v_add_co_u32 v[vgprG2LB+1], vcc, 4, v[vgprGlobalReadOffsetB+0]
v_cndmask_b32 v[vgprG2LB+1], -0x1, v[vgprG2LB+1], s[68:69]
v_and_b32 v174, v[vgprSerial], 1
v_lshlrev_b32 v174, 3, v174
v_add_co_u32 v174, vcc, 3, v174
v_cmp_lt_u32 s[68:69], v174, s[sgprLoopCounterL]
v_add_co_u32 v174, vcc, 6, v[vgprGlobalReadOffsetB+0]
v_cndmask_b32 v174, -0x1, v174, s[68:69]
v_and_b32 v[vgprG2LB+2], v[vgprSerial], 1
v_lshlrev_b32 v[vgprG2LB+2], 3, v[vgprG2LB+2]
v_add_co_u32 v[vgprG2LB+2], vcc, 4, v[vgprG2LB+2]
v_cmp_lt_u32 s[68:69], v[vgprG2LB+2], s[sgprLoopCounterL]
v_add_co_u32 v[vgprG2LB+2], vcc, 8, v[vgprGlobalReadOffsetB+0]
v_cndmask_b32 v[vgprG2LB+2], -0x1, v[vgprG2LB+2], s[68:69]
v_and_b32 v175, v[vgprSerial], 1
v_lshlrev_b32 v175, 3, v175
v_add_co_u32 v175, vcc, 5, v175
v_cmp_lt_u32 s[68:69], v175, s[sgprLoopCounterL]
v_add_co_u32 v175, vcc, 10, v[vgprGlobalReadOffsetB+0]
v_cndmask_b32 v175, -0x1, v175, s[68:69]
v_and_b32 v[vgprG2LB+3], v[vgprSerial], 1
v_lshlrev_b32 v[vgprG2LB+3], 3, v[vgprG2LB+3]
v_add_co_u32 v[vgprG2LB+3], vcc, 6, v[vgprG2LB+3]
v_cmp_lt_u32 s[68:69], v[vgprG2LB+3], s[sgprLoopCounterL]
v_add_co_u32 v[vgprG2LB+3], vcc, 12, v[vgprGlobalReadOffsetB+0]
v_cndmask_b32 v[vgprG2LB+3], -0x1, v[vgprG2LB+3], s[68:69]
v_and_b32 v176, v[vgprSerial], 1
v_lshlrev_b32 v176, 3, v176
v_add_co_u32 v176, vcc, 7, v176
v_cmp_lt_u32 s[68:69], v176, s[sgprLoopCounterL]
v_add_co_u32 v176, vcc, 14, v[vgprGlobalReadOffsetB+0]
v_cndmask_b32 v176, -0x1, v176, s[68:69]
s_add_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 15
s_lshr_b32 s[sgprLoopCounterL], s[sgprLoopCounterL], 4

s_add_u32 m0, s[sgprLocalWriteAddrA], 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 lds

buffer_load_short_d16 v[vgprG2LB+0+0], v[vgprG2LB+0+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0

buffer_load_short_d16_hi v173, v173, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v173

buffer_load_short_d16 v[vgprG2LB+0+1], v[vgprG2LB+0+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0

buffer_load_short_d16_hi v174, v174, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v174

buffer_load_short_d16 v[vgprG2LB+0+2], v[vgprG2LB+0+2], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0

buffer_load_short_d16_hi v175, v175, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+2], v[vgprG2LB+0+2], v175

buffer_load_short_d16 v[vgprG2LB+0+3], v[vgprG2LB+0+3], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0

buffer_load_short_d16_hi v176, v176, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+3], v[vgprG2LB+0+3], v176

s_waitcnt vmcnt(0)

s_barrier

ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0

s_waitcnt lgkmcnt(0)

s_barrier

v_mov_b32 v[vgprLocalReadAddrA], v[vgprLocalReadAddrA_ori]
v_mov_b32 v[vgprLocalReadAddrASwap], v[vgprLocalReadAddrASwap_ori]

v_mov_b32 v[vgprLocalReadAddrB], v[vgprLocalReadAddrB_ori]
v_mov_b32 v[vgprLocalReadAddrBSwap], v[vgprLocalReadAddrBSwap_ori]

TailLoopBeginL_5:

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x80
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB_forLdsWrapTailLoop] offset:0
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB_forLdsWrapTailLoop] offset:0x1000
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB_forLdsWrapTailLoop] offset:0x200
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB_forLdsWrapTailLoop] offset:0x1200

s_mov_b32 s62, 0x2000
_v_add_co_u32 v[vgprLocalReadAddrA], vcc, s62, v[vgprLocalReadAddrA]
_v_add_co_u32 v[vgprLocalReadAddrASwap], vcc, s62, v[vgprLocalReadAddrASwap]

s_mov_b32 s62, 0x20
_v_add_co_u32 v[vgprLocalReadAddrB], vcc, s62, v[vgprLocalReadAddrB]
_v_add_co_u32 v[vgprLocalReadAddrBSwap], vcc, s62, v[vgprLocalReadAddrBSwap]

s_waitcnt lgkmcnt(0)

.align32 8, 0xbf800001
MAC_16x8_X0

s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x1
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x1
s_cmp_le_i32 s[sgprLoopCounterL], 0x0
s_cbranch_scc0 TailLoopBeginL_5
TailLoopEndL_6:

SkipTailLoopL_7:

Summation_End_23:

.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
.set ShadowLimitA, UNDEF
.set ShadowLimitB, UNDEF
.set LocalWriteAddrA, UNDEF
.set LocalWriteAddrB, UNDEF

v_mov_b32 v130, s[sgprWorkGroup0]
v_mul_i32_i24 v130, -0x100, v130
_v_add_co_u32 v130, vcc, s[sgprSizesFree+0], v130
v_mov_b32 v128, 0x100
v_cmp_lt_u32 s[54:55], v130, v128
v_cndmask_b32 v130, v128, v130, s[54:55]
v_lshrrev_b32 v132, 7, v130
v_lshrrev_b32 v132, 5, v130
v_and_b32 v132, 1, v132
v_and_b32 v133, 127, v130
v_and_b32 v133, 255, v130
v_lshrrev_b32 v133, 0, v133
v_lshrrev_b32 v134, 4, v133
v_and_b32 v135, 15, v133
v_mul_i32_i24 v137, 0x10, v132
v_and_b32 v133, 7, v130
_v_add_co_u32 v145, vcc, v135, v137
_v_sub_co_u32 v145, vcc, v145, v133
v_lshrrev_b32 v134, 6, v[vgprSerial]
v_and_b32 v135, 1, v134
v_and_b32 v137, 15, v[vgprSerial]
v_mul_i32_i24 v135, 0x10, v135
_v_add_co_u32 v147, vcc, v135, v137
v_mov_b32 v135, v130
v_and_b32 v130, 127, v130
v_lshrrev_b32 v134, 4, v130
v_mov_b32 v130, v135
v_and_b32 v130, 255, v130
v_lshrrev_b32 v134, 6, v130
v_lshlrev_b32 v134, 1, v134
v_and_b32 v135, 31, v130
v_lshrrev_b32 v135, 4, v135
_v_add_co_u32 v134, vcc, v134, v135
v_and_b32 v135, 1, v130
v_mov_b32 v136, v135
v_lshrrev_b32 v135, 1, v136
_v_add_co_u32 v135, vcc, v134, v135
v_cmp_eq_u32 vcc, v133, 0x1
s_cbranch_vccnz label_0024
v_cmp_eq_u32 vcc, v133, 0x2
s_cbranch_vccnz label_0033
v_cmp_eq_u32 vcc, v133, 0x3
s_cbranch_vccnz label_0042
v_cmp_eq_u32 vcc, v133, 0x4
s_cbranch_vccnz label_0051
v_cmp_eq_u32 vcc, v133, 0x5
s_cbranch_vccnz label_0060
v_cmp_eq_u32 vcc, v133, 0x6
s_cbranch_vccnz label_0069
v_cmp_eq_u32 vcc, v133, 0x7
s_cbranch_vccnz label_0078
s_branch label_0087

label_0024:
v_cmp_eq_u32 vcc, v135, 0x0
s_cbranch_vccnz label_0025
v_cmp_eq_u32 vcc, v135, 0x1
s_cbranch_vccnz label_0026
v_cmp_eq_u32 vcc, v135, 0x2
s_cbranch_vccnz label_0027
v_cmp_eq_u32 vcc, v135, 0x3
s_cbranch_vccnz label_0028
v_cmp_eq_u32 vcc, v135, 0x4
s_cbranch_vccnz label_0029
v_cmp_eq_u32 vcc, v135, 0x5
s_cbranch_vccnz label_0030
v_cmp_eq_u32 vcc, v135, 0x6
s_cbranch_vccnz label_0031
v_cmp_eq_u32 vcc, v135, 0x7
s_cbranch_vccnz label_0032

label_0025:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v0, v134, v0, offset:28
ds_bpermute_b32 v1, v134, v1, offset:28
ds_bpermute_b32 v2, v134, v2, offset:28
ds_bpermute_b32 v3, v134, v3, offset:28
ds_bpermute_b32 v32, v134, v32, offset:28
ds_bpermute_b32 v33, v134, v33, offset:28
ds_bpermute_b32 v34, v134, v34, offset:28
ds_bpermute_b32 v35, v134, v35, offset:28
ds_bpermute_b32 v64, v134, v64, offset:28
ds_bpermute_b32 v65, v134, v65, offset:28
ds_bpermute_b32 v66, v134, v66, offset:28
ds_bpermute_b32 v67, v134, v67, offset:28
ds_bpermute_b32 v96, v134, v96, offset:28
ds_bpermute_b32 v97, v134, v97, offset:28
ds_bpermute_b32 v98, v134, v98, offset:28
ds_bpermute_b32 v99, v134, v99, offset:28
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0026:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v4, v134, v4, offset:28
ds_bpermute_b32 v5, v134, v5, offset:28
ds_bpermute_b32 v6, v134, v6, offset:28
ds_bpermute_b32 v7, v134, v7, offset:28
ds_bpermute_b32 v36, v134, v36, offset:28
ds_bpermute_b32 v37, v134, v37, offset:28
ds_bpermute_b32 v38, v134, v38, offset:28
ds_bpermute_b32 v39, v134, v39, offset:28
ds_bpermute_b32 v68, v134, v68, offset:28
ds_bpermute_b32 v69, v134, v69, offset:28
ds_bpermute_b32 v70, v134, v70, offset:28
ds_bpermute_b32 v71, v134, v71, offset:28
ds_bpermute_b32 v100, v134, v100, offset:28
ds_bpermute_b32 v101, v134, v101, offset:28
ds_bpermute_b32 v102, v134, v102, offset:28
ds_bpermute_b32 v103, v134, v103, offset:28
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0027:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v8, v134, v8, offset:28
ds_bpermute_b32 v9, v134, v9, offset:28
ds_bpermute_b32 v10, v134, v10, offset:28
ds_bpermute_b32 v11, v134, v11, offset:28
ds_bpermute_b32 v40, v134, v40, offset:28
ds_bpermute_b32 v41, v134, v41, offset:28
ds_bpermute_b32 v42, v134, v42, offset:28
ds_bpermute_b32 v43, v134, v43, offset:28
ds_bpermute_b32 v72, v134, v72, offset:28
ds_bpermute_b32 v73, v134, v73, offset:28
ds_bpermute_b32 v74, v134, v74, offset:28
ds_bpermute_b32 v75, v134, v75, offset:28
ds_bpermute_b32 v104, v134, v104, offset:28
ds_bpermute_b32 v105, v134, v105, offset:28
ds_bpermute_b32 v106, v134, v106, offset:28
ds_bpermute_b32 v107, v134, v107, offset:28
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0028:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v12, v134, v12, offset:28
ds_bpermute_b32 v13, v134, v13, offset:28
ds_bpermute_b32 v14, v134, v14, offset:28
ds_bpermute_b32 v15, v134, v15, offset:28
ds_bpermute_b32 v44, v134, v44, offset:28
ds_bpermute_b32 v45, v134, v45, offset:28
ds_bpermute_b32 v46, v134, v46, offset:28
ds_bpermute_b32 v47, v134, v47, offset:28
ds_bpermute_b32 v76, v134, v76, offset:28
ds_bpermute_b32 v77, v134, v77, offset:28
ds_bpermute_b32 v78, v134, v78, offset:28
ds_bpermute_b32 v79, v134, v79, offset:28
ds_bpermute_b32 v108, v134, v108, offset:28
ds_bpermute_b32 v109, v134, v109, offset:28
ds_bpermute_b32 v110, v134, v110, offset:28
ds_bpermute_b32 v111, v134, v111, offset:28
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0029:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v16, v134, v16, offset:28
ds_bpermute_b32 v17, v134, v17, offset:28
ds_bpermute_b32 v18, v134, v18, offset:28
ds_bpermute_b32 v19, v134, v19, offset:28
ds_bpermute_b32 v48, v134, v48, offset:28
ds_bpermute_b32 v49, v134, v49, offset:28
ds_bpermute_b32 v50, v134, v50, offset:28
ds_bpermute_b32 v51, v134, v51, offset:28
ds_bpermute_b32 v80, v134, v80, offset:28
ds_bpermute_b32 v81, v134, v81, offset:28
ds_bpermute_b32 v82, v134, v82, offset:28
ds_bpermute_b32 v83, v134, v83, offset:28
ds_bpermute_b32 v112, v134, v112, offset:28
ds_bpermute_b32 v113, v134, v113, offset:28
ds_bpermute_b32 v114, v134, v114, offset:28
ds_bpermute_b32 v115, v134, v115, offset:28
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0030:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v20, v134, v20, offset:28
ds_bpermute_b32 v21, v134, v21, offset:28
ds_bpermute_b32 v22, v134, v22, offset:28
ds_bpermute_b32 v23, v134, v23, offset:28
ds_bpermute_b32 v52, v134, v52, offset:28
ds_bpermute_b32 v53, v134, v53, offset:28
ds_bpermute_b32 v54, v134, v54, offset:28
ds_bpermute_b32 v55, v134, v55, offset:28
ds_bpermute_b32 v84, v134, v84, offset:28
ds_bpermute_b32 v85, v134, v85, offset:28
ds_bpermute_b32 v86, v134, v86, offset:28
ds_bpermute_b32 v87, v134, v87, offset:28
ds_bpermute_b32 v116, v134, v116, offset:28
ds_bpermute_b32 v117, v134, v117, offset:28
ds_bpermute_b32 v118, v134, v118, offset:28
ds_bpermute_b32 v119, v134, v119, offset:28
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0031:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v24, v134, v24, offset:28
ds_bpermute_b32 v25, v134, v25, offset:28
ds_bpermute_b32 v26, v134, v26, offset:28
ds_bpermute_b32 v27, v134, v27, offset:28
ds_bpermute_b32 v56, v134, v56, offset:28
ds_bpermute_b32 v57, v134, v57, offset:28
ds_bpermute_b32 v58, v134, v58, offset:28
ds_bpermute_b32 v59, v134, v59, offset:28
ds_bpermute_b32 v88, v134, v88, offset:28
ds_bpermute_b32 v89, v134, v89, offset:28
ds_bpermute_b32 v90, v134, v90, offset:28
ds_bpermute_b32 v91, v134, v91, offset:28
ds_bpermute_b32 v120, v134, v120, offset:28
ds_bpermute_b32 v121, v134, v121, offset:28
ds_bpermute_b32 v122, v134, v122, offset:28
ds_bpermute_b32 v123, v134, v123, offset:28
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0032:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v28, v134, v28, offset:28
ds_bpermute_b32 v29, v134, v29, offset:28
ds_bpermute_b32 v30, v134, v30, offset:28
ds_bpermute_b32 v31, v134, v31, offset:28
ds_bpermute_b32 v60, v134, v60, offset:28
ds_bpermute_b32 v61, v134, v61, offset:28
ds_bpermute_b32 v62, v134, v62, offset:28
ds_bpermute_b32 v63, v134, v63, offset:28
ds_bpermute_b32 v92, v134, v92, offset:28
ds_bpermute_b32 v93, v134, v93, offset:28
ds_bpermute_b32 v94, v134, v94, offset:28
ds_bpermute_b32 v95, v134, v95, offset:28
ds_bpermute_b32 v124, v134, v124, offset:28
ds_bpermute_b32 v125, v134, v125, offset:28
ds_bpermute_b32 v126, v134, v126, offset:28
ds_bpermute_b32 v127, v134, v127, offset:28
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0033:
v_cmp_eq_u32 vcc, v135, 0x0
s_cbranch_vccnz label_0034
v_cmp_eq_u32 vcc, v135, 0x1
s_cbranch_vccnz label_0035
v_cmp_eq_u32 vcc, v135, 0x2
s_cbranch_vccnz label_0036
v_cmp_eq_u32 vcc, v135, 0x3
s_cbranch_vccnz label_0037
v_cmp_eq_u32 vcc, v135, 0x4
s_cbranch_vccnz label_0038
v_cmp_eq_u32 vcc, v135, 0x5
s_cbranch_vccnz label_0039
v_cmp_eq_u32 vcc, v135, 0x6
s_cbranch_vccnz label_0040
v_cmp_eq_u32 vcc, v135, 0x7
s_cbranch_vccnz label_0041

label_0034:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v0, v134, v0, offset:24
ds_bpermute_b32 v1, v134, v1, offset:24
ds_bpermute_b32 v2, v134, v2, offset:24
ds_bpermute_b32 v3, v134, v3, offset:24
ds_bpermute_b32 v32, v134, v32, offset:24
ds_bpermute_b32 v33, v134, v33, offset:24
ds_bpermute_b32 v34, v134, v34, offset:24
ds_bpermute_b32 v35, v134, v35, offset:24
ds_bpermute_b32 v64, v134, v64, offset:24
ds_bpermute_b32 v65, v134, v65, offset:24
ds_bpermute_b32 v66, v134, v66, offset:24
ds_bpermute_b32 v67, v134, v67, offset:24
ds_bpermute_b32 v96, v134, v96, offset:24
ds_bpermute_b32 v97, v134, v97, offset:24
ds_bpermute_b32 v98, v134, v98, offset:24
ds_bpermute_b32 v99, v134, v99, offset:24
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0035:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v4, v134, v4, offset:24
ds_bpermute_b32 v5, v134, v5, offset:24
ds_bpermute_b32 v6, v134, v6, offset:24
ds_bpermute_b32 v7, v134, v7, offset:24
ds_bpermute_b32 v36, v134, v36, offset:24
ds_bpermute_b32 v37, v134, v37, offset:24
ds_bpermute_b32 v38, v134, v38, offset:24
ds_bpermute_b32 v39, v134, v39, offset:24
ds_bpermute_b32 v68, v134, v68, offset:24
ds_bpermute_b32 v69, v134, v69, offset:24
ds_bpermute_b32 v70, v134, v70, offset:24
ds_bpermute_b32 v71, v134, v71, offset:24
ds_bpermute_b32 v100, v134, v100, offset:24
ds_bpermute_b32 v101, v134, v101, offset:24
ds_bpermute_b32 v102, v134, v102, offset:24
ds_bpermute_b32 v103, v134, v103, offset:24
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0036:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v8, v134, v8, offset:24
ds_bpermute_b32 v9, v134, v9, offset:24
ds_bpermute_b32 v10, v134, v10, offset:24
ds_bpermute_b32 v11, v134, v11, offset:24
ds_bpermute_b32 v40, v134, v40, offset:24
ds_bpermute_b32 v41, v134, v41, offset:24
ds_bpermute_b32 v42, v134, v42, offset:24
ds_bpermute_b32 v43, v134, v43, offset:24
ds_bpermute_b32 v72, v134, v72, offset:24
ds_bpermute_b32 v73, v134, v73, offset:24
ds_bpermute_b32 v74, v134, v74, offset:24
ds_bpermute_b32 v75, v134, v75, offset:24
ds_bpermute_b32 v104, v134, v104, offset:24
ds_bpermute_b32 v105, v134, v105, offset:24
ds_bpermute_b32 v106, v134, v106, offset:24
ds_bpermute_b32 v107, v134, v107, offset:24
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0037:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v12, v134, v12, offset:24
ds_bpermute_b32 v13, v134, v13, offset:24
ds_bpermute_b32 v14, v134, v14, offset:24
ds_bpermute_b32 v15, v134, v15, offset:24
ds_bpermute_b32 v44, v134, v44, offset:24
ds_bpermute_b32 v45, v134, v45, offset:24
ds_bpermute_b32 v46, v134, v46, offset:24
ds_bpermute_b32 v47, v134, v47, offset:24
ds_bpermute_b32 v76, v134, v76, offset:24
ds_bpermute_b32 v77, v134, v77, offset:24
ds_bpermute_b32 v78, v134, v78, offset:24
ds_bpermute_b32 v79, v134, v79, offset:24
ds_bpermute_b32 v108, v134, v108, offset:24
ds_bpermute_b32 v109, v134, v109, offset:24
ds_bpermute_b32 v110, v134, v110, offset:24
ds_bpermute_b32 v111, v134, v111, offset:24
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0038:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v16, v134, v16, offset:24
ds_bpermute_b32 v17, v134, v17, offset:24
ds_bpermute_b32 v18, v134, v18, offset:24
ds_bpermute_b32 v19, v134, v19, offset:24
ds_bpermute_b32 v48, v134, v48, offset:24
ds_bpermute_b32 v49, v134, v49, offset:24
ds_bpermute_b32 v50, v134, v50, offset:24
ds_bpermute_b32 v51, v134, v51, offset:24
ds_bpermute_b32 v80, v134, v80, offset:24
ds_bpermute_b32 v81, v134, v81, offset:24
ds_bpermute_b32 v82, v134, v82, offset:24
ds_bpermute_b32 v83, v134, v83, offset:24
ds_bpermute_b32 v112, v134, v112, offset:24
ds_bpermute_b32 v113, v134, v113, offset:24
ds_bpermute_b32 v114, v134, v114, offset:24
ds_bpermute_b32 v115, v134, v115, offset:24
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0039:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v20, v134, v20, offset:24
ds_bpermute_b32 v21, v134, v21, offset:24
ds_bpermute_b32 v22, v134, v22, offset:24
ds_bpermute_b32 v23, v134, v23, offset:24
ds_bpermute_b32 v52, v134, v52, offset:24
ds_bpermute_b32 v53, v134, v53, offset:24
ds_bpermute_b32 v54, v134, v54, offset:24
ds_bpermute_b32 v55, v134, v55, offset:24
ds_bpermute_b32 v84, v134, v84, offset:24
ds_bpermute_b32 v85, v134, v85, offset:24
ds_bpermute_b32 v86, v134, v86, offset:24
ds_bpermute_b32 v87, v134, v87, offset:24
ds_bpermute_b32 v116, v134, v116, offset:24
ds_bpermute_b32 v117, v134, v117, offset:24
ds_bpermute_b32 v118, v134, v118, offset:24
ds_bpermute_b32 v119, v134, v119, offset:24
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0040:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v24, v134, v24, offset:24
ds_bpermute_b32 v25, v134, v25, offset:24
ds_bpermute_b32 v26, v134, v26, offset:24
ds_bpermute_b32 v27, v134, v27, offset:24
ds_bpermute_b32 v56, v134, v56, offset:24
ds_bpermute_b32 v57, v134, v57, offset:24
ds_bpermute_b32 v58, v134, v58, offset:24
ds_bpermute_b32 v59, v134, v59, offset:24
ds_bpermute_b32 v88, v134, v88, offset:24
ds_bpermute_b32 v89, v134, v89, offset:24
ds_bpermute_b32 v90, v134, v90, offset:24
ds_bpermute_b32 v91, v134, v91, offset:24
ds_bpermute_b32 v120, v134, v120, offset:24
ds_bpermute_b32 v121, v134, v121, offset:24
ds_bpermute_b32 v122, v134, v122, offset:24
ds_bpermute_b32 v123, v134, v123, offset:24
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0041:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v28, v134, v28, offset:24
ds_bpermute_b32 v29, v134, v29, offset:24
ds_bpermute_b32 v30, v134, v30, offset:24
ds_bpermute_b32 v31, v134, v31, offset:24
ds_bpermute_b32 v60, v134, v60, offset:24
ds_bpermute_b32 v61, v134, v61, offset:24
ds_bpermute_b32 v62, v134, v62, offset:24
ds_bpermute_b32 v63, v134, v63, offset:24
ds_bpermute_b32 v92, v134, v92, offset:24
ds_bpermute_b32 v93, v134, v93, offset:24
ds_bpermute_b32 v94, v134, v94, offset:24
ds_bpermute_b32 v95, v134, v95, offset:24
ds_bpermute_b32 v124, v134, v124, offset:24
ds_bpermute_b32 v125, v134, v125, offset:24
ds_bpermute_b32 v126, v134, v126, offset:24
ds_bpermute_b32 v127, v134, v127, offset:24
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0042:
v_cmp_eq_u32 vcc, v135, 0x0
s_cbranch_vccnz label_0043
v_cmp_eq_u32 vcc, v135, 0x1
s_cbranch_vccnz label_0044
v_cmp_eq_u32 vcc, v135, 0x2
s_cbranch_vccnz label_0045
v_cmp_eq_u32 vcc, v135, 0x3
s_cbranch_vccnz label_0046
v_cmp_eq_u32 vcc, v135, 0x4
s_cbranch_vccnz label_0047
v_cmp_eq_u32 vcc, v135, 0x5
s_cbranch_vccnz label_0048
v_cmp_eq_u32 vcc, v135, 0x6
s_cbranch_vccnz label_0049
v_cmp_eq_u32 vcc, v135, 0x7
s_cbranch_vccnz label_0050

label_0043:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v0, v134, v0, offset:20
ds_bpermute_b32 v1, v134, v1, offset:20
ds_bpermute_b32 v2, v134, v2, offset:20
ds_bpermute_b32 v3, v134, v3, offset:20
ds_bpermute_b32 v32, v134, v32, offset:20
ds_bpermute_b32 v33, v134, v33, offset:20
ds_bpermute_b32 v34, v134, v34, offset:20
ds_bpermute_b32 v35, v134, v35, offset:20
ds_bpermute_b32 v64, v134, v64, offset:20
ds_bpermute_b32 v65, v134, v65, offset:20
ds_bpermute_b32 v66, v134, v66, offset:20
ds_bpermute_b32 v67, v134, v67, offset:20
ds_bpermute_b32 v96, v134, v96, offset:20
ds_bpermute_b32 v97, v134, v97, offset:20
ds_bpermute_b32 v98, v134, v98, offset:20
ds_bpermute_b32 v99, v134, v99, offset:20
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0044:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v4, v134, v4, offset:20
ds_bpermute_b32 v5, v134, v5, offset:20
ds_bpermute_b32 v6, v134, v6, offset:20
ds_bpermute_b32 v7, v134, v7, offset:20
ds_bpermute_b32 v36, v134, v36, offset:20
ds_bpermute_b32 v37, v134, v37, offset:20
ds_bpermute_b32 v38, v134, v38, offset:20
ds_bpermute_b32 v39, v134, v39, offset:20
ds_bpermute_b32 v68, v134, v68, offset:20
ds_bpermute_b32 v69, v134, v69, offset:20
ds_bpermute_b32 v70, v134, v70, offset:20
ds_bpermute_b32 v71, v134, v71, offset:20
ds_bpermute_b32 v100, v134, v100, offset:20
ds_bpermute_b32 v101, v134, v101, offset:20
ds_bpermute_b32 v102, v134, v102, offset:20
ds_bpermute_b32 v103, v134, v103, offset:20
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0045:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v8, v134, v8, offset:20
ds_bpermute_b32 v9, v134, v9, offset:20
ds_bpermute_b32 v10, v134, v10, offset:20
ds_bpermute_b32 v11, v134, v11, offset:20
ds_bpermute_b32 v40, v134, v40, offset:20
ds_bpermute_b32 v41, v134, v41, offset:20
ds_bpermute_b32 v42, v134, v42, offset:20
ds_bpermute_b32 v43, v134, v43, offset:20
ds_bpermute_b32 v72, v134, v72, offset:20
ds_bpermute_b32 v73, v134, v73, offset:20
ds_bpermute_b32 v74, v134, v74, offset:20
ds_bpermute_b32 v75, v134, v75, offset:20
ds_bpermute_b32 v104, v134, v104, offset:20
ds_bpermute_b32 v105, v134, v105, offset:20
ds_bpermute_b32 v106, v134, v106, offset:20
ds_bpermute_b32 v107, v134, v107, offset:20
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0046:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v12, v134, v12, offset:20
ds_bpermute_b32 v13, v134, v13, offset:20
ds_bpermute_b32 v14, v134, v14, offset:20
ds_bpermute_b32 v15, v134, v15, offset:20
ds_bpermute_b32 v44, v134, v44, offset:20
ds_bpermute_b32 v45, v134, v45, offset:20
ds_bpermute_b32 v46, v134, v46, offset:20
ds_bpermute_b32 v47, v134, v47, offset:20
ds_bpermute_b32 v76, v134, v76, offset:20
ds_bpermute_b32 v77, v134, v77, offset:20
ds_bpermute_b32 v78, v134, v78, offset:20
ds_bpermute_b32 v79, v134, v79, offset:20
ds_bpermute_b32 v108, v134, v108, offset:20
ds_bpermute_b32 v109, v134, v109, offset:20
ds_bpermute_b32 v110, v134, v110, offset:20
ds_bpermute_b32 v111, v134, v111, offset:20
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0047:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v16, v134, v16, offset:20
ds_bpermute_b32 v17, v134, v17, offset:20
ds_bpermute_b32 v18, v134, v18, offset:20
ds_bpermute_b32 v19, v134, v19, offset:20
ds_bpermute_b32 v48, v134, v48, offset:20
ds_bpermute_b32 v49, v134, v49, offset:20
ds_bpermute_b32 v50, v134, v50, offset:20
ds_bpermute_b32 v51, v134, v51, offset:20
ds_bpermute_b32 v80, v134, v80, offset:20
ds_bpermute_b32 v81, v134, v81, offset:20
ds_bpermute_b32 v82, v134, v82, offset:20
ds_bpermute_b32 v83, v134, v83, offset:20
ds_bpermute_b32 v112, v134, v112, offset:20
ds_bpermute_b32 v113, v134, v113, offset:20
ds_bpermute_b32 v114, v134, v114, offset:20
ds_bpermute_b32 v115, v134, v115, offset:20
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0048:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v20, v134, v20, offset:20
ds_bpermute_b32 v21, v134, v21, offset:20
ds_bpermute_b32 v22, v134, v22, offset:20
ds_bpermute_b32 v23, v134, v23, offset:20
ds_bpermute_b32 v52, v134, v52, offset:20
ds_bpermute_b32 v53, v134, v53, offset:20
ds_bpermute_b32 v54, v134, v54, offset:20
ds_bpermute_b32 v55, v134, v55, offset:20
ds_bpermute_b32 v84, v134, v84, offset:20
ds_bpermute_b32 v85, v134, v85, offset:20
ds_bpermute_b32 v86, v134, v86, offset:20
ds_bpermute_b32 v87, v134, v87, offset:20
ds_bpermute_b32 v116, v134, v116, offset:20
ds_bpermute_b32 v117, v134, v117, offset:20
ds_bpermute_b32 v118, v134, v118, offset:20
ds_bpermute_b32 v119, v134, v119, offset:20
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0049:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v24, v134, v24, offset:20
ds_bpermute_b32 v25, v134, v25, offset:20
ds_bpermute_b32 v26, v134, v26, offset:20
ds_bpermute_b32 v27, v134, v27, offset:20
ds_bpermute_b32 v56, v134, v56, offset:20
ds_bpermute_b32 v57, v134, v57, offset:20
ds_bpermute_b32 v58, v134, v58, offset:20
ds_bpermute_b32 v59, v134, v59, offset:20
ds_bpermute_b32 v88, v134, v88, offset:20
ds_bpermute_b32 v89, v134, v89, offset:20
ds_bpermute_b32 v90, v134, v90, offset:20
ds_bpermute_b32 v91, v134, v91, offset:20
ds_bpermute_b32 v120, v134, v120, offset:20
ds_bpermute_b32 v121, v134, v121, offset:20
ds_bpermute_b32 v122, v134, v122, offset:20
ds_bpermute_b32 v123, v134, v123, offset:20
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0050:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v28, v134, v28, offset:20
ds_bpermute_b32 v29, v134, v29, offset:20
ds_bpermute_b32 v30, v134, v30, offset:20
ds_bpermute_b32 v31, v134, v31, offset:20
ds_bpermute_b32 v60, v134, v60, offset:20
ds_bpermute_b32 v61, v134, v61, offset:20
ds_bpermute_b32 v62, v134, v62, offset:20
ds_bpermute_b32 v63, v134, v63, offset:20
ds_bpermute_b32 v92, v134, v92, offset:20
ds_bpermute_b32 v93, v134, v93, offset:20
ds_bpermute_b32 v94, v134, v94, offset:20
ds_bpermute_b32 v95, v134, v95, offset:20
ds_bpermute_b32 v124, v134, v124, offset:20
ds_bpermute_b32 v125, v134, v125, offset:20
ds_bpermute_b32 v126, v134, v126, offset:20
ds_bpermute_b32 v127, v134, v127, offset:20
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0051:
v_cmp_eq_u32 vcc, v135, 0x0
s_cbranch_vccnz label_0052
v_cmp_eq_u32 vcc, v135, 0x1
s_cbranch_vccnz label_0053
v_cmp_eq_u32 vcc, v135, 0x2
s_cbranch_vccnz label_0054
v_cmp_eq_u32 vcc, v135, 0x3
s_cbranch_vccnz label_0055
v_cmp_eq_u32 vcc, v135, 0x4
s_cbranch_vccnz label_0056
v_cmp_eq_u32 vcc, v135, 0x5
s_cbranch_vccnz label_0057
v_cmp_eq_u32 vcc, v135, 0x6
s_cbranch_vccnz label_0058
v_cmp_eq_u32 vcc, v135, 0x7
s_cbranch_vccnz label_0059

label_0052:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v0, v134, v0, offset:16
ds_bpermute_b32 v1, v134, v1, offset:16
ds_bpermute_b32 v2, v134, v2, offset:16
ds_bpermute_b32 v3, v134, v3, offset:16
ds_bpermute_b32 v32, v134, v32, offset:16
ds_bpermute_b32 v33, v134, v33, offset:16
ds_bpermute_b32 v34, v134, v34, offset:16
ds_bpermute_b32 v35, v134, v35, offset:16
ds_bpermute_b32 v64, v134, v64, offset:16
ds_bpermute_b32 v65, v134, v65, offset:16
ds_bpermute_b32 v66, v134, v66, offset:16
ds_bpermute_b32 v67, v134, v67, offset:16
ds_bpermute_b32 v96, v134, v96, offset:16
ds_bpermute_b32 v97, v134, v97, offset:16
ds_bpermute_b32 v98, v134, v98, offset:16
ds_bpermute_b32 v99, v134, v99, offset:16
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0053:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v4, v134, v4, offset:16
ds_bpermute_b32 v5, v134, v5, offset:16
ds_bpermute_b32 v6, v134, v6, offset:16
ds_bpermute_b32 v7, v134, v7, offset:16
ds_bpermute_b32 v36, v134, v36, offset:16
ds_bpermute_b32 v37, v134, v37, offset:16
ds_bpermute_b32 v38, v134, v38, offset:16
ds_bpermute_b32 v39, v134, v39, offset:16
ds_bpermute_b32 v68, v134, v68, offset:16
ds_bpermute_b32 v69, v134, v69, offset:16
ds_bpermute_b32 v70, v134, v70, offset:16
ds_bpermute_b32 v71, v134, v71, offset:16
ds_bpermute_b32 v100, v134, v100, offset:16
ds_bpermute_b32 v101, v134, v101, offset:16
ds_bpermute_b32 v102, v134, v102, offset:16
ds_bpermute_b32 v103, v134, v103, offset:16
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0054:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v8, v134, v8, offset:16
ds_bpermute_b32 v9, v134, v9, offset:16
ds_bpermute_b32 v10, v134, v10, offset:16
ds_bpermute_b32 v11, v134, v11, offset:16
ds_bpermute_b32 v40, v134, v40, offset:16
ds_bpermute_b32 v41, v134, v41, offset:16
ds_bpermute_b32 v42, v134, v42, offset:16
ds_bpermute_b32 v43, v134, v43, offset:16
ds_bpermute_b32 v72, v134, v72, offset:16
ds_bpermute_b32 v73, v134, v73, offset:16
ds_bpermute_b32 v74, v134, v74, offset:16
ds_bpermute_b32 v75, v134, v75, offset:16
ds_bpermute_b32 v104, v134, v104, offset:16
ds_bpermute_b32 v105, v134, v105, offset:16
ds_bpermute_b32 v106, v134, v106, offset:16
ds_bpermute_b32 v107, v134, v107, offset:16
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0055:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v12, v134, v12, offset:16
ds_bpermute_b32 v13, v134, v13, offset:16
ds_bpermute_b32 v14, v134, v14, offset:16
ds_bpermute_b32 v15, v134, v15, offset:16
ds_bpermute_b32 v44, v134, v44, offset:16
ds_bpermute_b32 v45, v134, v45, offset:16
ds_bpermute_b32 v46, v134, v46, offset:16
ds_bpermute_b32 v47, v134, v47, offset:16
ds_bpermute_b32 v76, v134, v76, offset:16
ds_bpermute_b32 v77, v134, v77, offset:16
ds_bpermute_b32 v78, v134, v78, offset:16
ds_bpermute_b32 v79, v134, v79, offset:16
ds_bpermute_b32 v108, v134, v108, offset:16
ds_bpermute_b32 v109, v134, v109, offset:16
ds_bpermute_b32 v110, v134, v110, offset:16
ds_bpermute_b32 v111, v134, v111, offset:16
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0056:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v16, v134, v16, offset:16
ds_bpermute_b32 v17, v134, v17, offset:16
ds_bpermute_b32 v18, v134, v18, offset:16
ds_bpermute_b32 v19, v134, v19, offset:16
ds_bpermute_b32 v48, v134, v48, offset:16
ds_bpermute_b32 v49, v134, v49, offset:16
ds_bpermute_b32 v50, v134, v50, offset:16
ds_bpermute_b32 v51, v134, v51, offset:16
ds_bpermute_b32 v80, v134, v80, offset:16
ds_bpermute_b32 v81, v134, v81, offset:16
ds_bpermute_b32 v82, v134, v82, offset:16
ds_bpermute_b32 v83, v134, v83, offset:16
ds_bpermute_b32 v112, v134, v112, offset:16
ds_bpermute_b32 v113, v134, v113, offset:16
ds_bpermute_b32 v114, v134, v114, offset:16
ds_bpermute_b32 v115, v134, v115, offset:16
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0057:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v20, v134, v20, offset:16
ds_bpermute_b32 v21, v134, v21, offset:16
ds_bpermute_b32 v22, v134, v22, offset:16
ds_bpermute_b32 v23, v134, v23, offset:16
ds_bpermute_b32 v52, v134, v52, offset:16
ds_bpermute_b32 v53, v134, v53, offset:16
ds_bpermute_b32 v54, v134, v54, offset:16
ds_bpermute_b32 v55, v134, v55, offset:16
ds_bpermute_b32 v84, v134, v84, offset:16
ds_bpermute_b32 v85, v134, v85, offset:16
ds_bpermute_b32 v86, v134, v86, offset:16
ds_bpermute_b32 v87, v134, v87, offset:16
ds_bpermute_b32 v116, v134, v116, offset:16
ds_bpermute_b32 v117, v134, v117, offset:16
ds_bpermute_b32 v118, v134, v118, offset:16
ds_bpermute_b32 v119, v134, v119, offset:16
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0058:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v24, v134, v24, offset:16
ds_bpermute_b32 v25, v134, v25, offset:16
ds_bpermute_b32 v26, v134, v26, offset:16
ds_bpermute_b32 v27, v134, v27, offset:16
ds_bpermute_b32 v56, v134, v56, offset:16
ds_bpermute_b32 v57, v134, v57, offset:16
ds_bpermute_b32 v58, v134, v58, offset:16
ds_bpermute_b32 v59, v134, v59, offset:16
ds_bpermute_b32 v88, v134, v88, offset:16
ds_bpermute_b32 v89, v134, v89, offset:16
ds_bpermute_b32 v90, v134, v90, offset:16
ds_bpermute_b32 v91, v134, v91, offset:16
ds_bpermute_b32 v120, v134, v120, offset:16
ds_bpermute_b32 v121, v134, v121, offset:16
ds_bpermute_b32 v122, v134, v122, offset:16
ds_bpermute_b32 v123, v134, v123, offset:16
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0059:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v28, v134, v28, offset:16
ds_bpermute_b32 v29, v134, v29, offset:16
ds_bpermute_b32 v30, v134, v30, offset:16
ds_bpermute_b32 v31, v134, v31, offset:16
ds_bpermute_b32 v60, v134, v60, offset:16
ds_bpermute_b32 v61, v134, v61, offset:16
ds_bpermute_b32 v62, v134, v62, offset:16
ds_bpermute_b32 v63, v134, v63, offset:16
ds_bpermute_b32 v92, v134, v92, offset:16
ds_bpermute_b32 v93, v134, v93, offset:16
ds_bpermute_b32 v94, v134, v94, offset:16
ds_bpermute_b32 v95, v134, v95, offset:16
ds_bpermute_b32 v124, v134, v124, offset:16
ds_bpermute_b32 v125, v134, v125, offset:16
ds_bpermute_b32 v126, v134, v126, offset:16
ds_bpermute_b32 v127, v134, v127, offset:16
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0060:
v_cmp_eq_u32 vcc, v135, 0x0
s_cbranch_vccnz label_0061
v_cmp_eq_u32 vcc, v135, 0x1
s_cbranch_vccnz label_0062
v_cmp_eq_u32 vcc, v135, 0x2
s_cbranch_vccnz label_0063
v_cmp_eq_u32 vcc, v135, 0x3
s_cbranch_vccnz label_0064
v_cmp_eq_u32 vcc, v135, 0x4
s_cbranch_vccnz label_0065
v_cmp_eq_u32 vcc, v135, 0x5
s_cbranch_vccnz label_0066
v_cmp_eq_u32 vcc, v135, 0x6
s_cbranch_vccnz label_0067
v_cmp_eq_u32 vcc, v135, 0x7
s_cbranch_vccnz label_0068

label_0061:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v0, v134, v0, offset:12
ds_bpermute_b32 v1, v134, v1, offset:12
ds_bpermute_b32 v2, v134, v2, offset:12
ds_bpermute_b32 v3, v134, v3, offset:12
ds_bpermute_b32 v32, v134, v32, offset:12
ds_bpermute_b32 v33, v134, v33, offset:12
ds_bpermute_b32 v34, v134, v34, offset:12
ds_bpermute_b32 v35, v134, v35, offset:12
ds_bpermute_b32 v64, v134, v64, offset:12
ds_bpermute_b32 v65, v134, v65, offset:12
ds_bpermute_b32 v66, v134, v66, offset:12
ds_bpermute_b32 v67, v134, v67, offset:12
ds_bpermute_b32 v96, v134, v96, offset:12
ds_bpermute_b32 v97, v134, v97, offset:12
ds_bpermute_b32 v98, v134, v98, offset:12
ds_bpermute_b32 v99, v134, v99, offset:12
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0062:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v4, v134, v4, offset:12
ds_bpermute_b32 v5, v134, v5, offset:12
ds_bpermute_b32 v6, v134, v6, offset:12
ds_bpermute_b32 v7, v134, v7, offset:12
ds_bpermute_b32 v36, v134, v36, offset:12
ds_bpermute_b32 v37, v134, v37, offset:12
ds_bpermute_b32 v38, v134, v38, offset:12
ds_bpermute_b32 v39, v134, v39, offset:12
ds_bpermute_b32 v68, v134, v68, offset:12
ds_bpermute_b32 v69, v134, v69, offset:12
ds_bpermute_b32 v70, v134, v70, offset:12
ds_bpermute_b32 v71, v134, v71, offset:12
ds_bpermute_b32 v100, v134, v100, offset:12
ds_bpermute_b32 v101, v134, v101, offset:12
ds_bpermute_b32 v102, v134, v102, offset:12
ds_bpermute_b32 v103, v134, v103, offset:12
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0063:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v8, v134, v8, offset:12
ds_bpermute_b32 v9, v134, v9, offset:12
ds_bpermute_b32 v10, v134, v10, offset:12
ds_bpermute_b32 v11, v134, v11, offset:12
ds_bpermute_b32 v40, v134, v40, offset:12
ds_bpermute_b32 v41, v134, v41, offset:12
ds_bpermute_b32 v42, v134, v42, offset:12
ds_bpermute_b32 v43, v134, v43, offset:12
ds_bpermute_b32 v72, v134, v72, offset:12
ds_bpermute_b32 v73, v134, v73, offset:12
ds_bpermute_b32 v74, v134, v74, offset:12
ds_bpermute_b32 v75, v134, v75, offset:12
ds_bpermute_b32 v104, v134, v104, offset:12
ds_bpermute_b32 v105, v134, v105, offset:12
ds_bpermute_b32 v106, v134, v106, offset:12
ds_bpermute_b32 v107, v134, v107, offset:12
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0064:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v12, v134, v12, offset:12
ds_bpermute_b32 v13, v134, v13, offset:12
ds_bpermute_b32 v14, v134, v14, offset:12
ds_bpermute_b32 v15, v134, v15, offset:12
ds_bpermute_b32 v44, v134, v44, offset:12
ds_bpermute_b32 v45, v134, v45, offset:12
ds_bpermute_b32 v46, v134, v46, offset:12
ds_bpermute_b32 v47, v134, v47, offset:12
ds_bpermute_b32 v76, v134, v76, offset:12
ds_bpermute_b32 v77, v134, v77, offset:12
ds_bpermute_b32 v78, v134, v78, offset:12
ds_bpermute_b32 v79, v134, v79, offset:12
ds_bpermute_b32 v108, v134, v108, offset:12
ds_bpermute_b32 v109, v134, v109, offset:12
ds_bpermute_b32 v110, v134, v110, offset:12
ds_bpermute_b32 v111, v134, v111, offset:12
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0065:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v16, v134, v16, offset:12
ds_bpermute_b32 v17, v134, v17, offset:12
ds_bpermute_b32 v18, v134, v18, offset:12
ds_bpermute_b32 v19, v134, v19, offset:12
ds_bpermute_b32 v48, v134, v48, offset:12
ds_bpermute_b32 v49, v134, v49, offset:12
ds_bpermute_b32 v50, v134, v50, offset:12
ds_bpermute_b32 v51, v134, v51, offset:12
ds_bpermute_b32 v80, v134, v80, offset:12
ds_bpermute_b32 v81, v134, v81, offset:12
ds_bpermute_b32 v82, v134, v82, offset:12
ds_bpermute_b32 v83, v134, v83, offset:12
ds_bpermute_b32 v112, v134, v112, offset:12
ds_bpermute_b32 v113, v134, v113, offset:12
ds_bpermute_b32 v114, v134, v114, offset:12
ds_bpermute_b32 v115, v134, v115, offset:12
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0066:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v20, v134, v20, offset:12
ds_bpermute_b32 v21, v134, v21, offset:12
ds_bpermute_b32 v22, v134, v22, offset:12
ds_bpermute_b32 v23, v134, v23, offset:12
ds_bpermute_b32 v52, v134, v52, offset:12
ds_bpermute_b32 v53, v134, v53, offset:12
ds_bpermute_b32 v54, v134, v54, offset:12
ds_bpermute_b32 v55, v134, v55, offset:12
ds_bpermute_b32 v84, v134, v84, offset:12
ds_bpermute_b32 v85, v134, v85, offset:12
ds_bpermute_b32 v86, v134, v86, offset:12
ds_bpermute_b32 v87, v134, v87, offset:12
ds_bpermute_b32 v116, v134, v116, offset:12
ds_bpermute_b32 v117, v134, v117, offset:12
ds_bpermute_b32 v118, v134, v118, offset:12
ds_bpermute_b32 v119, v134, v119, offset:12
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0067:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v24, v134, v24, offset:12
ds_bpermute_b32 v25, v134, v25, offset:12
ds_bpermute_b32 v26, v134, v26, offset:12
ds_bpermute_b32 v27, v134, v27, offset:12
ds_bpermute_b32 v56, v134, v56, offset:12
ds_bpermute_b32 v57, v134, v57, offset:12
ds_bpermute_b32 v58, v134, v58, offset:12
ds_bpermute_b32 v59, v134, v59, offset:12
ds_bpermute_b32 v88, v134, v88, offset:12
ds_bpermute_b32 v89, v134, v89, offset:12
ds_bpermute_b32 v90, v134, v90, offset:12
ds_bpermute_b32 v91, v134, v91, offset:12
ds_bpermute_b32 v120, v134, v120, offset:12
ds_bpermute_b32 v121, v134, v121, offset:12
ds_bpermute_b32 v122, v134, v122, offset:12
ds_bpermute_b32 v123, v134, v123, offset:12
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0068:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v28, v134, v28, offset:12
ds_bpermute_b32 v29, v134, v29, offset:12
ds_bpermute_b32 v30, v134, v30, offset:12
ds_bpermute_b32 v31, v134, v31, offset:12
ds_bpermute_b32 v60, v134, v60, offset:12
ds_bpermute_b32 v61, v134, v61, offset:12
ds_bpermute_b32 v62, v134, v62, offset:12
ds_bpermute_b32 v63, v134, v63, offset:12
ds_bpermute_b32 v92, v134, v92, offset:12
ds_bpermute_b32 v93, v134, v93, offset:12
ds_bpermute_b32 v94, v134, v94, offset:12
ds_bpermute_b32 v95, v134, v95, offset:12
ds_bpermute_b32 v124, v134, v124, offset:12
ds_bpermute_b32 v125, v134, v125, offset:12
ds_bpermute_b32 v126, v134, v126, offset:12
ds_bpermute_b32 v127, v134, v127, offset:12
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0069:
v_cmp_eq_u32 vcc, v135, 0x0
s_cbranch_vccnz label_0070
v_cmp_eq_u32 vcc, v135, 0x1
s_cbranch_vccnz label_0071
v_cmp_eq_u32 vcc, v135, 0x2
s_cbranch_vccnz label_0072
v_cmp_eq_u32 vcc, v135, 0x3
s_cbranch_vccnz label_0073
v_cmp_eq_u32 vcc, v135, 0x4
s_cbranch_vccnz label_0074
v_cmp_eq_u32 vcc, v135, 0x5
s_cbranch_vccnz label_0075
v_cmp_eq_u32 vcc, v135, 0x6
s_cbranch_vccnz label_0076
v_cmp_eq_u32 vcc, v135, 0x7
s_cbranch_vccnz label_0077

label_0070:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v0, v134, v0, offset:8
ds_bpermute_b32 v1, v134, v1, offset:8
ds_bpermute_b32 v2, v134, v2, offset:8
ds_bpermute_b32 v3, v134, v3, offset:8
ds_bpermute_b32 v32, v134, v32, offset:8
ds_bpermute_b32 v33, v134, v33, offset:8
ds_bpermute_b32 v34, v134, v34, offset:8
ds_bpermute_b32 v35, v134, v35, offset:8
ds_bpermute_b32 v64, v134, v64, offset:8
ds_bpermute_b32 v65, v134, v65, offset:8
ds_bpermute_b32 v66, v134, v66, offset:8
ds_bpermute_b32 v67, v134, v67, offset:8
ds_bpermute_b32 v96, v134, v96, offset:8
ds_bpermute_b32 v97, v134, v97, offset:8
ds_bpermute_b32 v98, v134, v98, offset:8
ds_bpermute_b32 v99, v134, v99, offset:8
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0071:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v4, v134, v4, offset:8
ds_bpermute_b32 v5, v134, v5, offset:8
ds_bpermute_b32 v6, v134, v6, offset:8
ds_bpermute_b32 v7, v134, v7, offset:8
ds_bpermute_b32 v36, v134, v36, offset:8
ds_bpermute_b32 v37, v134, v37, offset:8
ds_bpermute_b32 v38, v134, v38, offset:8
ds_bpermute_b32 v39, v134, v39, offset:8
ds_bpermute_b32 v68, v134, v68, offset:8
ds_bpermute_b32 v69, v134, v69, offset:8
ds_bpermute_b32 v70, v134, v70, offset:8
ds_bpermute_b32 v71, v134, v71, offset:8
ds_bpermute_b32 v100, v134, v100, offset:8
ds_bpermute_b32 v101, v134, v101, offset:8
ds_bpermute_b32 v102, v134, v102, offset:8
ds_bpermute_b32 v103, v134, v103, offset:8
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0072:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v8, v134, v8, offset:8
ds_bpermute_b32 v9, v134, v9, offset:8
ds_bpermute_b32 v10, v134, v10, offset:8
ds_bpermute_b32 v11, v134, v11, offset:8
ds_bpermute_b32 v40, v134, v40, offset:8
ds_bpermute_b32 v41, v134, v41, offset:8
ds_bpermute_b32 v42, v134, v42, offset:8
ds_bpermute_b32 v43, v134, v43, offset:8
ds_bpermute_b32 v72, v134, v72, offset:8
ds_bpermute_b32 v73, v134, v73, offset:8
ds_bpermute_b32 v74, v134, v74, offset:8
ds_bpermute_b32 v75, v134, v75, offset:8
ds_bpermute_b32 v104, v134, v104, offset:8
ds_bpermute_b32 v105, v134, v105, offset:8
ds_bpermute_b32 v106, v134, v106, offset:8
ds_bpermute_b32 v107, v134, v107, offset:8
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0073:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v12, v134, v12, offset:8
ds_bpermute_b32 v13, v134, v13, offset:8
ds_bpermute_b32 v14, v134, v14, offset:8
ds_bpermute_b32 v15, v134, v15, offset:8
ds_bpermute_b32 v44, v134, v44, offset:8
ds_bpermute_b32 v45, v134, v45, offset:8
ds_bpermute_b32 v46, v134, v46, offset:8
ds_bpermute_b32 v47, v134, v47, offset:8
ds_bpermute_b32 v76, v134, v76, offset:8
ds_bpermute_b32 v77, v134, v77, offset:8
ds_bpermute_b32 v78, v134, v78, offset:8
ds_bpermute_b32 v79, v134, v79, offset:8
ds_bpermute_b32 v108, v134, v108, offset:8
ds_bpermute_b32 v109, v134, v109, offset:8
ds_bpermute_b32 v110, v134, v110, offset:8
ds_bpermute_b32 v111, v134, v111, offset:8
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0074:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v16, v134, v16, offset:8
ds_bpermute_b32 v17, v134, v17, offset:8
ds_bpermute_b32 v18, v134, v18, offset:8
ds_bpermute_b32 v19, v134, v19, offset:8
ds_bpermute_b32 v48, v134, v48, offset:8
ds_bpermute_b32 v49, v134, v49, offset:8
ds_bpermute_b32 v50, v134, v50, offset:8
ds_bpermute_b32 v51, v134, v51, offset:8
ds_bpermute_b32 v80, v134, v80, offset:8
ds_bpermute_b32 v81, v134, v81, offset:8
ds_bpermute_b32 v82, v134, v82, offset:8
ds_bpermute_b32 v83, v134, v83, offset:8
ds_bpermute_b32 v112, v134, v112, offset:8
ds_bpermute_b32 v113, v134, v113, offset:8
ds_bpermute_b32 v114, v134, v114, offset:8
ds_bpermute_b32 v115, v134, v115, offset:8
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0075:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v20, v134, v20, offset:8
ds_bpermute_b32 v21, v134, v21, offset:8
ds_bpermute_b32 v22, v134, v22, offset:8
ds_bpermute_b32 v23, v134, v23, offset:8
ds_bpermute_b32 v52, v134, v52, offset:8
ds_bpermute_b32 v53, v134, v53, offset:8
ds_bpermute_b32 v54, v134, v54, offset:8
ds_bpermute_b32 v55, v134, v55, offset:8
ds_bpermute_b32 v84, v134, v84, offset:8
ds_bpermute_b32 v85, v134, v85, offset:8
ds_bpermute_b32 v86, v134, v86, offset:8
ds_bpermute_b32 v87, v134, v87, offset:8
ds_bpermute_b32 v116, v134, v116, offset:8
ds_bpermute_b32 v117, v134, v117, offset:8
ds_bpermute_b32 v118, v134, v118, offset:8
ds_bpermute_b32 v119, v134, v119, offset:8
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0076:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v24, v134, v24, offset:8
ds_bpermute_b32 v25, v134, v25, offset:8
ds_bpermute_b32 v26, v134, v26, offset:8
ds_bpermute_b32 v27, v134, v27, offset:8
ds_bpermute_b32 v56, v134, v56, offset:8
ds_bpermute_b32 v57, v134, v57, offset:8
ds_bpermute_b32 v58, v134, v58, offset:8
ds_bpermute_b32 v59, v134, v59, offset:8
ds_bpermute_b32 v88, v134, v88, offset:8
ds_bpermute_b32 v89, v134, v89, offset:8
ds_bpermute_b32 v90, v134, v90, offset:8
ds_bpermute_b32 v91, v134, v91, offset:8
ds_bpermute_b32 v120, v134, v120, offset:8
ds_bpermute_b32 v121, v134, v121, offset:8
ds_bpermute_b32 v122, v134, v122, offset:8
ds_bpermute_b32 v123, v134, v123, offset:8
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0077:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v28, v134, v28, offset:8
ds_bpermute_b32 v29, v134, v29, offset:8
ds_bpermute_b32 v30, v134, v30, offset:8
ds_bpermute_b32 v31, v134, v31, offset:8
ds_bpermute_b32 v60, v134, v60, offset:8
ds_bpermute_b32 v61, v134, v61, offset:8
ds_bpermute_b32 v62, v134, v62, offset:8
ds_bpermute_b32 v63, v134, v63, offset:8
ds_bpermute_b32 v92, v134, v92, offset:8
ds_bpermute_b32 v93, v134, v93, offset:8
ds_bpermute_b32 v94, v134, v94, offset:8
ds_bpermute_b32 v95, v134, v95, offset:8
ds_bpermute_b32 v124, v134, v124, offset:8
ds_bpermute_b32 v125, v134, v125, offset:8
ds_bpermute_b32 v126, v134, v126, offset:8
ds_bpermute_b32 v127, v134, v127, offset:8
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0078:
v_cmp_eq_u32 vcc, v135, 0x0
s_cbranch_vccnz label_0079
v_cmp_eq_u32 vcc, v135, 0x1
s_cbranch_vccnz label_0080
v_cmp_eq_u32 vcc, v135, 0x2
s_cbranch_vccnz label_0081
v_cmp_eq_u32 vcc, v135, 0x3
s_cbranch_vccnz label_0082
v_cmp_eq_u32 vcc, v135, 0x4
s_cbranch_vccnz label_0083
v_cmp_eq_u32 vcc, v135, 0x5
s_cbranch_vccnz label_0084
v_cmp_eq_u32 vcc, v135, 0x6
s_cbranch_vccnz label_0085
v_cmp_eq_u32 vcc, v135, 0x7
s_cbranch_vccnz label_0086

label_0079:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v0, v134, v0, offset:4
ds_bpermute_b32 v1, v134, v1, offset:4
ds_bpermute_b32 v2, v134, v2, offset:4
ds_bpermute_b32 v3, v134, v3, offset:4
ds_bpermute_b32 v32, v134, v32, offset:4
ds_bpermute_b32 v33, v134, v33, offset:4
ds_bpermute_b32 v34, v134, v34, offset:4
ds_bpermute_b32 v35, v134, v35, offset:4
ds_bpermute_b32 v64, v134, v64, offset:4
ds_bpermute_b32 v65, v134, v65, offset:4
ds_bpermute_b32 v66, v134, v66, offset:4
ds_bpermute_b32 v67, v134, v67, offset:4
ds_bpermute_b32 v96, v134, v96, offset:4
ds_bpermute_b32 v97, v134, v97, offset:4
ds_bpermute_b32 v98, v134, v98, offset:4
ds_bpermute_b32 v99, v134, v99, offset:4
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0080:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v4, v134, v4, offset:4
ds_bpermute_b32 v5, v134, v5, offset:4
ds_bpermute_b32 v6, v134, v6, offset:4
ds_bpermute_b32 v7, v134, v7, offset:4
ds_bpermute_b32 v36, v134, v36, offset:4
ds_bpermute_b32 v37, v134, v37, offset:4
ds_bpermute_b32 v38, v134, v38, offset:4
ds_bpermute_b32 v39, v134, v39, offset:4
ds_bpermute_b32 v68, v134, v68, offset:4
ds_bpermute_b32 v69, v134, v69, offset:4
ds_bpermute_b32 v70, v134, v70, offset:4
ds_bpermute_b32 v71, v134, v71, offset:4
ds_bpermute_b32 v100, v134, v100, offset:4
ds_bpermute_b32 v101, v134, v101, offset:4
ds_bpermute_b32 v102, v134, v102, offset:4
ds_bpermute_b32 v103, v134, v103, offset:4
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0081:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v8, v134, v8, offset:4
ds_bpermute_b32 v9, v134, v9, offset:4
ds_bpermute_b32 v10, v134, v10, offset:4
ds_bpermute_b32 v11, v134, v11, offset:4
ds_bpermute_b32 v40, v134, v40, offset:4
ds_bpermute_b32 v41, v134, v41, offset:4
ds_bpermute_b32 v42, v134, v42, offset:4
ds_bpermute_b32 v43, v134, v43, offset:4
ds_bpermute_b32 v72, v134, v72, offset:4
ds_bpermute_b32 v73, v134, v73, offset:4
ds_bpermute_b32 v74, v134, v74, offset:4
ds_bpermute_b32 v75, v134, v75, offset:4
ds_bpermute_b32 v104, v134, v104, offset:4
ds_bpermute_b32 v105, v134, v105, offset:4
ds_bpermute_b32 v106, v134, v106, offset:4
ds_bpermute_b32 v107, v134, v107, offset:4
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0082:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v12, v134, v12, offset:4
ds_bpermute_b32 v13, v134, v13, offset:4
ds_bpermute_b32 v14, v134, v14, offset:4
ds_bpermute_b32 v15, v134, v15, offset:4
ds_bpermute_b32 v44, v134, v44, offset:4
ds_bpermute_b32 v45, v134, v45, offset:4
ds_bpermute_b32 v46, v134, v46, offset:4
ds_bpermute_b32 v47, v134, v47, offset:4
ds_bpermute_b32 v76, v134, v76, offset:4
ds_bpermute_b32 v77, v134, v77, offset:4
ds_bpermute_b32 v78, v134, v78, offset:4
ds_bpermute_b32 v79, v134, v79, offset:4
ds_bpermute_b32 v108, v134, v108, offset:4
ds_bpermute_b32 v109, v134, v109, offset:4
ds_bpermute_b32 v110, v134, v110, offset:4
ds_bpermute_b32 v111, v134, v111, offset:4
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0083:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v16, v134, v16, offset:4
ds_bpermute_b32 v17, v134, v17, offset:4
ds_bpermute_b32 v18, v134, v18, offset:4
ds_bpermute_b32 v19, v134, v19, offset:4
ds_bpermute_b32 v48, v134, v48, offset:4
ds_bpermute_b32 v49, v134, v49, offset:4
ds_bpermute_b32 v50, v134, v50, offset:4
ds_bpermute_b32 v51, v134, v51, offset:4
ds_bpermute_b32 v80, v134, v80, offset:4
ds_bpermute_b32 v81, v134, v81, offset:4
ds_bpermute_b32 v82, v134, v82, offset:4
ds_bpermute_b32 v83, v134, v83, offset:4
ds_bpermute_b32 v112, v134, v112, offset:4
ds_bpermute_b32 v113, v134, v113, offset:4
ds_bpermute_b32 v114, v134, v114, offset:4
ds_bpermute_b32 v115, v134, v115, offset:4
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0084:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v20, v134, v20, offset:4
ds_bpermute_b32 v21, v134, v21, offset:4
ds_bpermute_b32 v22, v134, v22, offset:4
ds_bpermute_b32 v23, v134, v23, offset:4
ds_bpermute_b32 v52, v134, v52, offset:4
ds_bpermute_b32 v53, v134, v53, offset:4
ds_bpermute_b32 v54, v134, v54, offset:4
ds_bpermute_b32 v55, v134, v55, offset:4
ds_bpermute_b32 v84, v134, v84, offset:4
ds_bpermute_b32 v85, v134, v85, offset:4
ds_bpermute_b32 v86, v134, v86, offset:4
ds_bpermute_b32 v87, v134, v87, offset:4
ds_bpermute_b32 v116, v134, v116, offset:4
ds_bpermute_b32 v117, v134, v117, offset:4
ds_bpermute_b32 v118, v134, v118, offset:4
ds_bpermute_b32 v119, v134, v119, offset:4
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0085:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v24, v134, v24, offset:4
ds_bpermute_b32 v25, v134, v25, offset:4
ds_bpermute_b32 v26, v134, v26, offset:4
ds_bpermute_b32 v27, v134, v27, offset:4
ds_bpermute_b32 v56, v134, v56, offset:4
ds_bpermute_b32 v57, v134, v57, offset:4
ds_bpermute_b32 v58, v134, v58, offset:4
ds_bpermute_b32 v59, v134, v59, offset:4
ds_bpermute_b32 v88, v134, v88, offset:4
ds_bpermute_b32 v89, v134, v89, offset:4
ds_bpermute_b32 v90, v134, v90, offset:4
ds_bpermute_b32 v91, v134, v91, offset:4
ds_bpermute_b32 v120, v134, v120, offset:4
ds_bpermute_b32 v121, v134, v121, offset:4
ds_bpermute_b32 v122, v134, v122, offset:4
ds_bpermute_b32 v123, v134, v123, offset:4
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087

label_0086:
_v_cmpx_ge_u32 s[54:55], v147, v145
v_mov_b32 v134, v[vgprSerial]
v_lshlrev_b32 v134, 2, v134
ds_bpermute_b32 v28, v134, v28, offset:4
ds_bpermute_b32 v29, v134, v29, offset:4
ds_bpermute_b32 v30, v134, v30, offset:4
ds_bpermute_b32 v31, v134, v31, offset:4
ds_bpermute_b32 v60, v134, v60, offset:4
ds_bpermute_b32 v61, v134, v61, offset:4
ds_bpermute_b32 v62, v134, v62, offset:4
ds_bpermute_b32 v63, v134, v63, offset:4
ds_bpermute_b32 v92, v134, v92, offset:4
ds_bpermute_b32 v93, v134, v93, offset:4
ds_bpermute_b32 v94, v134, v94, offset:4
ds_bpermute_b32 v95, v134, v95, offset:4
ds_bpermute_b32 v124, v134, v124, offset:4
ds_bpermute_b32 v125, v134, v125, offset:4
ds_bpermute_b32 v126, v134, v126, offset:4
ds_bpermute_b32 v127, v134, v127, offset:4
s_waitcnt 0
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF
s_or_saveexec_b64 vcc, s[54:55]
s_branch label_0087
label_0087:

v_and_b32 v128, 63, v[vgprSerial]
v_lshrrev_b32 v129, 6, v[vgprSerial]
v_and_b32 v132, 15, v128
v_lshrrev_b32 v133, 4, v128
v_and_b32 v134, 1, v129
s_mov_b32 s54, 32
v_mul_lo_u32 v134, s54, v134
v_lshlrev_b32 v128, 0, v132
_v_add_co_u32 v128, vcc, v128, v134
v_lshrrev_b32 v134, 1, v129
s_mov_b32 s54, 32
v_mul_lo_u32 v134, s54, v134
v_lshlrev_b32 v129, 0, v133
_v_add_co_u32 v129, vcc, v129, v134
s_mul_i32 s[sgprStrideoffC+0], 8, s[sgprStridesC]
s_mul_i32 s[sgprStrideoffC+1], 16, s[sgprStridesC]
s_mul_i32 s[sgprStrideoffC+2], 24, s[sgprStridesC]
s_mul_i32 s[sgprStrideoffD+0], 8, s[sgprStridesD]
s_mul_i32 s[sgprStrideoffD+1], 16, s[sgprStridesD]
s_mul_i32 s[sgprStrideoffD+2], 24, s[sgprStridesD]
v_mul_lo_u32 v130, v129, s[sgprStrideC1J]
v_mul_lo_u32 v131, v129, s[sgprStrideD1J]

s_mul_i32 s54, 0x100, s[sgprWorkGroup0]
_v_add_co_u32 v128, vcc, s54, v128
s_mul_i32 s56, 0x100, s[sgprWorkGroup1]
_v_add_co_u32 v129, vcc, s56, v129

GW_B0_DIRECT_101:

v_mov_b32 v206, 0x7fff0000
v_mov_b32 v207, 0x7fff

s_load_dwordx2 s[sgprSrdC:sgprSrdC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_mov_b32 s[sgprSrdC+2], 0xfffffffe
s_mov_b32 s[sgprSrdC+3], Srd127_96
v_readfirstlane_b32 s7, v202
s_mov_b32 s66, 80
s_mov_b32 s67, 128
s_mov_b32 s68, 144
s_mov_b32 s69, 192
s_mov_b32 s70, 208
s_waitcnt vmcnt(0) lgkmcnt(0)

PREFETCH_ROWMAP_D 180, 0
PREFETCH_ROWMAP_D 181, 4
PREFETCH_ROWMAP_D 182, 8
PREFETCH_ROWMAP_D 183, 12
PREFETCH_ROWMAP_D 184, 16
PREFETCH_ROWMAP_D 185, 20
PREFETCH_ROWMAP_D 186, 24
PREFETCH_ROWMAP_D 187, 28
PREFETCH_ROWMAP_D 188, 128
PREFETCH_ROWMAP_D 189, 132
PREFETCH_ROWMAP_D 190, 136
PREFETCH_ROWMAP_D 191, 140
PREFETCH_ROWMAP_D 192, 144
PREFETCH_ROWMAP_D 193, 148
PREFETCH_ROWMAP_D 194, 152
PREFETCH_ROWMAP_D 195, 156
s_waitcnt vmcnt(0)
s_and_b32 s56, s77, 2
s_cmp_eq_u32 s56, 2
s_cbranch_scc1 GW_B0_DIRECT_FAST_BUFFER_102

GLOBAL_ROWMAP_D_ROW_BASE 180, 0
DIRECT_ROWMAP_D_STORE 0, 0
DIRECT_ROWMAP_D_STORE 4, 16
DIRECT_ROWMAP_D_STORE 8, 64
DIRECT_ROWMAP_D_STORE 12, 80
DIRECT_ROWMAP_D_STORE 16, 128
DIRECT_ROWMAP_D_STORE 20, 144
DIRECT_ROWMAP_D_STORE 24, 192
DIRECT_ROWMAP_D_STORE 28, 208
GLOBAL_ROWMAP_D_ROW_BASE 181, 4
DIRECT_ROWMAP_D_STORE 1, 0
DIRECT_ROWMAP_D_STORE 5, 16
DIRECT_ROWMAP_D_STORE 9, 64
DIRECT_ROWMAP_D_STORE 13, 80
DIRECT_ROWMAP_D_STORE 17, 128
DIRECT_ROWMAP_D_STORE 21, 144
DIRECT_ROWMAP_D_STORE 25, 192
DIRECT_ROWMAP_D_STORE 29, 208
GLOBAL_ROWMAP_D_ROW_BASE 182, 8
DIRECT_ROWMAP_D_STORE 2, 0
DIRECT_ROWMAP_D_STORE 6, 16
DIRECT_ROWMAP_D_STORE 10, 64
DIRECT_ROWMAP_D_STORE 14, 80
DIRECT_ROWMAP_D_STORE 18, 128
DIRECT_ROWMAP_D_STORE 22, 144
DIRECT_ROWMAP_D_STORE 26, 192
DIRECT_ROWMAP_D_STORE 30, 208
GLOBAL_ROWMAP_D_ROW_BASE 183, 12
DIRECT_ROWMAP_D_STORE 3, 0
DIRECT_ROWMAP_D_STORE 7, 16
DIRECT_ROWMAP_D_STORE 11, 64
DIRECT_ROWMAP_D_STORE 15, 80
DIRECT_ROWMAP_D_STORE 19, 128
DIRECT_ROWMAP_D_STORE 23, 144
DIRECT_ROWMAP_D_STORE 27, 192
DIRECT_ROWMAP_D_STORE 31, 208

GLOBAL_ROWMAP_D_ROW_BASE 184, 16
DIRECT_ROWMAP_D_STORE 32, 0
DIRECT_ROWMAP_D_STORE 36, 16
DIRECT_ROWMAP_D_STORE 40, 64
DIRECT_ROWMAP_D_STORE 44, 80
DIRECT_ROWMAP_D_STORE 48, 128
DIRECT_ROWMAP_D_STORE 52, 144
DIRECT_ROWMAP_D_STORE 56, 192
DIRECT_ROWMAP_D_STORE 60, 208
GLOBAL_ROWMAP_D_ROW_BASE 185, 20
DIRECT_ROWMAP_D_STORE 33, 0
DIRECT_ROWMAP_D_STORE 37, 16
DIRECT_ROWMAP_D_STORE 41, 64
DIRECT_ROWMAP_D_STORE 45, 80
DIRECT_ROWMAP_D_STORE 49, 128
DIRECT_ROWMAP_D_STORE 53, 144
DIRECT_ROWMAP_D_STORE 57, 192
DIRECT_ROWMAP_D_STORE 61, 208
GLOBAL_ROWMAP_D_ROW_BASE 186, 24
DIRECT_ROWMAP_D_STORE 34, 0
DIRECT_ROWMAP_D_STORE 38, 16
DIRECT_ROWMAP_D_STORE 42, 64
DIRECT_ROWMAP_D_STORE 46, 80
DIRECT_ROWMAP_D_STORE 50, 128
DIRECT_ROWMAP_D_STORE 54, 144
DIRECT_ROWMAP_D_STORE 58, 192
DIRECT_ROWMAP_D_STORE 62, 208
GLOBAL_ROWMAP_D_ROW_BASE 187, 28
DIRECT_ROWMAP_D_STORE 35, 0
DIRECT_ROWMAP_D_STORE 39, 16
DIRECT_ROWMAP_D_STORE 43, 64
DIRECT_ROWMAP_D_STORE 47, 80
DIRECT_ROWMAP_D_STORE 51, 128
DIRECT_ROWMAP_D_STORE 55, 144
DIRECT_ROWMAP_D_STORE 59, 192
DIRECT_ROWMAP_D_STORE 63, 208

GLOBAL_ROWMAP_D_ROW_BASE 188, 128
DIRECT_ROWMAP_D_STORE 64, 0
DIRECT_ROWMAP_D_STORE 68, 16
DIRECT_ROWMAP_D_STORE 72, 64
DIRECT_ROWMAP_D_STORE 76, 80
DIRECT_ROWMAP_D_STORE 80, 128
DIRECT_ROWMAP_D_STORE 84, 144
DIRECT_ROWMAP_D_STORE 88, 192
DIRECT_ROWMAP_D_STORE 92, 208
GLOBAL_ROWMAP_D_ROW_BASE 189, 132
DIRECT_ROWMAP_D_STORE 65, 0
DIRECT_ROWMAP_D_STORE 69, 16
DIRECT_ROWMAP_D_STORE 73, 64
DIRECT_ROWMAP_D_STORE 77, 80
DIRECT_ROWMAP_D_STORE 81, 128
DIRECT_ROWMAP_D_STORE 85, 144
DIRECT_ROWMAP_D_STORE 89, 192
DIRECT_ROWMAP_D_STORE 93, 208
GLOBAL_ROWMAP_D_ROW_BASE 190, 136
DIRECT_ROWMAP_D_STORE 66, 0
DIRECT_ROWMAP_D_STORE 70, 16
DIRECT_ROWMAP_D_STORE 74, 64
DIRECT_ROWMAP_D_STORE 78, 80
DIRECT_ROWMAP_D_STORE 82, 128
DIRECT_ROWMAP_D_STORE 86, 144
DIRECT_ROWMAP_D_STORE 90, 192
DIRECT_ROWMAP_D_STORE 94, 208
GLOBAL_ROWMAP_D_ROW_BASE 191, 140
DIRECT_ROWMAP_D_STORE 67, 0
DIRECT_ROWMAP_D_STORE 71, 16
DIRECT_ROWMAP_D_STORE 75, 64
DIRECT_ROWMAP_D_STORE 79, 80
DIRECT_ROWMAP_D_STORE 83, 128
DIRECT_ROWMAP_D_STORE 87, 144
DIRECT_ROWMAP_D_STORE 91, 192
DIRECT_ROWMAP_D_STORE 95, 208

GLOBAL_ROWMAP_D_ROW_BASE 192, 144
DIRECT_ROWMAP_D_STORE 96, 0
DIRECT_ROWMAP_D_STORE 100, 16
DIRECT_ROWMAP_D_STORE 104, 64
DIRECT_ROWMAP_D_STORE 108, 80
DIRECT_ROWMAP_D_STORE 112, 128
DIRECT_ROWMAP_D_STORE 116, 144
DIRECT_ROWMAP_D_STORE 120, 192
DIRECT_ROWMAP_D_STORE 124, 208
GLOBAL_ROWMAP_D_ROW_BASE 193, 148
DIRECT_ROWMAP_D_STORE 97, 0
DIRECT_ROWMAP_D_STORE 101, 16
DIRECT_ROWMAP_D_STORE 105, 64
DIRECT_ROWMAP_D_STORE 109, 80
DIRECT_ROWMAP_D_STORE 113, 128
DIRECT_ROWMAP_D_STORE 117, 144
DIRECT_ROWMAP_D_STORE 121, 192
DIRECT_ROWMAP_D_STORE 125, 208
GLOBAL_ROWMAP_D_ROW_BASE 194, 152
DIRECT_ROWMAP_D_STORE 98, 0
DIRECT_ROWMAP_D_STORE 102, 16
DIRECT_ROWMAP_D_STORE 106, 64
DIRECT_ROWMAP_D_STORE 110, 80
DIRECT_ROWMAP_D_STORE 114, 128
DIRECT_ROWMAP_D_STORE 118, 144
DIRECT_ROWMAP_D_STORE 122, 192
DIRECT_ROWMAP_D_STORE 126, 208
GLOBAL_ROWMAP_D_ROW_BASE 195, 156
DIRECT_ROWMAP_D_STORE 99, 0
DIRECT_ROWMAP_D_STORE 103, 16
DIRECT_ROWMAP_D_STORE 107, 64
DIRECT_ROWMAP_D_STORE 111, 80
DIRECT_ROWMAP_D_STORE 115, 128
DIRECT_ROWMAP_D_STORE 119, 144
DIRECT_ROWMAP_D_STORE 123, 192
DIRECT_ROWMAP_D_STORE 127, 208
s_branch label_GW_End_109

GW_B0_DIRECT_FAST_BUFFER_102:
DIRECT_ROWMAP_D_BUFFER_ROW 180, 0, 0
DIRECT_ROWMAP_D_BUFFER_ROW 181, 4, 1
DIRECT_ROWMAP_D_BUFFER_ROW 182, 8, 2
DIRECT_ROWMAP_D_BUFFER_ROW 183, 12, 3
DIRECT_ROWMAP_D_BUFFER_ROW 184, 16, 32
DIRECT_ROWMAP_D_BUFFER_ROW 185, 20, 33
DIRECT_ROWMAP_D_BUFFER_ROW 186, 24, 34
DIRECT_ROWMAP_D_BUFFER_ROW 187, 28, 35
DIRECT_ROWMAP_D_BUFFER_ROW 188, 128, 64
DIRECT_ROWMAP_D_BUFFER_ROW 189, 132, 65
DIRECT_ROWMAP_D_BUFFER_ROW 190, 136, 66
DIRECT_ROWMAP_D_BUFFER_ROW 191, 140, 67
DIRECT_ROWMAP_D_BUFFER_ROW 192, 144, 96
DIRECT_ROWMAP_D_BUFFER_ROW 193, 148, 97
DIRECT_ROWMAP_D_BUFFER_ROW 194, 152, 98
DIRECT_ROWMAP_D_BUFFER_ROW 195, 156, 99
s_branch label_GW_End_109

label_GW_End_109:

label_0111:

s_load_dwordx2 s[90:91], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60
s_load_dwordx2 s[92:93], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68
s_waitcnt lgkmcnt(0)
s_or_b32 s80, s90, s91
s_cmp_eq_u32 s80, 0
s_cbranch_scc1 .L_nn_moe_nchunk_signal_end
s_or_b32 s80, s92, s93
s_cmp_eq_u32 s80, 0
s_cbranch_scc1 .L_nn_moe_nchunk_signal_end

s_waitcnt vmcnt(0) lgkmcnt(0)
buffer_wbinvl1_vol
s_barrier

v_cmp_eq_u32 vcc, v[vgprSerial], 0
s_and_saveexec_b64 s[98:99], vcc
s_cbranch_execz .L_nn_moe_nchunk_signal_restore_exec

s_load_dword s80, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x4c
s_load_dword s83, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_load_dword s81, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x30
s_load_dword s82, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x34
s_load_dword s86, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x44
s_waitcnt lgkmcnt(0)
s_lshr_b32 s80, s80, 8
s_cmp_eq_u32 s80, 0
s_cbranch_scc1 .L_nn_moe_nchunk_signal_restore_exec
s_cmp_eq_u32 s83, 0
s_cbranch_scc1 .L_nn_moe_nchunk_signal_restore_exec

s_add_u32 s82, s82, 255
s_lshr_b32 s82, s82, 8
s_mov_b32 s84, 0
s_mov_b32 s85, s100
.L_nn_moe_nchunk_group_div:
s_cmp_lt_u32 s85, s80
s_cbranch_scc1 .L_nn_moe_nchunk_group_div_done
s_sub_u32 s85, s85, s80
s_add_u32 s84, s84, 1
s_branch .L_nn_moe_nchunk_group_div
.L_nn_moe_nchunk_group_div_done:
s_cmp_ge_u32 s84, s83
s_cbranch_scc1 .L_nn_moe_nchunk_signal_restore_exec

s_mul_i32 s87, s84, s80
s_sub_u32 s88, s82, s87
s_cmp_lt_u32 s88, s80
s_cselect_b32 s87, s88, s80
s_mul_i32 s88, s81, s87

s_lshl_b32 s89, s84, 2
s_add_u32 s90, s90, s89
s_addc_u32 s91, s91, 0
v_mov_b32 v0, s90
v_mov_b32 v1, s91
v_mov_b32 v2, 1
global_atomic_add v3, v[0:1], v2, off glc
s_waitcnt vmcnt(0)
v_readfirstlane_b32 s87, v3
s_add_u32 s87, s87, 1
s_cmp_eq_u32 s87, s88
s_cbranch_scc0 .L_nn_moe_nchunk_signal_restore_exec

s_lshl_b32 s89, s84, 2
s_add_u32 s92, s92, s89
s_addc_u32 s93, s93, 0
v_mov_b32 v0, s92
v_mov_b32 v1, s93
v_mov_b32 v2, s86
s_waitcnt vmcnt(0) lgkmcnt(0)
buffer_wbinvl1_vol
global_store_dword v[0:1], v2, off
s_waitcnt vmcnt(0) lgkmcnt(0)
buffer_wbinvl1_vol

.L_nn_moe_nchunk_signal_restore_exec:
s_or_b64 exec, exec, s[98:99]
.L_nn_moe_nchunk_signal_end:
s_endpgm
