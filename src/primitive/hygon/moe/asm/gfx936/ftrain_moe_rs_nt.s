.amdgcn_target "amdgcn-amd-amdhsa--gfx936"
.ifndef FTRAIN_FC2_TILE_SWIZZLE_LOG2
.set FTRAIN_FC2_TILE_SWIZZLE_LOG2, 3
.endif
.text
	.protected ftrain_moe_rs_nt
	.globl ftrain_moe_rs_nt
	.p2align 8
	.type ftrain_moe_rs_nt,@function
	.section .rodata,#alloc
	.p2align 6
	.amdhsa_kernel ftrain_moe_rs_nt
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
  - .name: ftrain_moe_rs_nt
    .symbol: ftrain_moe_rs_nt.kd
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
    .max_flat_workgroup_size: 768
    .private_segment_fixed_size: 0
    .sgpr_count: 101
    .sgpr_spill_count: 0
    .vgpr_count: 255
    .vgpr_spill_count: 0
    .wavefront_size: 64
...
.end_amdgpu_metadata
ftrain_moe_rs_nt:

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
s_mov_b32 s56, \nOffset
v_add_u32 v132, s56, v128
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
s_mov_b32 s56, \nOffset
v_add_u32 v132, s56, v128
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
s_mov_b32 s[sgprStrideAL], s[sgprSizeL]
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

s_mul_i32 s66, s62, s[sgprStrideD1J]
s_add_u32 s66, s66, s64
s_mul_hi_u32 s67, s66, s[sgprSizeL]
s_mul_i32 s66, s66, s[sgprSizeL]
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

s_mul_hi_u32 s[sgprTensor2dSizeA+1], s[sgprSizeL], s84
s_mul_i32 s[sgprTensor2dSizeA+0], s[sgprSizeL], s84
s_mul_hi_u32 s[sgprTensor2dSizeB+1], s81, s[sgprSizeL]
s_mul_i32 s[sgprTensor2dSizeB+0], s81, s[sgprSizeL]

s_barrier
s_branch .L_moe_problem_body
.L_moe_problem_end:
s_endpgm

.L_moe_problem_body:
v_readfirstlane_b32 s98, v202
s_mov_b32 s93, s[sgprSizeJ]
s_mov_b32 s91, s[sgprSizeI]
s_mov_b64 s[94:95], s[sgprAddressD:sgprAddressD+1]
s_mov_b32 s22, s77
s_mov_b32 s61, s[sgprStrideD1J]
v_mov_b32 v32, s[sgprAddressA]
v_mov_b32 v33, s[sgprAddressA+1]
v_mov_b32 v34, s[sgprAddressB]
v_mov_b32 v35, s[sgprAddressB+1]
v_mov_b32 v40, s[sgprSizeL]
v_mov_b32 v246, v[vgprSerial]
s_mov_b32 s20, s91
s_mov_b32 s21, s93
v_readfirstlane_b32 s23, v40
s_mov_b64 s[24:25], s[94:95]
s_mov_b64 s[26:27], s[94:95]
v_readfirstlane_b32 s28, v32
v_readfirstlane_b32 s29, v33
v_readfirstlane_b32 s30, v34
v_readfirstlane_b32 s31, v35
s_sub_u32 s28, s28, 16
s_subb_u32 s29, s29, 0
s_sub_u32 s30, s30, 16
s_subb_u32 s31, s31, 0
s_mov_b32 s32, s61
s_mov_b32 s33, 0
s_mov_b32 s34, s61
s_mov_b32 s35, 0
s_mov_b32 s36, s23
s_mov_b32 s37, 0
s_mov_b32 s38, s23
s_mov_b32 s39, 0
s_mov_b32 s40, 0x3f800000
s_mov_b32 s41, 0
s_mov_b32 s42, 1
s_mov_b32 s2, 0
s_mov_b32 s3, 0
s_mov_b32 s4, 0
s_mov_b32 s7, 1
s_mov_b32 s16, 1
v_lshrrev_b32_e32 v128, 6, v246
v_readfirstlane_b32 s73, v128
s_mov_b32 s72, 0x10000
s_mov_b32 s76, 0
v_and_b32_e32 v0, 63, v246
v_lshrrev_b32_e32 v4, 6, v246
v_and_b32_e32 v5, 15, v0
v_lshrrev_b32_e32 v3, 4, v0
v_and_b32_e32 v6, 15, v0
v_lshrrev_b32_e32 v7, 4, v0
v_and_b32_e32 v8, 3, v6
v_lshrrev_b32_e32 v9, 2, v6
v_lshlrev_b32_e32 v11, 4, v8
v_lshlrev_b32_e32 v8, 9, v8
v_lshlrev_b32_e32 v9, 6, v9
v_lshlrev_b32_e32 v7, 3, v7
v_add_co_u32_e32 v7, vcc, v7, v8
v_add_co_u32_e32 v7, vcc, v7, v9
v_add_co_u32_e32 v7, vcc, v7, v11
v_lshlrev_b32_e32 v242, 1, v7
v_sub_co_u32_e32 v247, vcc, v7, v11
v_lshlrev_b32_e32 v247, 1, v247
v_cmp_eq_u32_e64 s[64:65], v0, 47
v_cmp_eq_u32_e64 s[66:67], v0, 63
s_or_b64 s[64:65], s[64:65], s[66:67]
v_mov_b32_e32 v8, 0x200
v_cndmask_b32_e64 v8, 0, v8, s[64:65]
v_sub_i32 v9, v7, v8
v_lshlrev_b32_e32 v244, 1, v9
v_cmp_eq_u32_e64 s[64:65], v0, 45
v_cmp_eq_u32_e64 s[66:67], v0, 61
s_or_b64 s[68:69], s[64:65], s[66:67]
v_and_b32_e32 v6, 15, v0
v_cmp_eq_u32_e64 s[64:65], v6, 15
v_cmp_eq_u32_e64 s[66:67], v6, 14
s_or_b64 s[64:65], s[64:65], s[66:67]
s_or_b64 s[64:65], s[68:69], s[64:65]
v_mov_b32_e32 v8, 0x200
v_cndmask_b32_e64 v8, 0, v8, s[64:65]
v_sub_i32 v7, v7, v8
v_lshlrev_b32_e32 v249, 1, v7
v_mov_b32_e32 v243, v242
v_mov_b32_e32 v245, v244
v_and_b32_e32 v4, 0xff, v246
v_and_b32_e32 v1, 7, v4
v_lshrrev_b32_e32 v0, 3, v4
v_lshlrev_b32_e32 v1, 3, v1
v_mul_u32_u24_e32 v240, 64, v0
v_add_lshl_u32 v240, v1, v240, 1
v_readfirstlane_b32 s92, v240
s_and_b32 s90, 3, s73
s_lshl_b32 s90, s90, 17
s_or_b32 s92, s92, s90
s_mov_b32 s80, s92
s_waitcnt lgkmcnt(0)
v_and_b32_e32 v6, 0xff, v246
v_lshrrev_b32_e32 v0, 6, v6
v_and_b32_e32 v1, 63, v6
v_and_b32_e32 v2, 7, v1
v_lshrrev_b32_e32 v3, 3, v1
v_lshlrev_b32_e32 v1, 3, v2
v_lshlrev_b32_e32 v3, 2, v3
v_add_co_u32_e32 v0, vcc, v0, v3
v_and_b32_e32 v4, 63, v246
v_lshrrev_b32_e32 v5, 6, v246
v_lshrrev_b32_e32 v3, 4, v4
v_and_b32_e32 v2, 15, v4
v_lshlrev_b32_e32 v3, 3, v3
v_lshlrev_b32_e32 v5, 5, v5
v_add_co_u32_e32 v2, vcc, v2, v5
v_mov_b32_e32 v4, v0
v_add_co_u32_e32 v5, vcc, 32, v4
v_add_co_u32_e32 v11, vcc, 32, v5
v_add_co_u32_e32 v12, vcc, 32, v11
v_add_co_u32_e32 v13, vcc, 32, v12
v_add_co_u32_e32 v14, vcc, 32, v13
v_add_co_u32_e32 v15, vcc, 32, v14
v_add_co_u32_e32 v16, vcc, 32, v15
v_mov_b32_e32 v6, v2
v_add_co_u32_e32 v7, vcc, 16, v6
v_mov_b32_e32 v8, v1
v_mov_b32_e32 v9, v3
v_add_co_u32_e32 v18, vcc, 32, v9
v_mul_lo_u32 v10, s36, v4
v_add_co_u32_e32 v224, vcc, v8, v10
v_add_u32_e32 v224, 8, v224
v_lshlrev_b32_e32 v224, 1, v224
v_mul_lo_u32 v10, s36, v5
v_add_co_u32_e32 v225, vcc, v8, v10
v_add_u32_e32 v225, 8, v225
v_lshlrev_b32_e32 v225, 1, v225
v_mul_lo_u32 v10, s36, v11
v_add_co_u32_e32 v226, vcc, v8, v10
v_add_u32_e32 v226, 8, v226
v_lshlrev_b32_e32 v226, 1, v226
v_mul_lo_u32 v10, s36, v12
v_add_co_u32_e32 v227, vcc, v8, v10
v_add_u32_e32 v227, 8, v227
v_lshlrev_b32_e32 v227, 1, v227
v_mul_lo_u32 v10, s36, v13
v_add_co_u32_e32 v228, vcc, v8, v10
v_add_u32_e32 v228, 8, v228
v_lshlrev_b32_e32 v228, 1, v228
v_mul_lo_u32 v10, s36, v14
v_add_co_u32_e32 v229, vcc, v8, v10
v_add_u32_e32 v229, 8, v229
v_lshlrev_b32_e32 v229, 1, v229
v_mul_lo_u32 v10, s36, v15
v_add_co_u32_e32 v230, vcc, v8, v10
v_add_u32_e32 v230, 8, v230
v_lshlrev_b32_e32 v230, 1, v230
v_mul_lo_u32 v10, s36, v16
v_add_co_u32_e32 v231, vcc, v8, v10
v_add_u32_e32 v231, 8, v231
v_lshlrev_b32_e32 v231, 1, v231
v_mul_lo_u32 v10, s38, v6
v_add_co_u32_e32 v250, vcc, v9, v10
v_add_u32_e32 v250, 8, v250
v_lshlrev_b32_e32 v250, 1, v250
v_mul_lo_u32 v10, s38, v6
v_add_co_u32_e32 v251, vcc, v18, v10
v_add_u32_e32 v251, 8, v251
v_lshlrev_b32_e32 v251, 1, v251
v_mul_lo_u32 v10, s38, v7
v_add_co_u32_e32 v252, vcc, v9, v10
v_add_u32_e32 v252, 8, v252
v_lshlrev_b32_e32 v252, 1, v252
v_mul_lo_u32 v10, s38, v7
v_add_co_u32_e32 v253, vcc, v18, v10
v_add_u32_e32 v253, 8, v253
v_lshlrev_b32_e32 v253, 1, v253
s_cmp_ge_u32 s38, 0x100
s_cmov_b32 s76, 8
s_cmp_ge_u32 s38, 0x200
s_cmov_b32 s76, 9
s_cmp_ge_u32 s38, 0x400
s_cmov_b32 s76, 10
s_cmp_ge_u32 s38, 0x800
s_cmov_b32 s76, 11
s_cmp_ge_u32 s38, 0x1000
s_cmov_b32 s76, 12
s_lshl_b32 s74, 1, s76
s_cmp_lt_u32 s38, s74
s_cbranch_scc1 .L_fc2_tn_2476314
s_lshr_b32 s78, s38, s76
s_lshl_b32 s79, s78, s76
s_sub_u32 s79, s38, s79
s_branch .L_fc2_tn_247631c
.L_fc2_tn_2476314:
s_mov_b32 s78, 1
s_mov_b32 s79, 0
.L_fc2_tn_247631c:
v_mul_lo_u32 v232, s78, v6
v_mul_lo_u32 v233, s79, v6
v_add_u32_e32 v233, v233, v9
v_add_u32_e32 v233, 8, v233
v_lshlrev_b32_e32 v233, 1, v233
v_mul_lo_u32 v234, s78, v6
v_mul_lo_u32 v235, s79, v6
v_add_u32_e32 v235, v235, v18
v_add_u32_e32 v235, 8, v235
v_lshlrev_b32_e32 v235, 1, v235
v_mul_lo_u32 v236, s78, v7
v_mul_lo_u32 v237, s79, v7
v_add_u32_e32 v237, v237, v9
v_add_u32_e32 v237, 8, v237
v_lshlrev_b32_e32 v237, 1, v237
v_mul_lo_u32 v238, s78, v7
v_mul_lo_u32 v239, s79, v7
v_add_u32_e32 v239, v239, v18
v_add_u32_e32 v239, 8, v239
v_lshlrev_b32_e32 v239, 1, v239
s_mul_hi_u32 s57, s2, 0x100
s_mul_i32 s56, s2, 0x100
s_mul_hi_u32 s57, s56, s36
s_mul_i32 s56, s56, s36
s_mov_b32 s58, 1
s_mov_b32 s59, 0
s_sub_u32 s54, s23, 1
s_mul_hi_u32 s55, 1, s54
s_mul_i32 s54, 1, s54
s_add_u32 s58, s58, s54
s_addc_u32 s59, s59, s55
s_sub_u32 s54, s20, 1
s_mul_hi_u32 s55, s36, s54
s_mul_i32 s54, s36, s54
s_add_u32 s58, s58, s54
s_addc_u32 s59, s59, s55
s_sub_u32 s18, s58, s56
s_subb_u32 s19, s59, s57
s_lshl_b64 s[18:19], s[18:19], 1
s_add_u32 s18, s18, 16
s_addc_u32 s19, s19, 0
s_cmp_eq_u32 s19, 0
s_cselect_b32 s46, s18, -2
s_mul_hi_u32 s55, s37, s4
s_mul_i32 s54, s37, s4
s_add_u32 s56, s56, s54
s_addc_u32 s57, s57, s55
s_lshl_b64 s[56:57], s[56:57], 1
s_add_u32 s44, s28, s56
s_addc_u32 s45, s29, s57
s_mov_b32 s47, 0x20000
s_lshr_b32 s5, s23, 6
s_mov_b64 s[84:85], s[44:45]
s_mov_b64 s[86:87], s[46:47]
s_mov_b64 s[88:89], s[18:19]
s_mov_b32 s17, 0x80
s_mul_i32 s17, s17, s5
s_add_u32 s84, s84, s17
s_addc_u32 s85, s85, 0
s_sub_u32 s88, s88, s17
s_subb_u32 s89, s89, 0
s_cmp_eq_u32 s89, 0
s_cmov_b32 s86, s88
s_mul_hi_u32 s57, s3, 0x100
s_mul_i32 s56, s3, 0x100
s_mul_hi_u32 s57, s56, s38
s_mul_i32 s56, s56, s38
s_mov_b32 s58, 1
s_mov_b32 s59, 0
s_sub_u32 s54, s23, 1
s_mul_hi_u32 s55, 1, s54
s_mul_i32 s54, 1, s54
s_add_u32 s58, s58, s54
s_addc_u32 s59, s59, s55
s_sub_u32 s54, s21, 1
s_mul_hi_u32 s55, s38, s54
s_mul_i32 s54, s38, s54
s_add_u32 s58, s58, s54
s_addc_u32 s59, s59, s55
s_mul_hi_u32 s65, s3, 0x100
s_mul_i32 s64, s3, 0x100
s_mul_hi_u32 s65, s64, s38
s_mul_i32 s64, s64, s38
s_sub_u32 s82, s58, s64
s_subb_u32 s83, s59, s65
s_lshl_b64 s[82:83], s[82:83], 1
s_add_u32 s82, s82, 16
s_addc_u32 s83, s83, 0
s_lshr_b32 s5, s23, 6
s_mov_b32 s43, 0x80
s_mul_i32 s43, s43, s5
s_sub_u32 s82, s82, s43
s_subb_u32 s83, s83, 0
s_mul_hi_u32 s65, s3, 0x100
s_mul_i32 s64, s3, 0x100
s_cmp_lt_u32 s38, s74
s_cbranch_scc1 .L_fc2_tn_2476504
s_lshr_b32 s78, s38, s76
s_sub_u32 s52, s21, s64
s_subb_u32 s53, 0, s65
s_mul_hi_u32 s53, s52, s78
s_mul_i32 s52, s52, s78
s_branch .L_fc2_tn_2476514
.L_fc2_tn_2476504:
s_lshr_b32 s78, s38, 13
s_sub_u32 s52, s21, s64
s_subb_u32 s53, 0, s65
s_lshl_b64 s[52:53], s[52:53], s78
.L_fc2_tn_2476514:
s_mul_hi_u32 s65, s64, s38
s_mul_i32 s64, s64, s38
s_cmp_eq_u32 s53, 0
s_cselect_b32 s50, s52, -2
s_mul_hi_u32 s63, s39, s4
s_mul_i32 s62, s39, s4
s_add_u32 s64, s64, s62
s_addc_u32 s65, s65, s63
s_lshl_b64 s[64:65], s[64:65], 1
s_add_u32 s48, s30, s64
s_addc_u32 s49, s31, s65
s_mov_b32 s51, 0x20000
s_add_u32 s77, 17, s76
s_cmp_lt_u32 s38, s74
s_cbranch_scc1 .L_fc2_tn_2476560
s_mov_b32 s75, 1
s_lshl_b32 s75, s75, s77
s_branch .L_fc2_tn_2476570
.L_fc2_tn_2476560:
s_lshr_b32 s78, s38, 13
s_lshl_b32 s75, s38, 1
s_lshr_b32 s75, s75, s78
s_lshl_b32 s75, s75, 16
.L_fc2_tn_2476570:
s_or_b32 s75, s75, 2.0
s_mov_b32 s90, s75
s_or_b32 s49, s49, s90
s_mul_i32 s54, s42, 0x80
s_mov_b32 s17, s54
s_mul_i32 s54, s42, 0x80
s_mov_b32 s43, s54
s_lshr_b32 s5, s23, 6
s_mov_b32 s6, s5
s_cmp_eq_u32 s5, 0
s_cbranch_scc1 .L_fc2_tn_2476688
s_cmp_le_i32 s73, 7
s_cbranch_scc1 .L_fc2_tn_2476688
s_mov_b32 s81, 0
.L_fc2_tn_24765b0:
s_mov_b32 m0, s92
buffer_load_dwordx4 v224, s[44:47], 0 offen lds
s_add_u32 m0, m0, 0x1000
buffer_load_dwordx4 v225, s[44:47], 0 offen lds
s_add_u32 m0, m0, 0x1000
buffer_load_dwordx4 v226, s[44:47], 0 offen lds
s_add_u32 m0, m0, 0x1000
buffer_load_dwordx4 v227, s[44:47], 0 offen lds
s_add_u32 m0, m0, 0x1000
buffer_load_dwordx4 v228, s[44:47], 0 offen lds
s_add_u32 m0, m0, 0x1000
buffer_load_dwordx4 v229, s[44:47], 0 offen lds
s_add_u32 m0, m0, 0x1000
buffer_load_dwordx4 v230, s[44:47], 0 offen lds
s_add_u32 m0, m0, 0x1000
buffer_load_dwordx4 v231, s[44:47], 0 offen lds
s_barrier
s_add_u32 s81, s81, 1
s_add_u32 s44, s44, s17
s_addc_u32 s45, s45, 0
s_sub_u32 s18, s18, s17
s_subb_u32 s19, s19, 0
s_cmp_eq_u32 s19, 0
s_cmov_b32 s46, s18
s_add_u32 s92, 0x8000, s92
s_and_b32 s90, s92, 0x1ffff
s_cmp_ge_u32 s90, s72
s_cmov_b32 s92, s80
s_cmp_lt_i32 s81, 1
s_cbranch_scc1 .L_fc2_tn_2476674
s_waitcnt vmcnt(0)
s_barrier
.L_fc2_tn_2476674:
s_cmp_lt_i32 s81, s5
s_cbranch_scc1 .L_fc2_tn_24765b0
s_waitcnt vmcnt(0)
s_barrier
s_endpgm
.L_fc2_tn_2476688:
s_mov_b32 s8, s24
s_mov_b32 s9, s25
s_mov_b32 s10, -2
s_mov_b32 s11, 0x20000
s_mov_b32 s12, s26
s_mov_b32 s13, s27
s_mov_b32 s14, -2
s_mov_b32 s15, 0x20000
s_mul_i32 s64, 0x100, s3
s_mul_hi_u32 s63, s64, s34
s_mul_i32 s62, s64, s34
s_lshl_b64 s[62:63], s[62:63], 1
s_add_u32 s12, s12, s62
s_addc_u32 s13, s13, s63
s_mul_hi_u32 s63, s64, s32
s_mul_i32 s62, s64, s32
s_lshl_b64 s[62:63], s[62:63], 1
s_add_u32 s8, s8, s62
s_addc_u32 s9, s9, s63
s_mul_hi_u32 s63, s4, s35
s_mul_i32 s62, s4, s35
s_lshl_b64 s[62:63], s[62:63], 1
s_add_u32 s12, s12, s62
s_addc_u32 s13, s13, s63
s_mul_hi_u32 s63, s4, s33
s_mul_i32 s62, s4, s33
s_lshl_b64 s[62:63], s[62:63], 1
s_add_u32 s8, s8, s62
s_addc_u32 s9, s9, s63
v_mov_b32_e32 v0, 0
v_mov_b32_e32 v1, 0
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 0
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v9, 0
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
v_mov_b32_e32 v14, 0
v_mov_b32_e32 v15, 0
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 0
v_mov_b32_e32 v18, 0
v_mov_b32_e32 v19, 0
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v21, 0
v_mov_b32_e32 v22, 0
v_mov_b32_e32 v23, 0
v_mov_b32_e32 v24, 0
v_mov_b32_e32 v25, 0
v_mov_b32_e32 v26, 0
v_mov_b32_e32 v27, 0
v_mov_b32_e32 v28, 0
v_mov_b32_e32 v29, 0
v_mov_b32_e32 v30, 0
v_mov_b32_e32 v31, 0
v_mov_b32_e32 v32, 0
v_mov_b32_e32 v33, 0
v_mov_b32_e32 v34, 0
v_mov_b32_e32 v35, 0
v_mov_b32_e32 v36, 0
v_mov_b32_e32 v37, 0
v_mov_b32_e32 v38, 0
v_mov_b32_e32 v39, 0
v_mov_b32_e32 v40, 0
v_mov_b32_e32 v41, 0
v_mov_b32_e32 v42, 0
v_mov_b32_e32 v43, 0
v_mov_b32_e32 v44, 0
v_mov_b32_e32 v45, 0
v_mov_b32_e32 v46, 0
v_mov_b32_e32 v47, 0
v_mov_b32_e32 v48, 0
v_mov_b32_e32 v49, 0
v_mov_b32_e32 v50, 0
v_mov_b32_e32 v51, 0
v_mov_b32_e32 v52, 0
v_mov_b32_e32 v53, 0
v_mov_b32_e32 v54, 0
v_mov_b32_e32 v55, 0
v_mov_b32_e32 v56, 0
v_mov_b32_e32 v57, 0
v_mov_b32_e32 v58, 0
v_mov_b32_e32 v59, 0
v_mov_b32_e32 v60, 0
v_mov_b32_e32 v61, 0
v_mov_b32_e32 v62, 0
v_mov_b32_e32 v63, 0
v_mov_b32_e32 v64, 0
v_mov_b32_e32 v65, 0
v_mov_b32_e32 v66, 0
v_mov_b32_e32 v67, 0
v_mov_b32_e32 v68, 0
v_mov_b32_e32 v69, 0
v_mov_b32_e32 v70, 0
v_mov_b32_e32 v71, 0
v_mov_b32_e32 v72, 0
v_mov_b32_e32 v73, 0
v_mov_b32_e32 v74, 0
v_mov_b32_e32 v75, 0
v_mov_b32_e32 v76, 0
v_mov_b32_e32 v77, 0
v_mov_b32_e32 v78, 0
v_mov_b32_e32 v79, 0
v_mov_b32_e32 v80, 0
v_mov_b32_e32 v81, 0
v_mov_b32_e32 v82, 0
v_mov_b32_e32 v83, 0
v_mov_b32_e32 v84, 0
v_mov_b32_e32 v85, 0
v_mov_b32_e32 v86, 0
v_mov_b32_e32 v87, 0
v_mov_b32_e32 v88, 0
v_mov_b32_e32 v89, 0
v_mov_b32_e32 v90, 0
v_mov_b32_e32 v91, 0
v_mov_b32_e32 v92, 0
v_mov_b32_e32 v93, 0
v_mov_b32_e32 v94, 0
v_mov_b32_e32 v95, 0
v_mov_b32_e32 v96, 0
v_mov_b32_e32 v97, 0
v_mov_b32_e32 v98, 0
v_mov_b32_e32 v99, 0
v_mov_b32_e32 v100, 0
v_mov_b32_e32 v101, 0
v_mov_b32_e32 v102, 0
v_mov_b32_e32 v103, 0
v_mov_b32_e32 v104, 0
v_mov_b32_e32 v105, 0
v_mov_b32_e32 v106, 0
v_mov_b32_e32 v107, 0
v_mov_b32_e32 v108, 0
v_mov_b32_e32 v109, 0
v_mov_b32_e32 v110, 0
v_mov_b32_e32 v111, 0
v_mov_b32_e32 v112, 0
v_mov_b32_e32 v113, 0
v_mov_b32_e32 v114, 0
v_mov_b32_e32 v115, 0
v_mov_b32_e32 v116, 0
v_mov_b32_e32 v117, 0
v_mov_b32_e32 v118, 0
v_mov_b32_e32 v119, 0
v_mov_b32_e32 v120, 0
v_mov_b32_e32 v121, 0
v_mov_b32_e32 v122, 0
v_mov_b32_e32 v123, 0
v_mov_b32_e32 v124, 0
v_mov_b32_e32 v125, 0
v_mov_b32_e32 v126, 0
v_mov_b32_e32 v127, 0
buffer_load_dwordx4 v[192:195], v[232:233], s[48:51], 0 idxen offen
buffer_load_dwordx4 v[196:199], v[234:235], s[48:51], 0 idxen offen
buffer_load_dwordx4 v[200:203], v[236:237], s[48:51], 0 idxen offen
buffer_load_dwordx4 v[204:207], v[238:239], s[48:51], 0 idxen offen
s_barrier
s_waitcnt vmcnt(0)
s_barrier
s_add_u32 s48, s48, s43
s_addc_u32 s49, s49, 0
s_cmp_le_i32 s5, 1
s_cmov_b32 s50, 0
s_cmov_b32 s43, 0
s_cmp_le_u32 s5, 0
s_cbranch_scc1 .L_fc2_tn_compute_end
s_and_b32 s90, 7, s73
s_cmp_ge_u32 s90, 4
s_cbranch_scc1 .L_fc2_tn_2477448
.L_fc2_tn_247697c:
ds_read_b128 v[128:131], v242
ds_read_b128 v[132:135], v244 offset:512
ds_read_b128 v[136:139], v242 offset:4096
ds_read_b128 v[140:143], v244 offset:4608
ds_read_b128 v[144:147], v242 offset:8192
ds_read_b128 v[148:151], v244 offset:8704
ds_read_b128 v[152:155], v242 offset:12288
ds_read_b128 v[156:159], v244 offset:12800
ds_read_b128 v[160:163], v242 offset:16384
ds_read_b128 v[164:167], v244 offset:16896
ds_read_b128 v[168:171], v242 offset:20480
ds_read_b128 v[172:175], v244 offset:20992
ds_read_b128 v[176:179], v242 offset:24576
ds_read_b128 v[180:183], v244 offset:25088
ds_read_b128 v[184:187], v242 offset:28672
ds_read_b128 v[188:191], v244 offset:29184
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_bf16 v[0:3], v[128:129], v[192:193], v[0:3]
s_setprio 1
v_mmac_f32_16x16x16_bf16 v[4:7], v[132:133], v[192:193], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[136:137], v[192:193], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[140:141], v[192:193], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[144:145], v[192:193], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[148:149], v[192:193], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[152:153], v[192:193], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[156:157], v[192:193], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[160:161], v[192:193], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[164:165], v[192:193], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[168:169], v[192:193], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[172:173], v[192:193], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[176:177], v[192:193], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[180:181], v[192:193], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[184:185], v[192:193], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[188:189], v[192:193], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[128:129], v[200:201], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[132:133], v[200:201], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[136:137], v[200:201], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[140:141], v[200:201], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[144:145], v[200:201], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[148:149], v[200:201], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[152:153], v[200:201], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[156:157], v[200:201], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[160:161], v[200:201], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[164:165], v[200:201], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[168:169], v[200:201], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[172:173], v[200:201], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[176:177], v[200:201], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[180:181], v[200:201], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[184:185], v[200:201], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[188:189], v[200:201], v[124:127]
v_mmac_f32_16x16x16_bf16 v[0:3], v[130:131], v[194:195], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[134:135], v[194:195], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[138:139], v[194:195], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[142:143], v[194:195], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[146:147], v[194:195], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[150:151], v[194:195], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[154:155], v[194:195], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[158:159], v[194:195], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[162:163], v[194:195], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[166:167], v[194:195], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[170:171], v[194:195], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[174:175], v[194:195], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[178:179], v[194:195], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[182:183], v[194:195], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[186:187], v[194:195], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[190:191], v[194:195], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[130:131], v[202:203], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[134:135], v[202:203], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[138:139], v[202:203], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[142:143], v[202:203], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[146:147], v[202:203], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[150:151], v[202:203], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[154:155], v[202:203], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[158:159], v[202:203], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[162:163], v[202:203], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[166:167], v[202:203], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[170:171], v[202:203], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[174:175], v[202:203], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[178:179], v[202:203], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[182:183], v[202:203], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[186:187], v[202:203], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[190:191], v[202:203], v[124:127]
s_setprio 0
s_barrier
ds_read_b128 v[128:131], v242 offset:64
ds_read_b128 v[132:135], v249 offset:576
ds_read_b128 v[136:139], v242 offset:4160
ds_read_b128 v[140:143], v249 offset:4672
ds_read_b128 v[144:147], v242 offset:8256
ds_read_b128 v[148:151], v249 offset:8768
ds_read_b128 v[152:155], v242 offset:12352
ds_read_b128 v[156:159], v249 offset:12864
buffer_load_dwordx4 v[208:211], v[232:233], s[48:51], 0 idxen offen
buffer_load_dwordx4 v[212:215], v[234:235], s[48:51], 0 idxen offen
buffer_load_dwordx4 v[216:219], v[236:237], s[48:51], 0 idxen offen
buffer_load_dwordx4 v[220:223], v[238:239], s[48:51], 0 idxen offen
ds_read_b128 v[160:163], v242 offset:16448
ds_read_b128 v[164:167], v249 offset:16960
ds_read_b128 v[168:171], v242 offset:20544
ds_read_b128 v[172:175], v249 offset:21056
ds_read_b128 v[176:179], v242 offset:24640
ds_read_b128 v[180:183], v249 offset:25152
ds_read_b128 v[184:187], v242 offset:28736
ds_read_b128 v[188:191], v249 offset:29248
s_add_u32 s48, s48, s43
s_addc_u32 s49, s49, 0
s_cmp_le_i32 s5, 2
s_cmov_b32 s50, 0
s_cmov_b32 s43, 0
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_bf16 v[0:3], v[128:129], v[196:197], v[0:3]
s_setprio 1
v_mmac_f32_16x16x16_bf16 v[4:7], v[132:133], v[196:197], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[136:137], v[196:197], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[140:141], v[196:197], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[144:145], v[196:197], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[148:149], v[196:197], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[152:153], v[196:197], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[156:157], v[196:197], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[160:161], v[196:197], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[164:165], v[196:197], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[168:169], v[196:197], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[172:173], v[196:197], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[176:177], v[196:197], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[180:181], v[196:197], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[184:185], v[196:197], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[188:189], v[196:197], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[128:129], v[204:205], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[132:133], v[204:205], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[136:137], v[204:205], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[140:141], v[204:205], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[144:145], v[204:205], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[148:149], v[204:205], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[152:153], v[204:205], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[156:157], v[204:205], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[160:161], v[204:205], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[164:165], v[204:205], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[168:169], v[204:205], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[172:173], v[204:205], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[176:177], v[204:205], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[180:181], v[204:205], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[184:185], v[204:205], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[188:189], v[204:205], v[124:127]
v_mmac_f32_16x16x16_bf16 v[0:3], v[130:131], v[198:199], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[134:135], v[198:199], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[138:139], v[198:199], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[142:143], v[198:199], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[146:147], v[198:199], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[150:151], v[198:199], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[154:155], v[198:199], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[158:159], v[198:199], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[162:163], v[198:199], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[166:167], v[198:199], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[170:171], v[198:199], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[174:175], v[198:199], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[178:179], v[198:199], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[182:183], v[198:199], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[186:187], v[198:199], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[190:191], v[198:199], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[130:131], v[206:207], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[134:135], v[206:207], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[138:139], v[206:207], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[142:143], v[206:207], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[146:147], v[206:207], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[150:151], v[206:207], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[154:155], v[206:207], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[158:159], v[206:207], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[162:163], v[206:207], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[166:167], v[206:207], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[170:171], v[206:207], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[174:175], v[206:207], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[178:179], v[206:207], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[182:183], v[206:207], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[186:187], v[206:207], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[190:191], v[206:207], v[124:127]
s_setprio 0
s_barrier
s_waitcnt vmcnt(0)
s_sub_u32 s5, s5, 1
s_cmp_eq_i32 s5, 0
s_cbranch_scc1 .L_fc2_tn_compute_end
ds_read_b128 v[128:131], v242 offset:32768
ds_read_b128 v[132:135], v244 offset:33280
ds_read_b128 v[136:139], v242 offset:36864
ds_read_b128 v[140:143], v244 offset:37376
ds_read_b128 v[144:147], v242 offset:40960
ds_read_b128 v[148:151], v244 offset:41472
ds_read_b128 v[152:155], v242 offset:45056
ds_read_b128 v[156:159], v244 offset:45568
ds_read_b128 v[160:163], v242 offset:49152
ds_read_b128 v[164:167], v244 offset:49664
ds_read_b128 v[168:171], v242 offset:53248
ds_read_b128 v[172:175], v244 offset:53760
ds_read_b128 v[176:179], v242 offset:57344
ds_read_b128 v[180:183], v244 offset:57856
ds_read_b128 v[184:187], v242 offset:61440
ds_read_b128 v[188:191], v244 offset:61952
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_bf16 v[0:3], v[128:129], v[208:209], v[0:3]
s_setprio 1
v_mmac_f32_16x16x16_bf16 v[4:7], v[132:133], v[208:209], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[136:137], v[208:209], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[140:141], v[208:209], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[144:145], v[208:209], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[148:149], v[208:209], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[152:153], v[208:209], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[156:157], v[208:209], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[160:161], v[208:209], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[164:165], v[208:209], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[168:169], v[208:209], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[172:173], v[208:209], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[176:177], v[208:209], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[180:181], v[208:209], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[184:185], v[208:209], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[188:189], v[208:209], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[128:129], v[216:217], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[132:133], v[216:217], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[136:137], v[216:217], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[140:141], v[216:217], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[144:145], v[216:217], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[148:149], v[216:217], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[152:153], v[216:217], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[156:157], v[216:217], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[160:161], v[216:217], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[164:165], v[216:217], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[168:169], v[216:217], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[172:173], v[216:217], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[176:177], v[216:217], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[180:181], v[216:217], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[184:185], v[216:217], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[188:189], v[216:217], v[124:127]
v_mmac_f32_16x16x16_bf16 v[0:3], v[130:131], v[210:211], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[134:135], v[210:211], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[138:139], v[210:211], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[142:143], v[210:211], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[146:147], v[210:211], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[150:151], v[210:211], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[154:155], v[210:211], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[158:159], v[210:211], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[162:163], v[210:211], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[166:167], v[210:211], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[170:171], v[210:211], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[174:175], v[210:211], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[178:179], v[210:211], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[182:183], v[210:211], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[186:187], v[210:211], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[190:191], v[210:211], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[130:131], v[218:219], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[134:135], v[218:219], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[138:139], v[218:219], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[142:143], v[218:219], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[146:147], v[218:219], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[150:151], v[218:219], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[154:155], v[218:219], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[158:159], v[218:219], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[162:163], v[218:219], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[166:167], v[218:219], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[170:171], v[218:219], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[174:175], v[218:219], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[178:179], v[218:219], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[182:183], v[218:219], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[186:187], v[218:219], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[190:191], v[218:219], v[124:127]
s_setprio 0
s_barrier
ds_read_b128 v[128:131], v242 offset:32832
ds_read_b128 v[132:135], v249 offset:33344
ds_read_b128 v[136:139], v242 offset:36928
ds_read_b128 v[140:143], v249 offset:37440
ds_read_b128 v[144:147], v242 offset:41024
ds_read_b128 v[148:151], v249 offset:41536
ds_read_b128 v[152:155], v242 offset:45120
ds_read_b128 v[156:159], v249 offset:45632
buffer_load_dwordx4 v[192:195], v[232:233], s[48:51], 0 idxen offen
buffer_load_dwordx4 v[196:199], v[234:235], s[48:51], 0 idxen offen
buffer_load_dwordx4 v[200:203], v[236:237], s[48:51], 0 idxen offen
buffer_load_dwordx4 v[204:207], v[238:239], s[48:51], 0 idxen offen
ds_read_b128 v[160:163], v242 offset:49216
ds_read_b128 v[164:167], v249 offset:49728
ds_read_b128 v[168:171], v242 offset:53312
ds_read_b128 v[172:175], v249 offset:53824
ds_read_b128 v[176:179], v242 offset:57408
ds_read_b128 v[180:183], v249 offset:57920
ds_read_b128 v[184:187], v242 offset:61504
ds_read_b128 v[188:191], v249 offset:62016
s_add_u32 s48, s48, s43
s_addc_u32 s49, s49, 0
s_cmp_le_i32 s5, 2
s_cmov_b32 s50, 0
s_cmov_b32 s43, 0
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_bf16 v[0:3], v[128:129], v[212:213], v[0:3]
s_setprio 1
v_mmac_f32_16x16x16_bf16 v[4:7], v[132:133], v[212:213], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[136:137], v[212:213], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[140:141], v[212:213], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[144:145], v[212:213], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[148:149], v[212:213], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[152:153], v[212:213], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[156:157], v[212:213], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[160:161], v[212:213], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[164:165], v[212:213], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[168:169], v[212:213], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[172:173], v[212:213], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[176:177], v[212:213], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[180:181], v[212:213], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[184:185], v[212:213], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[188:189], v[212:213], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[128:129], v[220:221], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[132:133], v[220:221], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[136:137], v[220:221], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[140:141], v[220:221], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[144:145], v[220:221], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[148:149], v[220:221], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[152:153], v[220:221], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[156:157], v[220:221], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[160:161], v[220:221], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[164:165], v[220:221], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[168:169], v[220:221], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[172:173], v[220:221], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[176:177], v[220:221], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[180:181], v[220:221], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[184:185], v[220:221], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[188:189], v[220:221], v[124:127]
v_mmac_f32_16x16x16_bf16 v[0:3], v[130:131], v[214:215], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[134:135], v[214:215], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[138:139], v[214:215], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[142:143], v[214:215], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[146:147], v[214:215], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[150:151], v[214:215], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[154:155], v[214:215], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[158:159], v[214:215], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[162:163], v[214:215], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[166:167], v[214:215], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[170:171], v[214:215], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[174:175], v[214:215], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[178:179], v[214:215], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[182:183], v[214:215], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[186:187], v[214:215], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[190:191], v[214:215], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[130:131], v[222:223], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[134:135], v[222:223], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[138:139], v[222:223], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[142:143], v[222:223], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[146:147], v[222:223], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[150:151], v[222:223], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[154:155], v[222:223], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[158:159], v[222:223], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[162:163], v[222:223], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[166:167], v[222:223], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[170:171], v[222:223], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[174:175], v[222:223], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[178:179], v[222:223], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[182:183], v[222:223], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[186:187], v[222:223], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[190:191], v[222:223], v[124:127]
s_setprio 0
s_barrier
s_waitcnt vmcnt(0)
s_sub_u32 s5, s5, 1
s_cmp_eq_i32 s5, 0
s_cbranch_scc0 .L_fc2_tn_247697c
s_branch .L_fc2_tn_compute_end
.L_fc2_tn_2477448:
ds_read_b128 v[128:131], v242
ds_read_b128 v[132:135], v244 offset:512
ds_read_b128 v[136:139], v242 offset:4096
ds_read_b128 v[140:143], v244 offset:4608
ds_read_b128 v[144:147], v242 offset:8192
ds_read_b128 v[148:151], v244 offset:8704
ds_read_b128 v[152:155], v242 offset:12288
ds_read_b128 v[156:159], v244 offset:12800
ds_read_b128 v[160:163], v242 offset:16384
ds_read_b128 v[164:167], v244 offset:16896
ds_read_b128 v[168:171], v242 offset:20480
ds_read_b128 v[172:175], v244 offset:20992
ds_read_b128 v[176:179], v242 offset:24576
ds_read_b128 v[180:183], v244 offset:25088
ds_read_b128 v[184:187], v242 offset:28672
ds_read_b128 v[188:191], v244 offset:29184
s_waitcnt lgkmcnt(0)
s_barrier
v_mmac_f32_16x16x16_bf16 v[0:3], v[128:129], v[192:193], v[0:3]
s_setprio 1
v_mmac_f32_16x16x16_bf16 v[4:7], v[132:133], v[192:193], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[136:137], v[192:193], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[140:141], v[192:193], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[144:145], v[192:193], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[148:149], v[192:193], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[152:153], v[192:193], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[156:157], v[192:193], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[160:161], v[192:193], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[164:165], v[192:193], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[168:169], v[192:193], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[172:173], v[192:193], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[176:177], v[192:193], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[180:181], v[192:193], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[184:185], v[192:193], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[188:189], v[192:193], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[128:129], v[200:201], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[132:133], v[200:201], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[136:137], v[200:201], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[140:141], v[200:201], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[144:145], v[200:201], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[148:149], v[200:201], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[152:153], v[200:201], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[156:157], v[200:201], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[160:161], v[200:201], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[164:165], v[200:201], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[168:169], v[200:201], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[172:173], v[200:201], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[176:177], v[200:201], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[180:181], v[200:201], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[184:185], v[200:201], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[188:189], v[200:201], v[124:127]
v_mmac_f32_16x16x16_bf16 v[0:3], v[130:131], v[194:195], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[134:135], v[194:195], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[138:139], v[194:195], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[142:143], v[194:195], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[146:147], v[194:195], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[150:151], v[194:195], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[154:155], v[194:195], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[158:159], v[194:195], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[162:163], v[194:195], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[166:167], v[194:195], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[170:171], v[194:195], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[174:175], v[194:195], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[178:179], v[194:195], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[182:183], v[194:195], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[186:187], v[194:195], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[190:191], v[194:195], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[130:131], v[202:203], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[134:135], v[202:203], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[138:139], v[202:203], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[142:143], v[202:203], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[146:147], v[202:203], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[150:151], v[202:203], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[154:155], v[202:203], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[158:159], v[202:203], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[162:163], v[202:203], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[166:167], v[202:203], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[170:171], v[202:203], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[174:175], v[202:203], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[178:179], v[202:203], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[182:183], v[202:203], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[186:187], v[202:203], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[190:191], v[202:203], v[124:127]
s_setprio 0
ds_read_b128 v[128:131], v242 offset:64
ds_read_b128 v[132:135], v249 offset:576
ds_read_b128 v[136:139], v242 offset:4160
ds_read_b128 v[140:143], v249 offset:4672
ds_read_b128 v[144:147], v242 offset:8256
ds_read_b128 v[148:151], v249 offset:8768
ds_read_b128 v[152:155], v242 offset:12352
ds_read_b128 v[156:159], v249 offset:12864
buffer_load_dwordx4 v[208:211], v[232:233], s[48:51], 0 idxen offen
buffer_load_dwordx4 v[212:215], v[234:235], s[48:51], 0 idxen offen
buffer_load_dwordx4 v[216:219], v[236:237], s[48:51], 0 idxen offen
buffer_load_dwordx4 v[220:223], v[238:239], s[48:51], 0 idxen offen
ds_read_b128 v[160:163], v242 offset:16448
ds_read_b128 v[164:167], v249 offset:16960
ds_read_b128 v[168:171], v242 offset:20544
ds_read_b128 v[172:175], v249 offset:21056
ds_read_b128 v[176:179], v242 offset:24640
ds_read_b128 v[180:183], v249 offset:25152
ds_read_b128 v[184:187], v242 offset:28736
ds_read_b128 v[188:191], v249 offset:29248
s_add_u32 s48, s48, s43
s_addc_u32 s49, s49, 0
s_cmp_le_i32 s5, 2
s_cmov_b32 s50, 0
s_cmov_b32 s43, 0
s_waitcnt lgkmcnt(0)
s_barrier
v_mmac_f32_16x16x16_bf16 v[0:3], v[128:129], v[196:197], v[0:3]
s_setprio 1
v_mmac_f32_16x16x16_bf16 v[4:7], v[132:133], v[196:197], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[136:137], v[196:197], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[140:141], v[196:197], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[144:145], v[196:197], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[148:149], v[196:197], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[152:153], v[196:197], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[156:157], v[196:197], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[160:161], v[196:197], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[164:165], v[196:197], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[168:169], v[196:197], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[172:173], v[196:197], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[176:177], v[196:197], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[180:181], v[196:197], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[184:185], v[196:197], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[188:189], v[196:197], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[128:129], v[204:205], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[132:133], v[204:205], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[136:137], v[204:205], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[140:141], v[204:205], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[144:145], v[204:205], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[148:149], v[204:205], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[152:153], v[204:205], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[156:157], v[204:205], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[160:161], v[204:205], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[164:165], v[204:205], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[168:169], v[204:205], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[172:173], v[204:205], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[176:177], v[204:205], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[180:181], v[204:205], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[184:185], v[204:205], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[188:189], v[204:205], v[124:127]
v_mmac_f32_16x16x16_bf16 v[0:3], v[130:131], v[198:199], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[134:135], v[198:199], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[138:139], v[198:199], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[142:143], v[198:199], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[146:147], v[198:199], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[150:151], v[198:199], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[154:155], v[198:199], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[158:159], v[198:199], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[162:163], v[198:199], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[166:167], v[198:199], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[170:171], v[198:199], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[174:175], v[198:199], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[178:179], v[198:199], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[182:183], v[198:199], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[186:187], v[198:199], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[190:191], v[198:199], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[130:131], v[206:207], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[134:135], v[206:207], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[138:139], v[206:207], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[142:143], v[206:207], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[146:147], v[206:207], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[150:151], v[206:207], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[154:155], v[206:207], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[158:159], v[206:207], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[162:163], v[206:207], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[166:167], v[206:207], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[170:171], v[206:207], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[174:175], v[206:207], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[178:179], v[206:207], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[182:183], v[206:207], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[186:187], v[206:207], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[190:191], v[206:207], v[124:127]
s_setprio 0
s_waitcnt vmcnt(0)
s_sub_u32 s5, s5, 1
s_cmp_eq_i32 s5, 0
s_cbranch_scc1 .L_fc2_tn_compute_end
ds_read_b128 v[128:131], v242 offset:32768
ds_read_b128 v[132:135], v244 offset:33280
ds_read_b128 v[136:139], v242 offset:36864
ds_read_b128 v[140:143], v244 offset:37376
ds_read_b128 v[144:147], v242 offset:40960
ds_read_b128 v[148:151], v244 offset:41472
ds_read_b128 v[152:155], v242 offset:45056
ds_read_b128 v[156:159], v244 offset:45568
ds_read_b128 v[160:163], v242 offset:49152
ds_read_b128 v[164:167], v244 offset:49664
ds_read_b128 v[168:171], v242 offset:53248
ds_read_b128 v[172:175], v244 offset:53760
ds_read_b128 v[176:179], v242 offset:57344
ds_read_b128 v[180:183], v244 offset:57856
ds_read_b128 v[184:187], v242 offset:61440
ds_read_b128 v[188:191], v244 offset:61952
s_waitcnt lgkmcnt(0)
s_barrier
v_mmac_f32_16x16x16_bf16 v[0:3], v[128:129], v[208:209], v[0:3]
s_setprio 1
v_mmac_f32_16x16x16_bf16 v[4:7], v[132:133], v[208:209], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[136:137], v[208:209], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[140:141], v[208:209], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[144:145], v[208:209], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[148:149], v[208:209], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[152:153], v[208:209], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[156:157], v[208:209], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[160:161], v[208:209], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[164:165], v[208:209], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[168:169], v[208:209], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[172:173], v[208:209], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[176:177], v[208:209], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[180:181], v[208:209], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[184:185], v[208:209], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[188:189], v[208:209], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[128:129], v[216:217], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[132:133], v[216:217], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[136:137], v[216:217], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[140:141], v[216:217], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[144:145], v[216:217], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[148:149], v[216:217], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[152:153], v[216:217], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[156:157], v[216:217], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[160:161], v[216:217], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[164:165], v[216:217], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[168:169], v[216:217], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[172:173], v[216:217], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[176:177], v[216:217], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[180:181], v[216:217], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[184:185], v[216:217], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[188:189], v[216:217], v[124:127]
v_mmac_f32_16x16x16_bf16 v[0:3], v[130:131], v[210:211], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[134:135], v[210:211], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[138:139], v[210:211], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[142:143], v[210:211], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[146:147], v[210:211], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[150:151], v[210:211], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[154:155], v[210:211], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[158:159], v[210:211], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[162:163], v[210:211], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[166:167], v[210:211], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[170:171], v[210:211], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[174:175], v[210:211], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[178:179], v[210:211], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[182:183], v[210:211], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[186:187], v[210:211], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[190:191], v[210:211], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[130:131], v[218:219], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[134:135], v[218:219], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[138:139], v[218:219], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[142:143], v[218:219], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[146:147], v[218:219], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[150:151], v[218:219], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[154:155], v[218:219], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[158:159], v[218:219], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[162:163], v[218:219], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[166:167], v[218:219], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[170:171], v[218:219], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[174:175], v[218:219], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[178:179], v[218:219], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[182:183], v[218:219], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[186:187], v[218:219], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[190:191], v[218:219], v[124:127]
s_setprio 0
ds_read_b128 v[128:131], v242 offset:32832
ds_read_b128 v[132:135], v249 offset:33344
ds_read_b128 v[136:139], v242 offset:36928
ds_read_b128 v[140:143], v249 offset:37440
ds_read_b128 v[144:147], v242 offset:41024
ds_read_b128 v[148:151], v249 offset:41536
ds_read_b128 v[152:155], v242 offset:45120
ds_read_b128 v[156:159], v249 offset:45632
buffer_load_dwordx4 v[192:195], v[232:233], s[48:51], 0 idxen offen
buffer_load_dwordx4 v[196:199], v[234:235], s[48:51], 0 idxen offen
buffer_load_dwordx4 v[200:203], v[236:237], s[48:51], 0 idxen offen
buffer_load_dwordx4 v[204:207], v[238:239], s[48:51], 0 idxen offen
ds_read_b128 v[160:163], v242 offset:49216
ds_read_b128 v[164:167], v249 offset:49728
ds_read_b128 v[168:171], v242 offset:53312
ds_read_b128 v[172:175], v249 offset:53824
ds_read_b128 v[176:179], v242 offset:57408
ds_read_b128 v[180:183], v249 offset:57920
ds_read_b128 v[184:187], v242 offset:61504
ds_read_b128 v[188:191], v249 offset:62016
s_add_u32 s48, s48, s43
s_addc_u32 s49, s49, 0
s_cmp_le_i32 s5, 2
s_cmov_b32 s50, 0
s_cmov_b32 s43, 0
s_waitcnt lgkmcnt(0)
s_barrier
v_mmac_f32_16x16x16_bf16 v[0:3], v[128:129], v[212:213], v[0:3]
s_setprio 1
v_mmac_f32_16x16x16_bf16 v[4:7], v[132:133], v[212:213], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[136:137], v[212:213], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[140:141], v[212:213], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[144:145], v[212:213], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[148:149], v[212:213], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[152:153], v[212:213], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[156:157], v[212:213], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[160:161], v[212:213], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[164:165], v[212:213], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[168:169], v[212:213], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[172:173], v[212:213], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[176:177], v[212:213], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[180:181], v[212:213], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[184:185], v[212:213], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[188:189], v[212:213], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[128:129], v[220:221], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[132:133], v[220:221], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[136:137], v[220:221], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[140:141], v[220:221], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[144:145], v[220:221], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[148:149], v[220:221], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[152:153], v[220:221], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[156:157], v[220:221], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[160:161], v[220:221], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[164:165], v[220:221], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[168:169], v[220:221], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[172:173], v[220:221], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[176:177], v[220:221], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[180:181], v[220:221], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[184:185], v[220:221], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[188:189], v[220:221], v[124:127]
v_mmac_f32_16x16x16_bf16 v[0:3], v[130:131], v[214:215], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[134:135], v[214:215], v[4:7]
v_mmac_f32_16x16x16_bf16 v[8:11], v[138:139], v[214:215], v[8:11]
v_mmac_f32_16x16x16_bf16 v[12:15], v[142:143], v[214:215], v[12:15]
v_mmac_f32_16x16x16_bf16 v[16:19], v[146:147], v[214:215], v[16:19]
v_mmac_f32_16x16x16_bf16 v[20:23], v[150:151], v[214:215], v[20:23]
v_mmac_f32_16x16x16_bf16 v[24:27], v[154:155], v[214:215], v[24:27]
v_mmac_f32_16x16x16_bf16 v[28:31], v[158:159], v[214:215], v[28:31]
v_mmac_f32_16x16x16_bf16 v[32:35], v[162:163], v[214:215], v[32:35]
v_mmac_f32_16x16x16_bf16 v[36:39], v[166:167], v[214:215], v[36:39]
v_mmac_f32_16x16x16_bf16 v[40:43], v[170:171], v[214:215], v[40:43]
v_mmac_f32_16x16x16_bf16 v[44:47], v[174:175], v[214:215], v[44:47]
v_mmac_f32_16x16x16_bf16 v[48:51], v[178:179], v[214:215], v[48:51]
v_mmac_f32_16x16x16_bf16 v[52:55], v[182:183], v[214:215], v[52:55]
v_mmac_f32_16x16x16_bf16 v[56:59], v[186:187], v[214:215], v[56:59]
v_mmac_f32_16x16x16_bf16 v[60:63], v[190:191], v[214:215], v[60:63]
v_mmac_f32_16x16x16_bf16 v[64:67], v[130:131], v[222:223], v[64:67]
v_mmac_f32_16x16x16_bf16 v[68:71], v[134:135], v[222:223], v[68:71]
v_mmac_f32_16x16x16_bf16 v[72:75], v[138:139], v[222:223], v[72:75]
v_mmac_f32_16x16x16_bf16 v[76:79], v[142:143], v[222:223], v[76:79]
v_mmac_f32_16x16x16_bf16 v[80:83], v[146:147], v[222:223], v[80:83]
v_mmac_f32_16x16x16_bf16 v[84:87], v[150:151], v[222:223], v[84:87]
v_mmac_f32_16x16x16_bf16 v[88:91], v[154:155], v[222:223], v[88:91]
v_mmac_f32_16x16x16_bf16 v[92:95], v[158:159], v[222:223], v[92:95]
v_mmac_f32_16x16x16_bf16 v[96:99], v[162:163], v[222:223], v[96:99]
v_mmac_f32_16x16x16_bf16 v[100:103], v[166:167], v[222:223], v[100:103]
v_mmac_f32_16x16x16_bf16 v[104:107], v[170:171], v[222:223], v[104:107]
v_mmac_f32_16x16x16_bf16 v[108:111], v[174:175], v[222:223], v[108:111]
v_mmac_f32_16x16x16_bf16 v[112:115], v[178:179], v[222:223], v[112:115]
v_mmac_f32_16x16x16_bf16 v[116:119], v[182:183], v[222:223], v[116:119]
v_mmac_f32_16x16x16_bf16 v[120:123], v[186:187], v[222:223], v[120:123]
v_mmac_f32_16x16x16_bf16 v[124:127], v[190:191], v[222:223], v[124:127]
s_setprio 0
s_waitcnt vmcnt(0)
s_sub_u32 s5, s5, 1
s_cmp_eq_i32 s5, 0
s_cbranch_scc0 .L_fc2_tn_2477448

.L_fc2_tn_compute_end:
s_waitcnt vmcnt(0) lgkmcnt(0)
s_barrier
s_mov_b32 s[sgprStrideD1J], s61
s_mov_b32 s[sgprSizeJ], s93
s_mov_b32 s[sgprSizeI], s91
s_mov_b64 s[sgprAddressD:sgprAddressD+1], s[94:95]
s_mov_b32 s77, s22
v_mov_b32 v202, s98
v_mov_b32 v[vgprSerial], v246
v_and_b32 v128, 15, v246
v_lshrrev_b32 v129, 6, v246
v_lshlrev_b32 v129, 5, v129
v_and_b32 v134, 48, v246
v_lshrrev_b32 v134, 4, v134
v_add_u32 v129, v129, v134
s_mov_b64 s[sgprSrdD:sgprSrdD+1], s[sgprAddressD:sgprAddressD+1]
s_mov_b32 s[sgprSrdD+2], BufferLimit
s_mov_b32 s[sgprSrdD+3], Srd127_96
v_mov_b32 v206, 0x7fff0000
v_mov_b32 v207, 0x7fff
s_load_dwordx2 s[sgprSrdC:sgprSrdC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_mov_b32 s[sgprSrdC+2], BufferLimit
s_mov_b32 s[sgprSrdC+3], Srd127_96
v_readfirstlane_b32 s7, v202
s_waitcnt lgkmcnt(0)
PREFETCH_ROWMAP_D 180, 0
PREFETCH_ROWMAP_D 181, 4
PREFETCH_ROWMAP_D 182, 8
PREFETCH_ROWMAP_D 183, 12
PREFETCH_ROWMAP_D 184, 16
PREFETCH_ROWMAP_D 185, 20
PREFETCH_ROWMAP_D 186, 24
PREFETCH_ROWMAP_D 187, 28
s_waitcnt vmcnt(0)
s_bitcmp1_b32 s77, 1
s_cbranch_scc1 .L_fc2_tn_buffer_output
GLOBAL_ROWMAP_D_ROW_BASE 180, 0
DIRECT_ROWMAP_D_STORE 0, 0
DIRECT_ROWMAP_D_STORE 4, 16
DIRECT_ROWMAP_D_STORE 8, 32
DIRECT_ROWMAP_D_STORE 12, 48
DIRECT_ROWMAP_D_STORE 16, 64
DIRECT_ROWMAP_D_STORE 20, 80
DIRECT_ROWMAP_D_STORE 24, 96
DIRECT_ROWMAP_D_STORE 28, 112
DIRECT_ROWMAP_D_STORE 32, 128
DIRECT_ROWMAP_D_STORE 36, 144
DIRECT_ROWMAP_D_STORE 40, 160
DIRECT_ROWMAP_D_STORE 44, 176
DIRECT_ROWMAP_D_STORE 48, 192
DIRECT_ROWMAP_D_STORE 52, 208
DIRECT_ROWMAP_D_STORE 56, 224
DIRECT_ROWMAP_D_STORE 60, 240
GLOBAL_ROWMAP_D_ROW_BASE 181, 4
DIRECT_ROWMAP_D_STORE 1, 0
DIRECT_ROWMAP_D_STORE 5, 16
DIRECT_ROWMAP_D_STORE 9, 32
DIRECT_ROWMAP_D_STORE 13, 48
DIRECT_ROWMAP_D_STORE 17, 64
DIRECT_ROWMAP_D_STORE 21, 80
DIRECT_ROWMAP_D_STORE 25, 96
DIRECT_ROWMAP_D_STORE 29, 112
DIRECT_ROWMAP_D_STORE 33, 128
DIRECT_ROWMAP_D_STORE 37, 144
DIRECT_ROWMAP_D_STORE 41, 160
DIRECT_ROWMAP_D_STORE 45, 176
DIRECT_ROWMAP_D_STORE 49, 192
DIRECT_ROWMAP_D_STORE 53, 208
DIRECT_ROWMAP_D_STORE 57, 224
DIRECT_ROWMAP_D_STORE 61, 240
GLOBAL_ROWMAP_D_ROW_BASE 182, 8
DIRECT_ROWMAP_D_STORE 2, 0
DIRECT_ROWMAP_D_STORE 6, 16
DIRECT_ROWMAP_D_STORE 10, 32
DIRECT_ROWMAP_D_STORE 14, 48
DIRECT_ROWMAP_D_STORE 18, 64
DIRECT_ROWMAP_D_STORE 22, 80
DIRECT_ROWMAP_D_STORE 26, 96
DIRECT_ROWMAP_D_STORE 30, 112
DIRECT_ROWMAP_D_STORE 34, 128
DIRECT_ROWMAP_D_STORE 38, 144
DIRECT_ROWMAP_D_STORE 42, 160
DIRECT_ROWMAP_D_STORE 46, 176
DIRECT_ROWMAP_D_STORE 50, 192
DIRECT_ROWMAP_D_STORE 54, 208
DIRECT_ROWMAP_D_STORE 58, 224
DIRECT_ROWMAP_D_STORE 62, 240
GLOBAL_ROWMAP_D_ROW_BASE 183, 12
DIRECT_ROWMAP_D_STORE 3, 0
DIRECT_ROWMAP_D_STORE 7, 16
DIRECT_ROWMAP_D_STORE 11, 32
DIRECT_ROWMAP_D_STORE 15, 48
DIRECT_ROWMAP_D_STORE 19, 64
DIRECT_ROWMAP_D_STORE 23, 80
DIRECT_ROWMAP_D_STORE 27, 96
DIRECT_ROWMAP_D_STORE 31, 112
DIRECT_ROWMAP_D_STORE 35, 128
DIRECT_ROWMAP_D_STORE 39, 144
DIRECT_ROWMAP_D_STORE 43, 160
DIRECT_ROWMAP_D_STORE 47, 176
DIRECT_ROWMAP_D_STORE 51, 192
DIRECT_ROWMAP_D_STORE 55, 208
DIRECT_ROWMAP_D_STORE 59, 224
DIRECT_ROWMAP_D_STORE 63, 240
GLOBAL_ROWMAP_D_ROW_BASE 184, 16
DIRECT_ROWMAP_D_STORE 64, 0
DIRECT_ROWMAP_D_STORE 68, 16
DIRECT_ROWMAP_D_STORE 72, 32
DIRECT_ROWMAP_D_STORE 76, 48
DIRECT_ROWMAP_D_STORE 80, 64
DIRECT_ROWMAP_D_STORE 84, 80
DIRECT_ROWMAP_D_STORE 88, 96
DIRECT_ROWMAP_D_STORE 92, 112
DIRECT_ROWMAP_D_STORE 96, 128
DIRECT_ROWMAP_D_STORE 100, 144
DIRECT_ROWMAP_D_STORE 104, 160
DIRECT_ROWMAP_D_STORE 108, 176
DIRECT_ROWMAP_D_STORE 112, 192
DIRECT_ROWMAP_D_STORE 116, 208
DIRECT_ROWMAP_D_STORE 120, 224
DIRECT_ROWMAP_D_STORE 124, 240
GLOBAL_ROWMAP_D_ROW_BASE 185, 20
DIRECT_ROWMAP_D_STORE 65, 0
DIRECT_ROWMAP_D_STORE 69, 16
DIRECT_ROWMAP_D_STORE 73, 32
DIRECT_ROWMAP_D_STORE 77, 48
DIRECT_ROWMAP_D_STORE 81, 64
DIRECT_ROWMAP_D_STORE 85, 80
DIRECT_ROWMAP_D_STORE 89, 96
DIRECT_ROWMAP_D_STORE 93, 112
DIRECT_ROWMAP_D_STORE 97, 128
DIRECT_ROWMAP_D_STORE 101, 144
DIRECT_ROWMAP_D_STORE 105, 160
DIRECT_ROWMAP_D_STORE 109, 176
DIRECT_ROWMAP_D_STORE 113, 192
DIRECT_ROWMAP_D_STORE 117, 208
DIRECT_ROWMAP_D_STORE 121, 224
DIRECT_ROWMAP_D_STORE 125, 240
GLOBAL_ROWMAP_D_ROW_BASE 186, 24
DIRECT_ROWMAP_D_STORE 66, 0
DIRECT_ROWMAP_D_STORE 70, 16
DIRECT_ROWMAP_D_STORE 74, 32
DIRECT_ROWMAP_D_STORE 78, 48
DIRECT_ROWMAP_D_STORE 82, 64
DIRECT_ROWMAP_D_STORE 86, 80
DIRECT_ROWMAP_D_STORE 90, 96
DIRECT_ROWMAP_D_STORE 94, 112
DIRECT_ROWMAP_D_STORE 98, 128
DIRECT_ROWMAP_D_STORE 102, 144
DIRECT_ROWMAP_D_STORE 106, 160
DIRECT_ROWMAP_D_STORE 110, 176
DIRECT_ROWMAP_D_STORE 114, 192
DIRECT_ROWMAP_D_STORE 118, 208
DIRECT_ROWMAP_D_STORE 122, 224
DIRECT_ROWMAP_D_STORE 126, 240
GLOBAL_ROWMAP_D_ROW_BASE 187, 28
DIRECT_ROWMAP_D_STORE 67, 0
DIRECT_ROWMAP_D_STORE 71, 16
DIRECT_ROWMAP_D_STORE 75, 32
DIRECT_ROWMAP_D_STORE 79, 48
DIRECT_ROWMAP_D_STORE 83, 64
DIRECT_ROWMAP_D_STORE 87, 80
DIRECT_ROWMAP_D_STORE 91, 96
DIRECT_ROWMAP_D_STORE 95, 112
DIRECT_ROWMAP_D_STORE 99, 128
DIRECT_ROWMAP_D_STORE 103, 144
DIRECT_ROWMAP_D_STORE 107, 160
DIRECT_ROWMAP_D_STORE 111, 176
DIRECT_ROWMAP_D_STORE 115, 192
DIRECT_ROWMAP_D_STORE 119, 208
DIRECT_ROWMAP_D_STORE 123, 224
DIRECT_ROWMAP_D_STORE 127, 240
s_branch label_0111
.L_fc2_tn_buffer_output:
BUFFER_ROWMAP_D_ROW_BASE 180, 0
DIRECT_ROWMAP_D_BUFFER_STORE 0, 0
DIRECT_ROWMAP_D_BUFFER_STORE 4, 16
DIRECT_ROWMAP_D_BUFFER_STORE 8, 32
DIRECT_ROWMAP_D_BUFFER_STORE 12, 48
DIRECT_ROWMAP_D_BUFFER_STORE 16, 64
DIRECT_ROWMAP_D_BUFFER_STORE 20, 80
DIRECT_ROWMAP_D_BUFFER_STORE 24, 96
DIRECT_ROWMAP_D_BUFFER_STORE 28, 112
DIRECT_ROWMAP_D_BUFFER_STORE 32, 128
DIRECT_ROWMAP_D_BUFFER_STORE 36, 144
DIRECT_ROWMAP_D_BUFFER_STORE 40, 160
DIRECT_ROWMAP_D_BUFFER_STORE 44, 176
DIRECT_ROWMAP_D_BUFFER_STORE 48, 192
DIRECT_ROWMAP_D_BUFFER_STORE 52, 208
DIRECT_ROWMAP_D_BUFFER_STORE 56, 224
DIRECT_ROWMAP_D_BUFFER_STORE 60, 240
BUFFER_ROWMAP_D_ROW_BASE 181, 4
DIRECT_ROWMAP_D_BUFFER_STORE 1, 0
DIRECT_ROWMAP_D_BUFFER_STORE 5, 16
DIRECT_ROWMAP_D_BUFFER_STORE 9, 32
DIRECT_ROWMAP_D_BUFFER_STORE 13, 48
DIRECT_ROWMAP_D_BUFFER_STORE 17, 64
DIRECT_ROWMAP_D_BUFFER_STORE 21, 80
DIRECT_ROWMAP_D_BUFFER_STORE 25, 96
DIRECT_ROWMAP_D_BUFFER_STORE 29, 112
DIRECT_ROWMAP_D_BUFFER_STORE 33, 128
DIRECT_ROWMAP_D_BUFFER_STORE 37, 144
DIRECT_ROWMAP_D_BUFFER_STORE 41, 160
DIRECT_ROWMAP_D_BUFFER_STORE 45, 176
DIRECT_ROWMAP_D_BUFFER_STORE 49, 192
DIRECT_ROWMAP_D_BUFFER_STORE 53, 208
DIRECT_ROWMAP_D_BUFFER_STORE 57, 224
DIRECT_ROWMAP_D_BUFFER_STORE 61, 240
BUFFER_ROWMAP_D_ROW_BASE 182, 8
DIRECT_ROWMAP_D_BUFFER_STORE 2, 0
DIRECT_ROWMAP_D_BUFFER_STORE 6, 16
DIRECT_ROWMAP_D_BUFFER_STORE 10, 32
DIRECT_ROWMAP_D_BUFFER_STORE 14, 48
DIRECT_ROWMAP_D_BUFFER_STORE 18, 64
DIRECT_ROWMAP_D_BUFFER_STORE 22, 80
DIRECT_ROWMAP_D_BUFFER_STORE 26, 96
DIRECT_ROWMAP_D_BUFFER_STORE 30, 112
DIRECT_ROWMAP_D_BUFFER_STORE 34, 128
DIRECT_ROWMAP_D_BUFFER_STORE 38, 144
DIRECT_ROWMAP_D_BUFFER_STORE 42, 160
DIRECT_ROWMAP_D_BUFFER_STORE 46, 176
DIRECT_ROWMAP_D_BUFFER_STORE 50, 192
DIRECT_ROWMAP_D_BUFFER_STORE 54, 208
DIRECT_ROWMAP_D_BUFFER_STORE 58, 224
DIRECT_ROWMAP_D_BUFFER_STORE 62, 240
BUFFER_ROWMAP_D_ROW_BASE 183, 12
DIRECT_ROWMAP_D_BUFFER_STORE 3, 0
DIRECT_ROWMAP_D_BUFFER_STORE 7, 16
DIRECT_ROWMAP_D_BUFFER_STORE 11, 32
DIRECT_ROWMAP_D_BUFFER_STORE 15, 48
DIRECT_ROWMAP_D_BUFFER_STORE 19, 64
DIRECT_ROWMAP_D_BUFFER_STORE 23, 80
DIRECT_ROWMAP_D_BUFFER_STORE 27, 96
DIRECT_ROWMAP_D_BUFFER_STORE 31, 112
DIRECT_ROWMAP_D_BUFFER_STORE 35, 128
DIRECT_ROWMAP_D_BUFFER_STORE 39, 144
DIRECT_ROWMAP_D_BUFFER_STORE 43, 160
DIRECT_ROWMAP_D_BUFFER_STORE 47, 176
DIRECT_ROWMAP_D_BUFFER_STORE 51, 192
DIRECT_ROWMAP_D_BUFFER_STORE 55, 208
DIRECT_ROWMAP_D_BUFFER_STORE 59, 224
DIRECT_ROWMAP_D_BUFFER_STORE 63, 240
BUFFER_ROWMAP_D_ROW_BASE 184, 16
DIRECT_ROWMAP_D_BUFFER_STORE 64, 0
DIRECT_ROWMAP_D_BUFFER_STORE 68, 16
DIRECT_ROWMAP_D_BUFFER_STORE 72, 32
DIRECT_ROWMAP_D_BUFFER_STORE 76, 48
DIRECT_ROWMAP_D_BUFFER_STORE 80, 64
DIRECT_ROWMAP_D_BUFFER_STORE 84, 80
DIRECT_ROWMAP_D_BUFFER_STORE 88, 96
DIRECT_ROWMAP_D_BUFFER_STORE 92, 112
DIRECT_ROWMAP_D_BUFFER_STORE 96, 128
DIRECT_ROWMAP_D_BUFFER_STORE 100, 144
DIRECT_ROWMAP_D_BUFFER_STORE 104, 160
DIRECT_ROWMAP_D_BUFFER_STORE 108, 176
DIRECT_ROWMAP_D_BUFFER_STORE 112, 192
DIRECT_ROWMAP_D_BUFFER_STORE 116, 208
DIRECT_ROWMAP_D_BUFFER_STORE 120, 224
DIRECT_ROWMAP_D_BUFFER_STORE 124, 240
BUFFER_ROWMAP_D_ROW_BASE 185, 20
DIRECT_ROWMAP_D_BUFFER_STORE 65, 0
DIRECT_ROWMAP_D_BUFFER_STORE 69, 16
DIRECT_ROWMAP_D_BUFFER_STORE 73, 32
DIRECT_ROWMAP_D_BUFFER_STORE 77, 48
DIRECT_ROWMAP_D_BUFFER_STORE 81, 64
DIRECT_ROWMAP_D_BUFFER_STORE 85, 80
DIRECT_ROWMAP_D_BUFFER_STORE 89, 96
DIRECT_ROWMAP_D_BUFFER_STORE 93, 112
DIRECT_ROWMAP_D_BUFFER_STORE 97, 128
DIRECT_ROWMAP_D_BUFFER_STORE 101, 144
DIRECT_ROWMAP_D_BUFFER_STORE 105, 160
DIRECT_ROWMAP_D_BUFFER_STORE 109, 176
DIRECT_ROWMAP_D_BUFFER_STORE 113, 192
DIRECT_ROWMAP_D_BUFFER_STORE 117, 208
DIRECT_ROWMAP_D_BUFFER_STORE 121, 224
DIRECT_ROWMAP_D_BUFFER_STORE 125, 240
BUFFER_ROWMAP_D_ROW_BASE 186, 24
DIRECT_ROWMAP_D_BUFFER_STORE 66, 0
DIRECT_ROWMAP_D_BUFFER_STORE 70, 16
DIRECT_ROWMAP_D_BUFFER_STORE 74, 32
DIRECT_ROWMAP_D_BUFFER_STORE 78, 48
DIRECT_ROWMAP_D_BUFFER_STORE 82, 64
DIRECT_ROWMAP_D_BUFFER_STORE 86, 80
DIRECT_ROWMAP_D_BUFFER_STORE 90, 96
DIRECT_ROWMAP_D_BUFFER_STORE 94, 112
DIRECT_ROWMAP_D_BUFFER_STORE 98, 128
DIRECT_ROWMAP_D_BUFFER_STORE 102, 144
DIRECT_ROWMAP_D_BUFFER_STORE 106, 160
DIRECT_ROWMAP_D_BUFFER_STORE 110, 176
DIRECT_ROWMAP_D_BUFFER_STORE 114, 192
DIRECT_ROWMAP_D_BUFFER_STORE 118, 208
DIRECT_ROWMAP_D_BUFFER_STORE 122, 224
DIRECT_ROWMAP_D_BUFFER_STORE 126, 240
BUFFER_ROWMAP_D_ROW_BASE 187, 28
DIRECT_ROWMAP_D_BUFFER_STORE 67, 0
DIRECT_ROWMAP_D_BUFFER_STORE 71, 16
DIRECT_ROWMAP_D_BUFFER_STORE 75, 32
DIRECT_ROWMAP_D_BUFFER_STORE 79, 48
DIRECT_ROWMAP_D_BUFFER_STORE 83, 64
DIRECT_ROWMAP_D_BUFFER_STORE 87, 80
DIRECT_ROWMAP_D_BUFFER_STORE 91, 96
DIRECT_ROWMAP_D_BUFFER_STORE 95, 112
DIRECT_ROWMAP_D_BUFFER_STORE 99, 128
DIRECT_ROWMAP_D_BUFFER_STORE 103, 144
DIRECT_ROWMAP_D_BUFFER_STORE 107, 160
DIRECT_ROWMAP_D_BUFFER_STORE 111, 176
DIRECT_ROWMAP_D_BUFFER_STORE 115, 192
DIRECT_ROWMAP_D_BUFFER_STORE 119, 208
DIRECT_ROWMAP_D_BUFFER_STORE 123, 224
DIRECT_ROWMAP_D_BUFFER_STORE 127, 240
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
