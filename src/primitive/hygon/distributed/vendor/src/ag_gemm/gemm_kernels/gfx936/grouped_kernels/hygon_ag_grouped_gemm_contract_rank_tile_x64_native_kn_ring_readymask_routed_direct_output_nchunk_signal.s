

/******************************************/
/* Function Prefix                        */
/******************************************/



/******************************************/
/* Begin Kernel                           */
/******************************************/

// Component.Signature.SignatureCOV3
.amdgcn_target "amdgcn-amd-amdhsa--gfx936"
.text
	.protected flux_ag_grouped_gemm_contract_rank_tile_x64_native_kn_ring_readymask_routed_direct_output_nchunk_signal
	.globl flux_ag_grouped_gemm_contract_rank_tile_x64_native_kn_ring_readymask_routed_direct_output_nchunk_signal
	.p2align 8
	.type flux_ag_grouped_gemm_contract_rank_tile_x64_native_kn_ring_readymask_routed_direct_output_nchunk_signal,@function
	.section .rodata,#alloc
	.p2align 6
	.amdhsa_kernel flux_ag_grouped_gemm_contract_rank_tile_x64_native_kn_ring_readymask_routed_direct_output_nchunk_signal
	  .amdhsa_user_sgpr_kernarg_segment_ptr 1
	  .amdhsa_next_free_vgpr 255 // vgprs
	  .amdhsa_next_free_sgpr 101 // sgprs
	  .amdhsa_group_segment_fixed_size 65536 // lds bytes
	  .amdhsa_private_segment_fixed_size 0
	  .amdhsa_system_sgpr_workgroup_id_x 1
	  .amdhsa_system_sgpr_workgroup_id_y 1
	  .amdhsa_system_sgpr_workgroup_id_z 1
	  .amdhsa_system_vgpr_workitem_id 0
	.end_amdhsa_kernel
.text

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 16 x 8 */
/* SubGroup= 16 x 32 */
/* VectorWidth=8 */
/* GlobalLoadVectorWidthA=8, GlobalLoadVectorWidthB=8 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=0 */
.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: flux_ag_grouped_gemm_contract_rank_tile_x64_native_kn_ring_readymask_routed_direct_output_nchunk_signal
    .symbol: 'flux_ag_grouped_gemm_contract_rank_tile_x64_native_kn_ring_readymask_routed_direct_output_nchunk_signal.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            sizeC
        .size:            8
        .offset:          0
        .value_kind:      by_value
        .value_type:      u64
      - .name:            sizeA
        .size:            8
        .offset:          8
        .value_kind:      by_value
        .value_type:      u64
      - .name:            sizeB
        .size:            8
        .offset:          16
        .value_kind:      by_value
        .value_type:      u64
      - .name:            D
        .size:            8
        .offset:          24
        .value_kind:      global_buffer
        .value_type:      struct
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      struct
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      struct
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          48
        .value_kind:      global_buffer
        .value_type:      struct
        .address_space:   generic
      - .name:            alpha
        .size:            4
        .offset:          56
        .value_kind:      by_value
        .value_type:      f32
      - .name:            beta
        .size:            4
        .offset:          60
        .value_kind:      by_value
        .value_type:      f32
      - .name:            strideD0
        .size:            4
        .offset:          64
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          68
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          72
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          76
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA0
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree0
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree1
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree2
        .size:            4
        .offset:          104
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesSum0
        .size:            4
        .offset:          108
        .value_kind:      by_value
        .value_type:      u32
      - .name:            OrigStaggerUIter
        .size:            4
        .offset:          112
        .value_kind:      by_value
        .value_type:      i32
      - .name:            NumWorkGroups0
        .size:            4
        .offset:          116
        .value_kind:      by_value
        .value_type:      u32
      - .name:            NumWorkGroups1
        .size:            4
        .offset:          120
        .value_kind:      by_value
        .value_type:      u32
      - .name:            NumFullBlocks
        .size:            4
        .offset:          124
        .value_kind:      by_value
        .value_type:      u32
      - .name:            WgmRemainder1
        .size:            4
        .offset:          128
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicNumberWgmRemainder1
        .size:            4
        .offset:          132
        .value_kind:      by_value
        .value_type:      u32
      - .name:            OffsetD
        .size:            4
        .offset:          136
        .value_kind:      by_value
        .value_type:      u32
      - .name:            OffsetC
        .size:            4
        .offset:          140
        .value_kind:      by_value
        .value_type:      u32
      - .name:            OffsetA
        .size:            4
        .offset:          144
        .value_kind:      by_value
        .value_type:      u32
      - .name:            OffsetB
        .size:            4
        .offset:          148
        .value_kind:      by_value
        .value_type:      u32
      - .name:            padding
        .size:            4
        .offset:          152
        .value_kind:      by_value
        .value_type:      u32
      - .name:            ptr_barrier
        .size:            8
        .offset:          160
        .value_kind:      global_buffer
      - .name:            rank
        .size:            4
        .offset:          168
        .value_kind:      by_value
        .value_type:      u32 
      - .name:            world_size
        .size:            4
        .offset:          172
        .value_kind:      by_value
        .value_type:      u32 
    .group_segment_fixed_size:   65536
    .kernarg_segment_align:      8
    .kernarg_segment_size:       112
    .max_flat_workgroup_size:    512
    .private_segment_fixed_size: 0
    .sgpr_count:                 101
    .sgpr_spill_count:           0
    .vgpr_count:                 255
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
flux_ag_grouped_gemm_contract_rank_tile_x64_native_kn_ring_readymask_routed_direct_output_nchunk_signal:

/******************************************/
/* Asm syntax workarounds                 */
/******************************************/
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

/******************************************/
/* Magic div and mod functions            */
/******************************************/
.macro V_MAGIC_DIV dstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    v_mul_hi_u32 v[\dstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\dstIdx+0], \dividend, \magicA
    _v_add_u32 v[\dstIdx+0], v[\dstIdx+0], v[\dstIdx+1]
    v_lshrrev_b32 v[\dstIdx+0], \magicShift, v[\dstIdx+0]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-128), ,  */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
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
/* Num VGPR=196 */
/* Num AccVGPR=0 */

/******************************************/
/* SGPR Assignments                       */
/******************************************/
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

/* max SGPR=100 */

/* Size Assignments */
.set sgprSizeI, sgprSizesFree+0
.set sgprSizeJ, sgprSizesFree+1
.set sgprSizeK, sgprSizesFree+2
.set sgprSizeL, sgprSizesSum+0

/* Stride Assignments */
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
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 8
.set SrdShiftLeftB, 8
/* 4GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xfffffffe
.set BufferOOB, 0xfffffffe

/******************************************/
/* Bits 127:96 of SRD.                    */
/* hex: 0x00020000                        */
/* dst_sel_x (3b): 0                      */
/* dst_sel_y (3b): 0                      */
/* dst_sel_z (3b): 0                      */
/* dst_sel_w (3b): 0                      */
/* num_format (3b): 0                     */
/* data_format (4b): 4                    */
/* user_vm_enable (1b): 0                 */
/* user_vm_mode (1b): 0                   */
/* index_stride (2b): 0                   */
/* add_tid_enable (1b): 0                 */
/* _unusedA (3b): 0                       */
/* nv (1b): 0                             */
/* _unusedB (2b): 0                       */
/* type (2b): 0                           */
/******************************************/
.set Srd127_96, 0x00020000

/* Global Offset A */
.macro GLOBAL_OFFSET_A vgprAddr:req vgprOffset0I:req vgprOffsetL:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideAL], v[\vgprOffsetL] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffset0I], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B_1 vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/* Global Offset B, for stride not equal 4096 */
.macro GLOBAL_OFFSET_B1 vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideStruct], v[\vgprOffset1J]
v_mov_b32 v[\vgprAddr+0], v[\vgprOffset1J]
v_add_u32 v[\vgprAddr+1], v[\vgprOffsetL], v[\vgprTmp+0]
v_add_co_u32  v[\vgprAddr+1],vcc, 0x8, v[\vgprAddr+1]
v_lshlrev_b32 v[\vgprAddr+1], 0x1, v[\vgprAddr+1]
.endm
/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
v_lshlrev_b32 v[\vgprAddr+0], s[sgprStrideStruct], v[\vgprOffset1J] // L3346
v_mov_b32 v[\vgprAddr+1], v[\vgprOffsetL]          //  
v_add_co_u32  v[\vgprAddr+1],vcc, 0x8, v[\vgprAddr+1]
v_lshlrev_b32 v[\vgprAddr+1], 0x1, v[\vgprAddr+1]  // offset *= bytes/element
.endm

.macro GLOBAL_STRUCT_OFFSET_B vgprAddr:req vgprOffset:req vgprIndex:req // strict routed-input rows
/* Megatron FC1 has already expanded top-k and stable-sorted every rank-local
 * routed buffer by expert.  The strict expert/rank descriptor guarantees
 * that this tile is one physically contiguous span.  v204 is the physical
 * first row loaded once in the scheduler prologue, so the GEMM body no longer
 * performs one sorted_gather_index VM load and wait for every lane. */
v_cmp_lt_u32 vcc, v[\vgprIndex], s81               // row < tile_rows
_v_add_u32 v[\vgprAddr+0], v204, v[\vgprIndex]      // physical_base + local row
v_cndmask_b32 v[\vgprAddr+0], 0, v[\vgprAddr+0], vcc // physical row or safe zero
v_mul_lo_u32 v[\vgprAddr+0], s[sgprStrideB1J], v[\vgprAddr+0]
_v_add_u32 v[\vgprAddr+0], v[\vgprAddr+0], v[\vgprOffset]
_v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]
v_lshlrev_b32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]
.endm
.macro GLOBAL_STRUCT_INDEX_B vgprAddr:req vgprIndex:req // 
v_mov_b32 v[\vgprAddr+0], 0
.endm

/* Map one local routed row through sorted_token_ids and form a general
 * 64-bit byte address in split low/high VGPR blocks.  Address low halves are
 * v134..v153 and high halves are v154..v173, matching the 20-element edge
 * batches emitted by this Ailk/NN epilogue. */
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

/* Native Ailk direct-output epilogue helpers.  One thread owns sixteen
 * routed rows and eight N coordinates per row.  The GEMM global/local-read
 * registers v180..v195 are dead after summation, so use them to issue all
 * sixteen row-map loads before one wait instead of serializing load/wait for
 * every row. */
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

/* 32-bit buffer-store sibling used only when ABI reserved bit 1 proves the
 * complete row-mapped output range fits the SRD offset field.  Keeping this
 * as a separate epilogue avoids a uniform branch before every BF16 store. */
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

/* Global Offset B */
.macro GLOBAL_OFFSET_B_TailLoop vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/******************************************/
/* Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor; */
/******************************************/
.macro DYNAMIC_VECTOR_DIVIDE vQuotient vRemainder vDividend vDivisor vTmp0 vTmp1 sTmp
v_cvt_f32_u32 v[\vQuotient], v[\vDivisor]          // 
v_rcp_f32 v[\vQuotient], v[\vQuotient]             // 
v_mul_f32 v[\vQuotient], 0x4f800000, v[\vQuotient] // 
v_cvt_u32_f32 v[\vQuotient], v[\vQuotient]         // 
v_mul_lo_u32 v[\vRemainder], v[\vDivisor], v[\vQuotient] // 
v_mul_hi_u32 v[\vTmp0], v[\vDivisor], v[\vQuotient] // 
_v_sub_co_u32 v[\vTmp1], vcc, 0x0, v[\vRemainder]  // 
v_cmp_ne_i32 s[\sTmp:\sTmp+1], 0x0, v[\vTmp0]      // 
v_cndmask_b32 v[\vRemainder], v[\vTmp1], v[\vRemainder], s[\sTmp:\sTmp+1] // 
v_mul_hi_u32 v[\vRemainder], v[\vRemainder], v[\vQuotient] // 
_v_sub_co_u32 v[\vTmp0], vcc, v[\vQuotient], v[\vRemainder] // 
_v_add_co_u32 v[\vQuotient], vcc, v[\vQuotient], v[\vRemainder] // 
v_cndmask_b32 v[\vQuotient], v[\vQuotient], v[\vTmp0], s[\sTmp:\sTmp+1] // 
v_mul_hi_u32 v[\vQuotient], v[\vQuotient], v[\vDividend] // 
v_mul_lo_u32 v[\vRemainder], v[\vQuotient], v[\vDivisor] // 
_v_sub_co_u32 v[\vTmp0], vcc, v[\vDividend], v[\vRemainder] // 
v_cmp_ge_u32 s[\sTmp:\sTmp+1], v[\vDividend], v[\vRemainder] // 
_v_add_co_u32 v[\vRemainder], vcc, 0x1, v[\vQuotient] // 
_v_add_co_u32 v[\vTmp1], vcc, -1, v[\vQuotient]    // 
v_cmp_le_u32 vcc, v[\vDivisor], v[\vTmp0]          // 
s_and_b64 vcc, s[\sTmp:\sTmp+1], vcc               // 
v_cndmask_b32 v[\vQuotient], v[\vQuotient], v[\vRemainder], vcc // 
v_cndmask_b32 v[\vQuotient], v[\vTmp1], v[\vQuotient], s[\sTmp:\sTmp+1] // 
v_cmp_ne_i32 vcc, 0x0, v[\vDivisor]                // 
v_cndmask_b32 v[\vQuotient], -1, v[\vQuotient], vcc // final result
v_mul_lo_u32 v[\vRemainder], v[\vQuotient], v[\vDivisor] // 
_v_sub_co_u32 v[\vRemainder], vcc, v[\vDividend], v[\vRemainder] // final result
.endm

/******************************************/
/* 16x8 thread-tile                       */
/******************************************/
.macro MAC_16x8_X0
// Component.MAC.FMA_F16_HPA_MAD_MIX
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+0:vgprValuC+3] 
s_setprio 1 // Raise priority while processing macs
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
s_setprio 0 // Reset priority after macs
.endm
.macro MAC_16x8_X1
// Component.MAC.FMA_F16_HPA_MAD_MIX
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0:vgprValuC+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+1], v[vgprValuC+0:vgprValuC+3] 
s_setprio 1 // Raise priority while processing macs
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
s_setprio 0 // Reset priority after macs
.endm

.macro FP32_TO_BF16 vgprValue:req
	v_cmp_u_f32 s[98:99], v[\vgprValue], v[\vgprValue] // check Nan
	v_bfe_u32 v208, v[\vgprValue], 16, 1               // Non-Nan case: store lsb of bf16
	v_add3_u32 v208, v[\vgprValue], v208, v207           // Non-Nan case: add lsb and the increment for rounding
	v_cndmask_b32 v[\vgprValue], v208, v206, s[98:99]   // 
	v_lshrrev_b32 v[\vgprValue], 16, v[\vgprValue]   // convert C to bf16
.endm




/******************************************/
/* Allocate Resources                     */
/******************************************/

s_mov_b32 m0, 0x10000                               // LDS clamp at 32768 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

/* v36 dynamic 112-byte grouped-GEMM ABI.  The scheduler/control plane is
 * retained from v36; only the GEMM body below is the native Ailk/NN body.
 *
 * 0x00 input [routed-token,K], 0x08 weight [E,K,N]
 * 0x10 sorted gather rows,     0x18 final output
 * 0x20 owner signals,          0x28 tile_info SoA
 * 0x30 problem_count,          0x34 N
 * 0x38 K,                      0x3c topk
 * 0x40 rank,                   0x44 wait value
 * 0x48 direct epilogue mode,   0x4c logical chunk N
 * 0x50 done groups,            0x54 group-major launch mode
 * 0x58 sorted_token_ids,       0x60 claims/counters
 * 0x68 done flags
 */
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
s_mov_b32 s[sgprStrideAL], s[sgprStrideD1J]        // Ailk: N-contiguous physical [K,N]
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



/* Alpha is fixed to one by this direct grouped-GEMM contract. */
label_AlphaNonZero:

v_lshrrev_b32 v[vgprValuA_X0_I0], 6, v[vgprSerial]
v_readfirstlane_b32 s[sgprWaveiD], v[vgprValuA_X0_I0]

/******************************************/
/* Local Read Addresses                   */
/******************************************/


/* local read addresses: tile assignments a/b */

v_and_b32 v0, 63, v[vgprSerial]                    // v0 = v[vgprSerial] % 64
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // v4 = v[vgprSerial] / 64
v_and_b32 v5, 3, v0                                // v5 = v0 % 4
v_lshrrev_b32 v3, 2, v0                            // v3 = v0 / 4


/* local read addresses: final offsets a */

v_and_b32 v8, 1, v4                                // v4 & 1
s_mov_b32 s62, 32                                 // 
v_mul_lo_u32 v8, s62, v8                           // A_stride*(MT_A(128))

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
v_add_lshl_u32 v[vgprLocalReadAddrA], v6, v8, 0x1  // Final offset 1: LDS-A

v_lshrrev_b32 v7, 2, v0
v_and_b32 v7, 3, v7
v_cmp_eq_u32 s[64:65], v7, 3

v_and_b32 v2, 1, v4
v_cmp_eq_u32 s[62:63], v2, 1

s_and_b64 s[62:63], s[62:63], s[64:65]
v_mov_b32 v7,0x200

v_cndmask_b32 v7, 0, v7, s[62:63]

v_sub_i32 v6, v6, v7
v_add_lshl_u32 v[vgprLocalReadAddrASwap], v6, v8, 0x1  // Final offset 1: LDS-A


/* local read addresses: final offsets b */

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

// v_sub_i32 v[vgprLocalReadAddrB_forLdsWrapTailLoop], v7, v8
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

/* local read addresses: declare addresses a */

/* N/A */


/* local read addresses: declare addresses b */

_v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x2000, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)
_v_add_co_u32 v[vgprLocalReadAddrBSwap+0], vcc, 0x2000, v[vgprLocalReadAddrBSwap+0] //  += LdsOffsetB (lower)
_v_add_co_u32 v[vgprLocalReadAddrB_forLdsWrapTailLoop], vcc, 0x2000, v[vgprLocalReadAddrB_forLdsWrapTailLoop] //  += LdsOffsetB (lower)

v_mov_b32 v[vgprLocalReadAddrA_ori], v[vgprLocalReadAddrA]
v_mov_b32 v[vgprLocalReadAddrB_ori], v[vgprLocalReadAddrB]
v_mov_b32 v[vgprLocalReadAddrASwap_ori], v[vgprLocalReadAddrASwap]
v_mov_b32 v[vgprLocalReadAddrBSwap_ori], v[vgprLocalReadAddrBSwap]

/******************************************/
/* Begin setupNewTile, isPap=False           */
/******************************************/


/* global read addresses: work-group */

/* Flux AGGroupGemm rank-tile scheduler/prologue.
 * ABI overlay:
 *   tensor2dSizeC -> int32 sorted_gather_index
 *   dataD         -> packed/AG output base
 *   dataC         -> int32 tile_info SoA, fields [expert,sorted_start,seg_start,seg_end,count]
 *   dataA         -> all-expert NN weight, [expert][k_dim][n_dim]
 *   dataB         -> full input rows, [token][k_dim]
 *   sizeI         -> global n_dim before this block; replaced by local tile_n_count
 *   sizeK         -> rank tile count before this block; replaced by 1
 *   sizeL         -> k_dim
 *   strideD1J     -> global n_dim, also used for D row stride and KN weight K stride
 *
 * Physical-N launch shape (launch mode 0):
 *   WorkGroup0 = fixed N tile id, fastest-changing launch dimension
 *   WorkGroup1 = preferred problem id
 * Group-major launch shape (launch mode 1):
 *   WorkGroup0 = N tile within one logical completion group
 *   WorkGroup1 = logical_group * problem_count + preferred_problem
 * Each CTA keeps its physical N lane and scans only problem ids. Claims use
 * problem_id * n_tiles + n_tile_id, so no power-of-two tile count is assumed.
 * Group-major order lets MoEReduceRS consume early logical N groups while the
 * dynamic ready-problem scheduler continues computing later groups.
 */
s_load_dwordx2 s[68:69], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
s_add_u32 s85, s[sgprStrideD1J], 255
s_lshr_b32 s85, s85, 8                            // persistent n_tiles
s_cmp_eq_u32 s85, 0
s_cbranch_scc1 .L_moe_problem_end
s_and_b32 s79, s77, 1                            // reserved bit 0: group-major launch
s_cmp_eq_u32 s79, 1
s_cbranch_scc1 .L_v36_direct_group_major_decode

s_cmp_ge_u32 s[sgprWorkGroup0], s85
s_cbranch_scc1 .L_moe_problem_end
s_mov_b32 s94, s[sgprWorkGroup0]                  // fixed physical N tile
s_mov_b32 s100, s94                               // preserve physical tile across GEMM SGPR aliasing
s_branch .L_v36_direct_launch_decode_done

.L_v36_direct_group_major_decode:
s_cmp_eq_u32 s[sgprSizeK], 0
s_cbranch_scc1 .L_moe_problem_end
s_lshr_b32 s83, s80, 8                            // N tiles per logical group
s_cmp_eq_u32 s83, 0
s_cbranch_scc1 .L_moe_problem_end
s_cmp_ge_u32 s[sgprWorkGroup0], s83
s_cbranch_scc1 .L_moe_problem_end
s_cmp_eq_u32 s82, 0
s_cbranch_scc1 .L_moe_problem_end

/* Decode the flattened group/problem coordinate with repeated subtraction.
 * The host keeps wider-N experiments behind a configurable validation gate,
 * avoiding a scalar-divide magic number in the fixed 112-byte ABI.
 */
s_mov_b32 s84, s[sgprWorkGroup1]                  // flattened group/problem
s_mov_b32 s88, 0                                  // logical group
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
s_add_u32 s94, s89, s[sgprWorkGroup0]             // fixed physical N tile
s_cmp_ge_u32 s94, s85
s_cbranch_scc1 .L_moe_problem_end
s_mov_b32 s[sgprWorkGroup1], s84                  // preferred problem id
s_mov_b32 s100, s94                               // preserve physical tile for completion

.L_v36_direct_launch_decode_done:

/* Real AG only: front-load one 16-problem cluster from each window, then
 * return to contiguous window-major order for the remaining three clusters.
 */
s_cmp_eq_u32 s95, 0
s_cbranch_scc1 .L_v27_frontload_done
s_and_b32 s89, s[sgprSizeK], 0xffffffc0             // full-window region
s_cmp_eq_u32 s89, 0
s_cbranch_scc1 .L_v27_frontload_done
s_cmp_ge_u32 s[sgprWorkGroup1], s89
s_cbranch_scc1 .L_v27_frontload_done
s_lshr_b32 s84, s89, 6                              // number of full windows
s_lshr_b32 s80, s[sgprWorkGroup1], 4
s_and_b32 s82, s[sgprWorkGroup1], 15
s_cmp_lt_u32 s80, s84
s_cbranch_scc1 .L_v27_frontload_first_round
s_sub_u32 s80, s80, s84
s_mul_hi_u32 s66, s80, 0xaaaaaaab
s_lshr_b32 s66, s66, 1                              // remaining cluster / 3
s_mul_i32 s89, s66, 3
s_sub_u32 s80, s80, s89
s_add_u32 s80, s80, 1
s_lshl_b32 s66, s66, 6
s_lshl_b32 s80, s80, 4
s_add_u32 s[sgprWorkGroup1], s66, s80
s_branch .L_v27_frontload_add_inner
.L_v27_frontload_first_round:
s_lshl_b32 s[sgprWorkGroup1], s80, 6
.L_v27_frontload_add_inner:
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s82
.L_v27_frontload_done:

/* Only workitem 0 scans and claims. LDS word zero broadcasts the selected
 * problem id to all 8 waves in this 768-thread CTA.
 */
v_cmp_eq_u32 vcc, 0, v[vgprSerial]
s_and_saveexec_b64 s[98:99], vcc
s_cbranch_execz .L_v12_claim_broadcast

s_mov_b32 s97, s[sgprWorkGroup1]                  // candidate problem id
/* Use 64-problem windows, but merge a short final window with its predecessor.
 * For the target's 416 problems this turns the final 64+32 problems into one
 * 96-problem window, giving the last CTAs more ready work to steal.
 */
s_and_b32 s88, s[sgprWorkGroup1], 0xffffffc0        // default window base
s_and_b32 s89, s[sgprSizeK], 0xffffffc0             // short tail base
s_mov_b32 s66, 0                                    // CTA is in merged tail
s_cmp_eq_u32 s89, s[sgprSizeK]
s_cbranch_scc1 .L_v12_window_end
s_cmp_le_u32 s[sgprSizeK], 64
s_cbranch_scc1 .L_v12_window_end
s_sub_u32 s89, s89, 64                              // merged window base
s_cmp_ge_u32 s[sgprWorkGroup1], s89
s_cselect_b32 s88, s89, s88
s_cselect_b32 s66, 1, 0                             // CTA is in merged tail
.L_v12_window_end:
s_add_u32 s84, s88, 64
s_cmp_lt_u32 s84, s[sgprSizeK]
s_cselect_b32 s84, s84, s[sgprSizeK]
s_cmp_eq_u32 s66, 1
s_cselect_b32 s84, s[sgprSizeK], s84               // merged/exclusive end
s_mov_b32 s96, 0                                   // scan backoff counter
/* READY-MASK ABI (reserved bit 9): barrier word 23 caches acquired owners.
 * Keep the mask in s83 and refresh it every eight failed candidates.  A strict
 * one-owner tile can then be filtered with one SGPR AND instead of a global
 * signal load plus cache invalidation for every scanned problem. */
v_mov_b32 v0, 92
global_load_dword v6, v0, s[sgprBarrierPtrLow:sgprBarrierPtrHigh] glc
s_waitcnt vmcnt(0)
buffer_wbinvl1_vol
v_readfirstlane_b32 s83, v6

.L_v12_claim_scan:
/* Variant-17 hybrid chunk-ready mode filters candidates using the exact
 * global M-chunk interval before claiming a problem.  The normal routed
 * path keeps the five-field owner-rank interval below. */
s_and_b32 s66, s77, 0x80                          // reserved bit 7: chunks
s_cmp_eq_u32 s66, 0x80
s_cbranch_scc0 .L_v12_rank_ready_check
s_mul_i32 s66, s[sgprSizeK], 5
s_add_u32 s66, s66, s97
s_lshl_b32 s66, s66, 2
v_mov_b32 v0, s66
global_load_dword v3, v0, s[sgprAddressC:sgprAddressC+1] glc

s_mul_i32 s66, s[sgprSizeK], 6
s_add_u32 s66, s66, s97
s_lshl_b32 s66, s66, 2
v_mov_b32 v0, s66
global_load_dword v4, v0, s[sgprAddressC:sgprAddressC+1] glc
s_waitcnt vmcnt(0)
buffer_wbinvl1_vol
v_readfirstlane_b32 s80, v3                       // chunk_start
v_readfirstlane_b32 s82, v4                       // chunk_end
s_cmp_lt_i32 s80, 0
s_cbranch_scc1 .L_v12_rank_ready_check
s_cmp_gt_i32 s80, s82
s_cbranch_scc1 .L_v12_rank_ready_check
s_mov_b32 s83, s80
.L_v12_chunk_ready_owner_loop:
s_cmp_gt_i32 s83, s82
s_cbranch_scc1 .L_v12_try_claim
s_lshl_b32 s66, s83, 2
v_mov_b32 v0, s66
global_load_dword v6, v0, s[sgprBarrierPtrLow:sgprBarrierPtrHigh] glc
s_waitcnt vmcnt(0)
buffer_wbinvl1_vol
v_readfirstlane_b32 s66, v6
s_cmp_lt_i32 s66, s95
s_cbranch_scc1 .L_v12_claim_advance
s_add_u32 s83, s83, 1
s_branch .L_v12_chunk_ready_owner_loop

.L_v12_rank_ready_check:
/* Read the candidate's ready-slot interval from tile_info SoA fields 2/3. */
s_lshl_b32 s66, s[sgprSizeK], 1
s_add_u32 s66, s66, s97
s_lshl_b32 s66, s66, 2
v_mov_b32 v0, s66
global_load_dword v3, v0, s[sgprAddressC:sgprAddressC+1] glc

s_mul_i32 s66, s[sgprSizeK], 3
s_add_u32 s66, s66, s97
s_lshl_b32 s66, s66, 2
v_mov_b32 v0, s66
global_load_dword v4, v0, s[sgprAddressC:sgprAddressC+1] glc
s_waitcnt vmcnt(0)
buffer_wbinvl1_vol
v_readfirstlane_b32 s80, v3                       // segment_start
v_readfirstlane_b32 s82, v4                       // segment_end

s_cmp_lt_i32 s80, 0
s_cbranch_scc1 .L_v12_try_claim
s_cmp_gt_i32 s80, s82
s_cbranch_scc1 .L_v12_try_claim
s_and_b32 s66, s77, 0x200
s_cmp_eq_u32 s66, 0
s_cbranch_scc1 .L_v12_ready_owner_loop
s_cmp_lg_u32 s80, s82
s_cbranch_scc1 .L_v12_ready_owner_loop
s_sub_u32 s66, s80, 24
s_cmp_lt_i32 s66, 0
s_cbranch_scc1 .L_v12_ready_owner_loop
s_cmp_ge_i32 s66, 8
s_cbranch_scc1 .L_v12_ready_owner_loop
s_lshl_b32 s66, 1, s66
s_and_b32 s66, s83, s66
s_cmp_eq_u32 s66, 0
s_cbranch_scc1 .L_v12_claim_advance
s_branch .L_v12_try_claim
s_mov_b32 s83, s80
.L_v12_ready_owner_loop:
s_cmp_gt_i32 s83, s82
s_cbranch_scc1 .L_v12_try_claim
/* tile_info stores ready-slot indices.  Under the v36 stage-signal ABI the
 * local slot is 0, not the physical rank.  Do not skip s83==rank here: doing
 * so drops one remote stage on ranks 1..7.  The local slot is already
 * published before this kernel starts, so waiting every slot is safe for
 * both stage-indexed and rank-indexed layouts. */
s_lshl_b32 s66, s83, 2
v_mov_b32 v0, s66
global_load_dword v6, v0, s[sgprBarrierPtrLow:sgprBarrierPtrHigh] glc
s_waitcnt vmcnt(0)
buffer_wbinvl1_vol
v_readfirstlane_b32 s66, v6
s_cmp_lt_i32 s66, s95
s_cbranch_scc1 .L_v12_claim_advance
.L_v12_ready_next_owner:
s_add_u32 s83, s83, 1
s_branch .L_v12_ready_owner_loop

.L_v12_try_claim:
v_mov_b32 v0, 1                                   // atomic new value
v_mov_b32 v1, 0                                   // atomic compare value
s_mul_i32 s66, s97, s85                           // problem_id * n_tiles
s_add_u32 s66, s66, s94                           // + fixed tile_id
v_lshlrev_b32 v2, 2, s66                          // byte offset in claims
global_atomic_cmpswap v0, v2, v[0:1], s[86:87] glc
s_waitcnt vmcnt(0)
v_readfirstlane_b32 s66, v0
s_cmp_eq_u32 s66, 0
s_cbranch_scc1 .L_v12_claim_success

.L_v12_claim_advance:
/* Keep the physical N lane fixed and move only through problem ids. */
s_add_u32 s97, s97, 1
s_cmp_lt_u32 s97, s84
s_cselect_b32 s97, s97, s88
s_add_u32 s96, s96, 1
s_and_b32 s66, s96, 63
s_cmp_eq_u32 s66, 0
s_cbranch_scc0 .L_v12_claim_scan
s_sleep 1
s_and_b32 s66, s96, 7
s_cmp_eq_u32 s66, 0
s_cbranch_scc0 .L_v12_claim_scan
v_mov_b32 v0, 92
global_load_dword v6, v0, s[sgprBarrierPtrLow:sgprBarrierPtrHigh] glc
s_waitcnt vmcnt(0)
buffer_wbinvl1_vol
v_readfirstlane_b32 s83, v6
s_branch .L_v12_claim_scan

.L_v12_claim_success:
v_mov_b32 v0, 0
v_mov_b32 v1, s97
ds_write_b32 v0, v1

.L_v12_claim_broadcast:
s_or_b64 exec, exec, s[98:99]
s_waitcnt lgkmcnt(0)
s_barrier
v_mov_b32 v0, 0
ds_read_b32 v1, v0
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s97, v1
s_mov_b32 s[sgprWorkGroup0], s97                   // logical problem id
s_mov_b32 s[sgprWorkGroup1], s94                   // fixed logical N tile id

s_mul_i32 s64, s[sgprWorkGroup1], 256             // tile_n_base

/* Load rank tile_info fields. */
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

v_readfirstlane_b32 s62, v1                       // expert
v_readfirstlane_b32 s63, v2                       // sorted_start
v_mov_b32 v202, s63                              // uniform sorted_start survives the NN body
v_readfirstlane_b32 s80, v3                       // segment_start
v_readfirstlane_b32 s82, v4                       // segment_end
v_readfirstlane_b32 s81, v5                       // tile_m_count

s_cmp_le_i32 s81, 0
s_cbranch_scc1 .L_moe_problem_end
s_cmp_ge_u32 s64, s[sgprStrideD1J]
s_cbranch_scc1 .L_moe_problem_end
s_sub_u32 s84, s[sgprStrideD1J], s64              // remaining N
s_mov_b32 s85, 256
s_cmp_lt_u32 s84, s85
s_cselect_b32 s84, s84, s85                       // tile_n_count=min(remain,256)
s_cmp_eq_u32 s84, 0
s_cbranch_scc1 .L_moe_problem_end

/* Dedicated Megatron FC1 routed-input ABI.  Strict rank tiles contain one
 * contiguous physical input span, so load its first row exactly once.  The
 * independent code object is selected only by the strict routed test/route;
 * the historical variant=16 object retains its legacy per-row row map. */
v_mov_b32 v204, 0
s_lshl_b32 s66, s63, 2
v_mov_b32 v0, s66
global_load_dword v6, v0, s[68:69] glc
s_waitcnt vmcnt(0)
v_readfirstlane_b32 s67, v6
v_mov_b32 v204, s67
s_waitcnt lgkmcnt(0)
s_lshl_b32 s65, s[sgprWorkGroup0], 8

/* Hybrid chunk-ready mode has already waited for the exact input interval.
 * Do not reinterpret chunk slots as the historical owner-rank signals. */
s_and_b32 s66, s77, 0x80
s_cmp_eq_u32 s66, 0x80
s_cbranch_scc1 .L_moe_after_wait

/* Wait the strict owner rank in [segment_start, segment_end], same contract as
 * the hand-written rank-tile kernel.  The intermediate ABI uses fixed signal 1.
 */
s_cmp_lt_i32 s80, 0
s_cbranch_scc1 .L_moe_after_wait
s_cmp_gt_i32 s80, s82
s_cbranch_scc1 .L_moe_after_wait
s_mov_b32 s83, s80
.L_moe_wait_owner_loop:
s_cmp_gt_i32 s83, s82
s_cbranch_scc1 .L_moe_after_wait
/* See the scheduler-side wait above: s80..s82 name signal slots, not
 * necessarily physical ranks.  Waiting the already-ready local slot is
 * cheaper than misclassifying and skipping a remote stage. */
s_lshl_b32 s66, s83, 2
v_mov_b32 v0, s66
.L_moe_wait_owner:
global_load_dword v6, v0, s[sgprBarrierPtrLow:sgprBarrierPtrHigh] glc
s_waitcnt vmcnt(0)
buffer_wbinvl1_vol
v_readfirstlane_b32 s66, v6
s_cmp_lt_i32 s66, s95
s_cbranch_scc1 .L_moe_wait_owner
.L_moe_next_owner:
s_add_u32 s83, s83, 1
s_branch .L_moe_wait_owner_loop
.L_moe_after_wait:

/* D offset: final output N tile base = tile_n_base * sizeof(bf16). */
s_mov_b32 s66, s64
s_mov_b32 s67, 0
s_lshl_b64 s[66:67], s[66:67], 1
s_add_u32 s[sgprAddressD+0], s[sgprAddressD+0], s66
s_addc_u32 s[sgprAddressD+1], s[sgprAddressD+1], s67

/* Reuse the C pointer slot as sorted_token_ids rowmap for direct output. */
s_load_dwordx2 s[sgprAddressC:sgprAddressC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_waitcnt lgkmcnt(0)

/* A/weight is physical [expert][k_dim][n_dim].  Rebase this task to
 * weight[expert, 0, tile_n_base]; the per-lane GRO below supplies k*n_dim+n.
 */
s_mul_i32 s66, s62, s[sgprSizeL]
s_mul_hi_u32 s67, s66, s[sgprStrideD1J]
s_mul_i32 s66, s66, s[sgprStrideD1J]
s_add_u32 s66, s66, s64
s_addc_u32 s67, s67, 0
s_lshl_b64 s[66:67], s[66:67], 1
s_add_u32 s[sgprAddressA+0], s[sgprAddressA+0], s66
s_addc_u32 s[sgprAddressA+1], s[sgprAddressA+1], s67

/* B stays at full_input. GLOBAL_STRUCT_OFFSET_B adds
 * (physical_base + local_row) * k_dim + k. */

/* Collapse original AG tile mapping: this CTA already owns one 256x256 tile. */
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
/* Tighten A/B tensor limits after rebasing pointers to the problem tile.
 * KN A spans the remainder of this expert matrix from tile_n_base onward.
 */
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

// 打印信息, 用于调试用
// s_mov_b32 s[sgprSrdD+0], s[sgprAddressD+0]
// s_mov_b32 s[sgprSrdD+1], s[sgprAddressD+1]
// s_mov_b32 s[sgprSrdD+2], 0xfffffffe       
// s_mov_b32 s[sgprSrdD+3], Srd127_96
// 
// v_mov_b32 v134, s[sgprWorkGroup1]
// v_lshlrev_b32 v134, 1, v134
// v_mov_b32 v0, s99
// buffer_store_short v0, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0
// s_endpgm

/* global read addresses: tile offset assignment a */

/* LVCA = 32 */
/* v0 = (local)groA-tile = serial%LVCA (note (wgA*MTA) will be added to SRD) */
/* v1 = groA-unroll = serial/LVCA */
v_lshrrev_b32 v1, 5, v[vgprSerial]                 // v1 = v[vgprSerial] / 32
v_and_b32 v0, 31, v[vgprSerial]                    // v0 = v[vgprSerial] % 32
/* gro-tile *= glvw */
v_lshlrev_b32 v0, 0x3, v0                          // v0 = v0 * 8

v_and_b32 v2, 3, v1
v_lshrrev_b32 v3, 2, v1

v_and_b32  v4, 1, v2
v_lshrrev_b32 v5, 1, v2

v_lshlrev_b32 v4, 1, v4
v_lshlrev_b32 v3, 2, v3

v_add_co_u32 v4, vcc, v4, v5
v_add_co_u32 v1, vcc, v4, v3

v_lshlrev_b32 v[vgprSerial_forLdsWrapOptA], 5, v1  // 
v_and_b32 v7, 31, v[vgprSerial]                    // 
v_add_co_u32 v[vgprSerial_forLdsWrapOptA], vcc, v[vgprSerial_forLdsWrapOptA], v7 // forLdsWrapOpt

/* global read addresses: tile offset assignment b */

/* LVCB = 2 */
/* v2 = (local)groB-tile = serial/LVCB (note (wgB*MTB) will be added to SRD) */
/* v3 = groB-unroll = serial%LVCB */
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

v_lshlrev_b32 v[vgprSerial_forLdsWrapOptB], 1, v2  // 
v_and_b32 v7, 1, v[vgprSerial]                    // 
v_add_co_u32 v[vgprSerial_forLdsWrapOptB], vcc, v[vgprSerial_forLdsWrapOptB], v7 // forLdsWrapOpt


/* global read addresses: unroll assignment a */

/* v1 */


/* global read addresses: unroll assignment b */

/* v3 */


/* global read addresses: other free assignments */

/* s[sgprWorkGroup2] */


/* global read addresses: tile offsets a */

v_mov_b32 v4, v0                                   // groA0I_0


/* global read addresses: tile offsets b */

v_mov_b32 v5, v2                                   // groB1J_0


/* global read addresses: unroll offsets a */

v_mov_b32 v6, v1                                   // groAL_0


/* global read addresses: unroll offsets b */

v_mov_b32 v7, v3                                   // groBL_0


/* global read addresses: shift a */

s_mul_i32 s62, s[sgprWorkGroup0], 256              // WorkGroup[01] * MT
s_sub_u32 s62, s[sgprSizeI], s62                   // edge = Size0I - WG*MT
s_sub_u32 s62, s62, 8                              // edge -= margin(8)
v_mov_b32 v8, s62                                  // edge vgpr = Size0I- WG*MT - margin(8)
_v_add_co_u32 v9, vcc, v8, 8                       // shiftedEdge = edge + srdShiftLeft(8)
_v_add_co_u32 v10, vcc, v4, 8                      // shiftedOffset = offset + srdShiftLeft(8)
v_cmp_lt_u32 s[62:63], v10, v9                     // shiftedOffset < shiftedEdge
v_cndmask_b32 v4, v8, v4, s[62:63]                 // offset = (offset < edge) ? offset(v4) : edge(v8)


/* global read addresses: final offsets a */

GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  4,  6, 8 // gROA_0_0_0_0

GLOBAL_OFFSET_B_TailLoop vgprGlobalReadOffsetB1, 7,  5, 8

/* global read addresses: final offsets b */

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

s_cmp_lt_u32 s[sgprStrideB1J], s[sgprStrideSwizzle]                 // LINE7491
s_cbranch_scc1 Skip_supply_offset1B                // 
s_lshr_b32 s[sgprStrideStruct], s[sgprStrideB1J], s[sgprStructBit] // 
s_lshl_b32 s[sgprOffsetStruct], s[sgprStrideStruct], s[sgprStructBit] // 
s_sub_u32 s[sgprOffsetStruct], s[sgprStrideB1J], s[sgprOffsetStruct] // 
s_branch Skip_supply_offset2B                      // 
Skip_supply_offset1B:                              // 
s_mov_b32 s[sgprStrideStruct], 1                   // 
s_mov_b32 s[sgprOffsetStruct], 0
Skip_supply_offset2B:                              // 
GLOBAL_STRUCT_INDEX_B vgprGlobalReadOffsetB+0, 5
GLOBAL_STRUCT_OFFSET_B vgprGlobalReadOffsetB+1, 7, 5

// GLOBAL_OFFSET_B_1 vgprGlobalReadOffsetB1,  7,  5, 9 // gROB_0_0_0_0



/* global read addresses: addresses a */
/***********************forTailLoop*************************/
s_mul_hi_u32 s65, s[sgprWorkGroup1], 256           // WorkGroup[01] * MT
s_mul_i32 s64, s[sgprWorkGroup1], 256              // WorkGroup[01] * MT
s_mul_hi_u32 s65, s64, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s64, s64, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_sub_u32 s[sgprShadowLimitB_forTailLoop+0], s[sgprTensor2dSizeB], s64 // sub tileStart
s_subb_u32 s[sgprShadowLimitB_forTailLoop+1], s[sgprTensor2dSizeB+1], s65 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB_forTailLoop:sgprShadowLimitB_forTailLoop+1], s[sgprShadowLimitB_forTailLoop:sgprShadowLimitB_forTailLoop+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB_forTailLoop+0], s[sgprShadowLimitB_forTailLoop+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB_forTailLoop+1], s[sgprShadowLimitB_forTailLoop+1], 0 // extend limit for pre-pad

s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 4 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 16
s_mov_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeB    // incrB (unrollIdx)
s_mul_i32 s[sgprGlobalReadIncsB+0], s[sgprGlobalReadIncsB+0], s[sgprLoopCounterL]

// s_mov_b64 s[sgprShadowLimitB_forTailLoop+0:sgprShadowLimitB_forTailLoop+1], s[sgprShadowLimitB+0:sgprShadowLimitB+1]

s_sub_u32 s[sgprShadowLimitB_forTailLoop+0], s[sgprShadowLimitB_forTailLoop+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB_forTailLoop+1], s[sgprShadowLimitB_forTailLoop+1], 0 // limit -= inc)



/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s65, s[sgprWorkGroup0], 256           // WorkGroup[01] * MT
s_mul_i32 s64, s[sgprWorkGroup0], 256              // WorkGroup[01] * MT
s_sub_u32 s[sgprShadowLimitA+0], s[sgprTensor2dSizeA], s64 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprTensor2dSizeA+1], s65 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s63, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s62, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s64, s64, s62                            // accum wg term to tilestart
s_addc_u32 s65, s65, s63                           // accum wg term to tilestart
s_lshl_b64 s[64:65], s[64:65], 0x1                 // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s64    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s65   // SRD base = Address+ tileStart1
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: addresses b */

/* max read offset = size[n] * stride[n-1] */

s_mul_hi_u32 s65, s[sgprWorkGroup1], 256           // WorkGroup[01] * MT
s_mul_i32 s64, s[sgprWorkGroup1], 256              // WorkGroup[01] * MT

s_cmp_lt_u32 s[sgprStrideB1J], s[sgprStrideSwizzle]                 // LINE8036
s_cbranch_scc1 Skip_fix_strideB1                   // 
s_lshr_b32 s[sgprStrideStruct], s[sgprStrideB1J], s[sgprStructBit] // 
s_sub_u32 s[sgprShadowLimitB+0], s[sgprSizeJ], s64 // 
s_subb_u32 s[sgprShadowLimitB+1], 0, s65           // 
s_mul_hi_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+0], s[sgprStrideStruct] // 
s_mul_i32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprStrideStruct] // 
s_branch Skip_fix_strideB2                         // 
Skip_fix_strideB1:                                 // 
s_lshr_b32 s[sgprStrideStruct], s[sgprStrideB1J], 0xd // 
s_sub_u32 s[sgprShadowLimitB+0], s[sgprSizeJ], s64 // 
s_subb_u32 s[sgprShadowLimitB+1], 0, s65           // 
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprStrideStruct] // 
Skip_fix_strideB2:   
//s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
//s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad

s_mul_hi_u32 s65, s64, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s64, s64, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_mov_b32 s[sgprSrdB+2], BufferLimit              // direct-input offsets span full input


s_mul_hi_u32 s63, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s62, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s64, s64, s62                            // accum wg term to tilestart
s_addc_u32 s65, s65, s63                           // accum wg term to tilestart
s_lshl_b64 s[64:65], s[64:65], 0x1                 // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s64    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s65   // SRD base = Address+ tileStart1
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0         // pre-pad to make room for possible pointer shift
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD


/*
s_lshr_b32 s[sgprStrideStruct], s[sgprStrideB1J], 13 //  
s_mul_i32 s90, s[sgprStrideB1J], 2                 //  
s_lshr_b32 s90, s90, s[sgprStrideStruct]           //  
s_lshl_b32 s90, s90, 16                            //  
s_or_b32 s90, s90, 0x40000000                      //  
s_cmp_eq_u32 s[sgprTmpUse1], 1                     //  
s_cbranch_scc1 Skip1_K4096                         // 
s_cmp_le_u32 s[sgprSizesSum+0], 512                // 
s_cbranch_scc1 Skip1_K_Sub2048                     // 
s_mov_b32 s[sgprStructNum], 1                      //  
s_lshl_b32 s[sgprStructNum], s[sgprStructNum], s[sgprStructBit] // 
s_or_b32 s[sgprStructNum], s[sgprStructNum], 0x40000000 // 
s_mov_b32 s90, s[sgprStructNum]                    //  
Skip1_K_Sub2048:                                   // 
Skip1_K4096:                                       // 
s_or_b32 s[sgprSrdB+1], s[sgprSrdB+1], s90         // struct buffer,
*/



s_add_u32 s[sgprStructBit1], 17, s[sgprStructBit]
s_cmp_lt_u32 s[sgprStrideB1J], s[sgprStrideSwizzle]
s_cbranch_scc1 skip_fix_set
s_mov_b32 s[sgprStructNum], 1                      //  
s_lshl_b32 s[sgprStructNum], s[sgprStructNum], s[sgprStructBit1] // 
s_branch skip_fix_set2
skip_fix_set:
s_lshr_b32 s[sgprStrideStruct], s[sgprStrideB1J], 0xd
s_lshl_b32 s[sgprStructNum], s[sgprStrideB1J], 0x1
s_lshr_b32 s[sgprStructNum],s[sgprStructNum],s[sgprStrideStruct]
s_lshl_b32 s[sgprStructNum], s[sgprStructNum], 0x10
skip_fix_set2:
s_or_b32 s[sgprStructNum], s[sgprStructNum], 0x40000000 // 
s_mov_b32 s90, s[sgprStructNum]                    //  
s_or_b32 s[sgprSrdB+1], s[sgprSrdB+1], s90         // struct buffer,



/* global read addresses: increments a */

s_mul_i32 s[sgprGlobalReadIncsA+0], DepthU*BpeA, s[sgprStrideAL] // incrA unrollIdx)


/* global read addresses: increments b */

s_mov_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeB    // incrB (unrollIdx)


/******************************************/
/* Local Write Addresses                  */
/******************************************/

/* lwaTileAssignmentA = v0 */

/* lwaTileAssignmentB = v2 */

/* lwaUnrollAssignmentA = v1 */

/* lwaUnrollAssignmentB = v3 */


/* local write addresses: first offset a */
v_lshrrev_b32 v1, 5, v[vgprSerial]                 // v1 = v[vgprSerial] / 32
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x100, v1    // lwAL**(MTA + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x1 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpe
v_readfirstlane_b32 s[sgprLocalWriteAddrA], v[vgprLocalWriteAddrA] // Copy lds write address VGPR to SGPR

s_and_b32 s64, s[sgprWaveiD], 1                   // 
s_lshl_b32 s64, s64, 18                           // 
s_or_b32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA], s64 // 

/* local write addresses: first offset b */

v_and_b32 v3, 1, v[vgprSerial]
v_lshlrev_b32 v3, 3, v3
v_lshrrev_b32 v2, 1, v[vgprSerial]

v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x10, v2     // lwBL**(MTB + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrB], v3, v[vgprLocalWriteAddrB], 0x1 // lwFOB = (lwBB + lwBL*(MT1J+PAD))*bpe
_v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x2000, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=4096*2
v_readfirstlane_b32 s[sgprLocalWriteAddrB], v[vgprLocalWriteAddrB] // Copy lds write address VGPR to SGPR

s_and_b32 s64, s[sgprWaveiD], 3                   // 
s_lshl_b32 s64, s64, 16                           // 
s_or_b32 s[sgprLocalWriteAddrB], s[sgprLocalWriteAddrB], s64 // 





/* declare loop num iterations */


s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 4 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 16
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter

/* local read addresses: init pointers a */


/* localReadInitPointers */

/* local read addresses: init pointers b */


/* localReadInitPointers */

/*
tmp_loop:

buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+3], v[vgprGlobalReadOffsetB:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0, // G -> Reg 0_0_0_0                              // Restore LDS clamp at 32768 bytes

s_mov_b32 s[sgprSrdD], s[sgprAddressD]
s_mov_b32 s[sgprSrdD+1], s[sgprAddressD]
s_mov_b32 s[sgprSrdD+2], 0x800000000
s_mov_b32 s[sgprSrdD+3], 0x00020000

v_lshlrev_b32 v0, 4, v[vgprSerial]

s_waitcnt vmcnt(0)

buffer_store_dwordx4 v[vgprG2LB+0:vgprG2LB+3], v0, s[sgprSrdD+0:sgprSrdD+3], 0, offen, offset:0x00
s_endpgm
*/



/* prefetch: global -> local */

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 ShadowInitStart_9                   // skip to ShadowInitStart iter b/c numIter==0

////////////////////////////////////
//prefetch load 1
s_barrier 
s_mov_b32 m0, s[sgprLocalWriteAddrA]               // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_0_0
s_mov_b32 m0, s[sgprLocalWriteAddrB]               // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds // G -> Reg 0_0_0_0                              // Restore LDS clamp at 32768 bytes

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_mov_b32 s[sgprSrdB+2], BufferLimit              // direct-input offsets span full input

s_cmp_eq_i32 s[sgprLoopCounterL], 0x1
s_cbranch_scc1 ShadowInitStart_9
///////////////////////////////////////////

////////////////////////////////////
//prefetch load 2
s_barrier 
s_add_u32 m0, s[sgprLocalWriteAddrA],0x4000               // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB],0x4000               // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds // G -> Reg 0_0_0_0                              // Restore LDS clamp at 32768 bytes

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_mov_b32 s[sgprSrdB+2], BufferLimit              // direct-input offsets span full input

s_cmp_eq_i32 s[sgprLoopCounterL], 0x2
s_cbranch_scc1 ShadowInitStart_9
///////////////////////////////////////////

////////////////////////////////////
//prefetch load 3
s_barrier 
s_add_u32 m0, s[sgprLocalWriteAddrA],0x8000               // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB],0x8000               // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds // G -> Reg 0_0_0_0                              // Restore LDS clamp at 32768 bytes

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_mov_b32 s[sgprSrdB+2], BufferLimit              // direct-input offsets span full input

s_cmp_eq_i32 s[sgprLoopCounterL], 0x3
s_cbranch_scc1 ShadowInitStart_9

///////////////////////////////////////////

////////////////////////////////////
//prefetch load 4
s_barrier 
s_add_u32 m0, s[sgprLocalWriteAddrA],0xc000               // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB],0xc000               // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds // G -> Reg 0_0_0_0                              // Restore LDS clamp at 32768 bytes

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_mov_b32 s[sgprSrdB+2], BufferLimit              // direct-input offsets span full input


///////////////////////////////////////////


/******************************************/
/* End setupNewTile, isPap=False             */
/******************************************/

ShadowInitStart_9: // 

s_mov_b32 s[sgprSrdD+0], s[sgprAddressD+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdD+1], s[sgprAddressD+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdD+2], 0xfffffffe                // 
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s[sgprSrdC+0], s[sgprAddressC+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdC+1], s[sgprAddressC+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdC+2], 0xfffffffe                // 
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s64, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s63, s64, s[sgprStrideC1J]            // CScale s64 by Stride
s_mul_i32 s62, s64, s[sgprStrideC1J]               // CScale s64 by Stride
s_lshl_b64 s[62:63], s[62:63], 1                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s62        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s63       // add hi to SRD
s_mul_hi_u32 s63, s64, s[sgprStrideD1J]            // Scale s64 by Stride
s_mul_i32 s62, s64, s[sgprStrideD1J]               // Scale s64 by Stride
s_lshl_b64 s[62:63], s[62:63], 1                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s62        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s63       // add hi to SRD

s_mul_hi_u32 s63, s[sgprWorkGroup2], s[sgprStrideCK] // CScale s[sgprWorkGroup2] by Stride
s_mul_i32 s62, s[sgprWorkGroup2], s[sgprStrideCK]  // CScale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[62:63], s[62:63], 1                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s62        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s63       // add hi to SRD
s_mul_hi_u32 s63, s[sgprWorkGroup2], s[sgprStrideDK] // Scale s[sgprWorkGroup2] by Stride
s_mul_i32 s62, s[sgprWorkGroup2], s[sgprStrideDK]  // Scale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[62:63], s[62:63], 1                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s62        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s63       // add hi to SRD



/* initC: remove C-tile 0-128 from pool */

/* initC: remove AB-tile 128-176 from pool */
v_mov_b32 v[vgprValuC+0], 0x0                      // initC
v_mov_b32 v[vgprValuC+1], 0x0                      // initC
v_mov_b32 v[vgprValuC+2], 0x0                      // initC
v_mov_b32 v[vgprValuC+3], 0x0                      // initC
v_mov_b32 v[vgprValuC+4], 0x0                      // initC
v_mov_b32 v[vgprValuC+5], 0x0                      // initC
v_mov_b32 v[vgprValuC+6], 0x0                      // initC
v_mov_b32 v[vgprValuC+7], 0x0                      // initC
v_mov_b32 v[vgprValuC+8], 0x0                      // initC
v_mov_b32 v[vgprValuC+9], 0x0                      // initC
v_mov_b32 v[vgprValuC+10], 0x0                     // initC
v_mov_b32 v[vgprValuC+11], 0x0                     // initC
v_mov_b32 v[vgprValuC+12], 0x0                     // initC
v_mov_b32 v[vgprValuC+13], 0x0                     // initC
v_mov_b32 v[vgprValuC+14], 0x0                     // initC
v_mov_b32 v[vgprValuC+15], 0x0                     // initC
v_mov_b32 v[vgprValuC+16], 0x0                     // initC
v_mov_b32 v[vgprValuC+17], 0x0                     // initC
v_mov_b32 v[vgprValuC+18], 0x0                     // initC
v_mov_b32 v[vgprValuC+19], 0x0                     // initC
v_mov_b32 v[vgprValuC+20], 0x0                     // initC
v_mov_b32 v[vgprValuC+21], 0x0                     // initC
v_mov_b32 v[vgprValuC+22], 0x0                     // initC
v_mov_b32 v[vgprValuC+23], 0x0                     // initC
v_mov_b32 v[vgprValuC+24], 0x0                     // initC
v_mov_b32 v[vgprValuC+25], 0x0                     // initC
v_mov_b32 v[vgprValuC+26], 0x0                     // initC
v_mov_b32 v[vgprValuC+27], 0x0                     // initC
v_mov_b32 v[vgprValuC+28], 0x0                     // initC
v_mov_b32 v[vgprValuC+29], 0x0                     // initC
v_mov_b32 v[vgprValuC+30], 0x0                     // initC
v_mov_b32 v[vgprValuC+31], 0x0                     // initC
v_mov_b32 v[vgprValuC+32], 0x0                     // initC
v_mov_b32 v[vgprValuC+33], 0x0                     // initC
v_mov_b32 v[vgprValuC+34], 0x0                     // initC
v_mov_b32 v[vgprValuC+35], 0x0                     // initC
v_mov_b32 v[vgprValuC+36], 0x0                     // initC
v_mov_b32 v[vgprValuC+37], 0x0                     // initC
v_mov_b32 v[vgprValuC+38], 0x0                     // initC
v_mov_b32 v[vgprValuC+39], 0x0                     // initC
v_mov_b32 v[vgprValuC+40], 0x0                     // initC
v_mov_b32 v[vgprValuC+41], 0x0                     // initC
v_mov_b32 v[vgprValuC+42], 0x0                     // initC
v_mov_b32 v[vgprValuC+43], 0x0                     // initC
v_mov_b32 v[vgprValuC+44], 0x0                     // initC
v_mov_b32 v[vgprValuC+45], 0x0                     // initC
v_mov_b32 v[vgprValuC+46], 0x0                     // initC
v_mov_b32 v[vgprValuC+47], 0x0                     // initC
v_mov_b32 v[vgprValuC+48], 0x0                     // initC
v_mov_b32 v[vgprValuC+49], 0x0                     // initC
v_mov_b32 v[vgprValuC+50], 0x0                     // initC
v_mov_b32 v[vgprValuC+51], 0x0                     // initC
v_mov_b32 v[vgprValuC+52], 0x0                     // initC
v_mov_b32 v[vgprValuC+53], 0x0                     // initC
v_mov_b32 v[vgprValuC+54], 0x0                     // initC
v_mov_b32 v[vgprValuC+55], 0x0                     // initC
v_mov_b32 v[vgprValuC+56], 0x0                     // initC
v_mov_b32 v[vgprValuC+57], 0x0                     // initC
v_mov_b32 v[vgprValuC+58], 0x0                     // initC
v_mov_b32 v[vgprValuC+59], 0x0                     // initC
v_mov_b32 v[vgprValuC+60], 0x0                     // initC
v_mov_b32 v[vgprValuC+61], 0x0                     // initC
v_mov_b32 v[vgprValuC+62], 0x0                     // initC
v_mov_b32 v[vgprValuC+63], 0x0                     // initC
v_mov_b32 v[vgprValuC+64], 0x0                     // initC
v_mov_b32 v[vgprValuC+65], 0x0                     // initC
v_mov_b32 v[vgprValuC+66], 0x0                     // initC
v_mov_b32 v[vgprValuC+67], 0x0                     // initC
v_mov_b32 v[vgprValuC+68], 0x0                     // initC
v_mov_b32 v[vgprValuC+69], 0x0                     // initC
v_mov_b32 v[vgprValuC+70], 0x0                     // initC
v_mov_b32 v[vgprValuC+71], 0x0                     // initC
v_mov_b32 v[vgprValuC+72], 0x0                     // initC
v_mov_b32 v[vgprValuC+73], 0x0                     // initC
v_mov_b32 v[vgprValuC+74], 0x0                     // initC
v_mov_b32 v[vgprValuC+75], 0x0                     // initC
v_mov_b32 v[vgprValuC+76], 0x0                     // initC
v_mov_b32 v[vgprValuC+77], 0x0                     // initC
v_mov_b32 v[vgprValuC+78], 0x0                     // initC
v_mov_b32 v[vgprValuC+79], 0x0                     // initC
v_mov_b32 v[vgprValuC+80], 0x0                     // initC
v_mov_b32 v[vgprValuC+81], 0x0                     // initC
v_mov_b32 v[vgprValuC+82], 0x0                     // initC
v_mov_b32 v[vgprValuC+83], 0x0                     // initC
v_mov_b32 v[vgprValuC+84], 0x0                     // initC
v_mov_b32 v[vgprValuC+85], 0x0                     // initC
v_mov_b32 v[vgprValuC+86], 0x0                     // initC
v_mov_b32 v[vgprValuC+87], 0x0                     // initC
v_mov_b32 v[vgprValuC+88], 0x0                     // initC
v_mov_b32 v[vgprValuC+89], 0x0                     // initC
v_mov_b32 v[vgprValuC+90], 0x0                     // initC
v_mov_b32 v[vgprValuC+91], 0x0                     // initC
v_mov_b32 v[vgprValuC+92], 0x0                     // initC
v_mov_b32 v[vgprValuC+93], 0x0                     // initC
v_mov_b32 v[vgprValuC+94], 0x0                     // initC
v_mov_b32 v[vgprValuC+95], 0x0                     // initC
v_mov_b32 v[vgprValuC+96], 0x0                     // initC
v_mov_b32 v[vgprValuC+97], 0x0                     // initC
v_mov_b32 v[vgprValuC+98], 0x0                     // initC
v_mov_b32 v[vgprValuC+99], 0x0                     // initC
v_mov_b32 v[vgprValuC+100], 0x0                    // initC
v_mov_b32 v[vgprValuC+101], 0x0                    // initC
v_mov_b32 v[vgprValuC+102], 0x0                    // initC
v_mov_b32 v[vgprValuC+103], 0x0                    // initC
v_mov_b32 v[vgprValuC+104], 0x0                    // initC
v_mov_b32 v[vgprValuC+105], 0x0                    // initC
v_mov_b32 v[vgprValuC+106], 0x0                    // initC
v_mov_b32 v[vgprValuC+107], 0x0                    // initC
v_mov_b32 v[vgprValuC+108], 0x0                    // initC
v_mov_b32 v[vgprValuC+109], 0x0                    // initC
v_mov_b32 v[vgprValuC+110], 0x0                    // initC
v_mov_b32 v[vgprValuC+111], 0x0                    // initC
v_mov_b32 v[vgprValuC+112], 0x0                    // initC
v_mov_b32 v[vgprValuC+113], 0x0                    // initC
v_mov_b32 v[vgprValuC+114], 0x0                    // initC
v_mov_b32 v[vgprValuC+115], 0x0                    // initC
v_mov_b32 v[vgprValuC+116], 0x0                    // initC
v_mov_b32 v[vgprValuC+117], 0x0                    // initC
v_mov_b32 v[vgprValuC+118], 0x0                    // initC
v_mov_b32 v[vgprValuC+119], 0x0                    // initC
v_mov_b32 v[vgprValuC+120], 0x0                    // initC
v_mov_b32 v[vgprValuC+121], 0x0                    // initC
v_mov_b32 v[vgprValuC+122], 0x0                    // initC
v_mov_b32 v[vgprValuC+123], 0x0                    // initC
v_mov_b32 v[vgprValuC+124], 0x0                    // initC
v_mov_b32 v[vgprValuC+125], 0x0                    // initC
v_mov_b32 v[vgprValuC+126], 0x0                    // initC
v_mov_b32 v[vgprValuC+127], 0x0                    // initC

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_10                   // Only branch on scc1
s_getpc_B64 s[62:63]                               // addr of next instr
s_add_u32 s64, PrefetchGlobalLastIterEnd_4, 0x4    // target branch offset
s_add_u32 s62, s62, s64                            // add target branch offset
s_addc_u32 s63, 0, s63                             // add high and carry
s_setpc_b64 s[62:63]                               // branch to PrefetchGlobalLastIterEnd_4
label_NoBranch_10:

s_mov_b32 s[sgprMask], 0x10000


s_cmp_gt_u32 s[sgprLoopCounterL], 3
s_cbranch_scc1 prefetchwait
s_waitcnt vmcnt(0)
prefetchwait:
s_waitcnt vmcnt(4)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x80 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180 // read data

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0 // read data
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x1000 // read data
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x00 // read data
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x1000 // read data
s_waitcnt lgkmcnt(0)
s_barrier

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/

openLoopL_11:
//s_cmp_lt_u32 s[sgprLoopCounterL], 0x4              // LoopCounterL < EndCounter
s_cmp_le_u32 s[sgprLoopCounterL], 0x4              // LoopCounterL < EndCounter
s_cbranch_scc1 LoopEndL_2                          // do not enter LoopL

s_cmp_ge_i32 s[sgprWaveiD], 4
s_cbranch_scc1 LoopBeginL_2

LoopBeginL_1:


/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/

label_0012: // LoopCopy1 


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* iter 0 */////////////////////////////////////////////////


ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0x4000 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0x4080 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0x4100 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x4180 // read data

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0x4000 // read data
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0x5000 // read data
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x4000 // read data
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x5000 // read data

s_add_u32 m0, s[sgprLocalWriteAddrA], 0x000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds // G -> Reg 0_0_0_0                              // Restore LDS clamp at 32768 bytes

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32


/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_mov_b32 s[sgprSrdB+2], BufferLimit              // direct-input offsets span full input


s_waitcnt lgkmcnt(8)

s_waitcnt vmcnt(4)
s_barrier

MAC_16x8_X0

s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4              // counterL==1
s_cbranch_scc1 LoopEndL_oddexit_31                        // restart LoopL
/////////////////////////////////////////
/////////////////////////////////////////

/* iter 1 */////////////////////////////////////////////////


ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x8000 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x8080 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x8100 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x8180 // read data

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x8000 // read data
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x9000 // read data
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x8000 // read data
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x9000 // read data

s_add_u32 m0, s[sgprLocalWriteAddrA], 0x4000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x4000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds // G -> Reg 0_0_0_0                              // Restore LDS clamp at 32768 bytes

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32


/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_mov_b32 s[sgprSrdB+2], BufferLimit              // direct-input offsets span full input


s_waitcnt lgkmcnt(8)
s_waitcnt vmcnt(4)
s_barrier

MAC_16x8_X1

s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4              // counterL==1
s_cbranch_scc1 LoopEndL_oddexit_32                        // restart LoopL
/////////////////////////////////////////
/////////////////////////////////////////

/* iter 2 */////////////////////////////////////////////////


ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0xc000 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0xc080 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0xc100 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0xc180 // read data

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0xc000 // read data
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0xd000 // read data
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0xc000 // read data
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0xd000 // read data

s_add_u32 m0, s[sgprLocalWriteAddrA], 0x8000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x8000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds // G -> Reg 0_0_0_0                              // Restore LDS clamp at 32768 bytes

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32


/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_mov_b32 s[sgprSrdB+2], BufferLimit              // direct-input offsets span full input

s_waitcnt lgkmcnt(8)
s_waitcnt vmcnt(4)
s_barrier

MAC_16x8_X0

s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4              // counterL==1
s_cbranch_scc1 LoopEndL_oddexit_33                        // restart LoopL
/////////////////////////////////////////
/////////////////////////////////////////

/* iter 3 */////////////////////////////////////////////////


ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x000 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x80 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180 // read data

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x00 // read data
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x1000 // read data
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x00 // read data
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x1000 // read data

s_add_u32 m0, s[sgprLocalWriteAddrA], 0xc000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0xc000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds // G -> Reg 0_0_0_0                              // Restore LDS clamp at 32768 bytes

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32


/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_mov_b32 s[sgprSrdB+2], BufferLimit              // direct-input offsets span full input

s_waitcnt lgkmcnt(8)
s_waitcnt vmcnt(4)
s_barrier

MAC_16x8_X1


/////////////////////////////////////////
/////////////////////////////////////////

/******************************************/
/* Unrolled Loop - End 1/2                */
/******************************************/


s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4              // counterL==1
s_cbranch_scc0 LoopBeginL_1                        // restart LoopL
s_branch LoopEndL_oddexit_34


///////////////////////////////////////
///////////////////////////////////////


LoopBeginL_2:


/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/

label_0012_1: // LoopCopy1 


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* iter 0 */////////////////////////////////////////////////


ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0x4000 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0x4080 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0x4100 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x4180 // read data

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0x4000 // read data
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0x5000 // read data
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x4000 // read data
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x5000 // read data

s_add_u32 m0, s[sgprLocalWriteAddrA], 0x000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds // G -> Reg 0_0_0_0                              // Restore LDS clamp at 32768 bytes

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32


/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_mov_b32 s[sgprSrdB+2], BufferLimit              // direct-input offsets span full input

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt vmcnt(4)
s_barrier


s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4              // counterL==1
s_cbranch_scc1 LoopEndL_oddexit_31                        // restart LoopL
/////////////////////////////////////////
/////////////////////////////////////////

/* iter 1 */////////////////////////////////////////////////


ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x8000 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x8080 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x8100 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x8180 // read data

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x8000 // read data
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x9000 // read data
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x8000 // read data
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x9000 // read data


s_add_u32 m0, s[sgprLocalWriteAddrA], 0x4000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x4000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds // G -> Reg 0_0_0_0                              // Restore LDS clamp at 32768 bytes

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32


/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_mov_b32 s[sgprSrdB+2], BufferLimit              // direct-input offsets span full input

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt vmcnt(4)
s_barrier

s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4              // counterL==1
s_cbranch_scc1 LoopEndL_oddexit_32                        // restart LoopL
/////////////////////////////////////////
/////////////////////////////////////////

/* iter 2 */////////////////////////////////////////////////


ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0xc000 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0xc080 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0xc100 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0xc180 // read data

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0xc000 // read data
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0xd000 // read data
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0xc000 // read data
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0xd000 // read data

s_add_u32 m0, s[sgprLocalWriteAddrA], 0x8000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x8000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds // G -> Reg 0_0_0_0                              // Restore LDS clamp at 32768 bytes

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32


/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_mov_b32 s[sgprSrdB+2], BufferLimit              // direct-input offsets span full input

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt vmcnt(4)
s_barrier

s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4              // counterL==1
s_cbranch_scc1 LoopEndL_oddexit_33                        // restart LoopL
/////////////////////////////////////////
/////////////////////////////////////////

/* iter 3 */////////////////////////////////////////////////


ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x00 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x80 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180 // read data

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x00 // read data
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x1000 // read data
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x00 // read data1
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x1000 // read data

s_add_u32 m0, s[sgprLocalWriteAddrA], 0xc000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0xc000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0,  lds // G -> Reg 0_0_0_0                              // Restore LDS clamp at 32768 bytes

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32


/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 0 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_mov_b32 s[sgprSrdB+2], BufferLimit              // direct-input offsets span full input

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt vmcnt(4)
s_barrier


/////////////////////////////////////////
/////////////////////////////////////////

/******************************************/
/* Unrolled Loop - End 2/2 (final)        */
/******************************************/

s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x4              // counterL==1
s_cbranch_scc0 LoopBeginL_2                        // restart LoopL
s_branch LoopEndL_oddexit_34
////////////////////////////////////
LoopEndL_oddexit_31:

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x8000 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x8080 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x8100 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x8180 // read data

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x8000 // read data
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x9000 // read data
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x8000 // read data
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x9000 // read data

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt vmcnt(2)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0xc000 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0xc080 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0xc100 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0xc180 // read data

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0xc000 // read data
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0xd000 // read data
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0xc000 // read data
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0xd000 // read data

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt vmcnt(0)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x00 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x80 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180 // read data

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x00 // read data
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x1000 // read data
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x00 // read data1
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x1000 // read data

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt lgkmcnt(0)
MAC_16x8_X0

s_branch end_store

////////////////////////////////////////////
LoopEndL_oddexit_32:

ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0xc000 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0xc080 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0xc100 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0xc180 // read data

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0xc000 // read data
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0xd000 // read data
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0xc000 // read data
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0xd000 // read data

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt vmcnt(2)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x00 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x80 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180 // read data

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x00 // read data
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x1000 // read data
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x00 // read data1
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x1000 // read data

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt vmcnt(0)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0x4000 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0x4080 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0x4100 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x4180 // read data

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0x4000 // read data
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0x5000 // read data
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x4000 // read data
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x5000 // read data

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt lgkmcnt(0)
MAC_16x8_X1

s_branch end_store

/////////////////////////////////////////////
LoopEndL_oddexit_33:

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x00 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x80 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180 // read data

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x00 // read data
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x1000 // read data
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x00 // read data1
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x1000 // read data

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt vmcnt(2)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0x4000 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0x4080 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0x4100 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x4180 // read data

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0x4000 // read data
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0x5000 // read data
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x4000 // read data
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x5000 // read data

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt vmcnt(0)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x8000 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x8080 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x8100 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x8180 // read data

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x8000 // read data
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x9000 // read data
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x8000 // read data
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x9000 // read data

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt lgkmcnt(0)
MAC_16x8_X0

s_branch end_store

////////////////////////////////////////
LoopEndL_oddexit_34:
ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0x4000 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0x4080 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0x4100 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x4180 // read data

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0x4000 // read data
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0x5000 // read data
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x4000 // read data
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x5000 // read data

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt vmcnt(2)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x8000 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x8080 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x8100 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x8180 // read data

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x8000 // read data
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x9000 // read data
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x8000 // read data
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x9000 // read data

s_waitcnt lgkmcnt(8)
MAC_16x8_X1

s_waitcnt vmcnt(0)
s_barrier

ds_read_m32x16_b16 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA] offset:0xc000 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA] offset:0xc080 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], v[vgprLocalReadAddrA] offset:0xc100 // read data
ds_read_m32x16_b16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrASwap] offset:0xc180 // read data

ds_read_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB] offset:0xc000 // read data
ds_read_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB] offset:0xd000 // read data
ds_read_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0xc000 // read data
ds_read_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0xd000 // read data

s_waitcnt lgkmcnt(8)
MAC_16x8_X0

s_waitcnt lgkmcnt(0)
MAC_16x8_X1

s_branch end_store

//////////////////////////////////////////
LoopEndL_oddexit_3: // unroll loop odditer exit
LoopEndL_2:

/////////////////////////////////////////
//////////////////////////////////////////

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

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x000 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x080 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180 // read data

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0x000 // read data
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:0x1000 // read data
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrBSwap] offset:0x00 // read data
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrBSwap] offset:0x1000 // read data

s_add_u32 s64, s64, 1
s_cmp_lt_u32 s64, s[sgprLoopCounterL]
s_cbranch_scc1 last_loop

///////////////////////////////////////////
///////////////////////////////////////////

end_store:
/* Before NLL: Check VGPR.checkin for INT8 LW */

/******************************************/
/* Opt. NoLoadLoop Without PAP - Begin         */
/******************************************/

s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 OptNLL_End_14                       // Branch if Beta is not zero


s_cmp_eq_u32 s[sgprAlpha], 1.0                     // alpha == 1.0?
s_cbranch_scc0 OptNLL_End_14                       // branch if alpha != 1

s_and_b32 s62, 255, s[sgprSizeI]                   // s62 = s[sgprSizeI] % 256
s_add_u32 s64, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s64                // wg0 >= nwg0-1 ?
s_cselect_b32 s62, s62, 0                          // set rMT0
s_cmpk_gt_u32 s62, 0x0                             // rMT0 > 0
s_cbranch_scc1 OptNLL_End_14                       // jump if edges required
s_and_b32 s62, 255, s[sgprSizeJ]                   // s62 = s[sgprSizeJ] % 256
s_add_u32 s64, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s64                // wg1 >= nwg1-1
s_cselect_b32 s62, s62, 0                          // set rMT1
s_cmpk_gt_u32 s62, 0x0                             // rMT1 > 0
s_cbranch_scc1 OptNLL_End_14                       // jump if edges required

s_and_b32 s63, 15, s[sgprSizesSum+0]               // s63 = s[sgprSizesSum+0] % 16
s_cmp_eq_u32 s63, 0x0                              // numIterL == 0
s_cbranch_scc0 OptNLL_End_14                       // skip if tail loop required


/* Stores for OptNLL */
Summation_End_OptNLL_15:
/* endSummation: add vgpr [128...188) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
.set ShadowLimitA, UNDEF
.set ShadowLimitB, UNDEF
.set LocalWriteAddrA, UNDEF
.set LocalWriteAddrB, UNDEF
/* computeStoreVgprs */
v_and_b32 v128, 63, v[vgprSerial]                  // v0 = v[vgprSerial] & 63
v_lshrrev_b32 v129, 6, v[vgprSerial]               // v4 = v[vgprSerial] / 64
v_and_b32 v132, 15, v128                           // v5 = v0 & 15
v_lshrrev_b32 v133, 4, v128                        // v3 = v0 / 16
v_and_b32 v134, 1, v129                            // v4&1
s_mov_b32 s54, 32                                 //  
v_mul_lo_u32 v134, s54, v134                       //  
v_lshlrev_b32 v128, 0, v132                        // ((v[vgprSerial]&63)&15)*2
_v_add_co_u32 v128, vcc, v128, v134                // row
v_lshrrev_b32 v134, 1, v129                        // v4>>1
s_mov_b32 s54, 32                                  //  
v_mul_lo_u32 v134, s54, v134                       //  
v_lshlrev_b32 v129, 0, v133                        // ((v[vgprSerial]&63)&15)*2
_v_add_co_u32 v129, vcc, v129, v134                // col
s_mul_i32 s[sgprStrideoffC+0], 8, s[sgprStridesC]  // 
s_mul_i32 s[sgprStrideoffC+1], 16, s[sgprStridesC] // 
s_mul_i32 s[sgprStrideoffC+2], 24, s[sgprStridesC] // 
s_mul_i32 s[sgprStrideoffD+0], 8, s[sgprStridesD]  // 
s_mul_i32 s[sgprStrideoffD+1], 16, s[sgprStridesD] // 
s_mul_i32 s[sgprStrideoffD+2], 24, s[sgprStridesD] // 
v_mul_lo_u32 v130, v129, s[sgprStrideC1J]          // rowStart vgpr
v_mul_lo_u32 v131, v129, s[sgprStrideD1J]          // rowStart vgpr

s_mul_i32 s54, 0x100, s[sgprWorkGroup0]            // s54 = wg0*MT0
_v_add_co_u32 v128, vcc, s54, v128                 // coord0 = tid0*VW + wg0*MT0
s_mul_i32 s56, 0x100, s[sgprWorkGroup1]            // <- wg1*MT1
_v_add_co_u32 v129, vcc, s56, v129                 // coord1 = tid1*VW + wg1*MT1

/* Preserve the native optimized no-load-loop compute path, but bypass its
 * original packed non-edge writer.  The accumulator and coordinate layout is
 * identical to the regular path consumed by the rowmapped direct epilogue. */
s_branch GW_B0_DIRECT_101

GW_B0_E0_18:

/* edge=0, allocate 2 sgpr. perBatch=2 perElement=0 elementsPerBatch=128 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSharedMask=1 optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,0,4:vw1); (0,0,0,5:vw1); (0,0,0,6:vw1); (0,0,0,7:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1); (0,1,0,4:vw1); (0,1,0,5:vw1); (0,1,0,6:vw1); (0,1,0,7:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,0,1,4:vw1); (0,0,1,5:vw1); (0,0,1,6:vw1); (0,0,1,7:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,1,1,2:vw1); (0,1,1,3:vw1); (0,1,1,4:vw1); (0,1,1,5:vw1); (0,1,1,6:vw1); (0,1,1,7:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,2,4:vw1); (0,0,2,5:vw1); (0,0,2,6:vw1); (0,0,2,7:vw1); (0,1,2,0:vw1); (0,1,2,1:vw1); (0,1,2,2:vw1); (0,1,2,3:vw1); (0,1,2,4:vw1); (0,1,2,5:vw1); (0,1,2,6:vw1); (0,1,2,7:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,0,3,4:vw1); (0,0,3,5:vw1); (0,0,3,6:vw1); (0,0,3,7:vw1); (0,1,3,0:vw1); (0,1,3,1:vw1); (0,1,3,2:vw1); (0,1,3,3:vw1); (0,1,3,4:vw1); (0,1,3,5:vw1); (0,1,3,6:vw1); (0,1,3,7:vw1); (0,0,4,0:vw1); (0,0,4,1:vw1); (0,0,4,2:vw1); (0,0,4,3:vw1); (0,0,4,4:vw1); (0,0,4,5:vw1); (0,0,4,6:vw1); (0,0,4,7:vw1); (0,1,4,0:vw1); (0,1,4,1:vw1); (0,1,4,2:vw1); (0,1,4,3:vw1); (0,1,4,4:vw1); (0,1,4,5:vw1); (0,1,4,6:vw1); (0,1,4,7:vw1); (0,0,5,0:vw1); (0,0,5,1:vw1); (0,0,5,2:vw1); (0,0,5,3:vw1); (0,0,5,4:vw1); (0,0,5,5:vw1); (0,0,5,6:vw1); (0,0,5,7:vw1); (0,1,5,0:vw1); (0,1,5,1:vw1); (0,1,5,2:vw1); (0,1,5,3:vw1); (0,1,5,4:vw1); (0,1,5,5:vw1); (0,1,5,6:vw1); (0,1,5,7:vw1); (0,0,6,0:vw1); (0,0,6,1:vw1); (0,0,6,2:vw1); (0,0,6,3:vw1); (0,0,6,4:vw1); (0,0,6,5:vw1); (0,0,6,6:vw1); (0,0,6,7:vw1); (0,1,6,0:vw1); (0,1,6,1:vw1); (0,1,6,2:vw1); (0,1,6,3:vw1); (0,1,6,4:vw1); (0,1,6,5:vw1); (0,1,6,6:vw1); (0,1,6,7:vw1); (0,0,7,0:vw1); (0,0,7,1:vw1); (0,0,7,2:vw1); (0,0,7,3:vw1); (0,0,7,4:vw1); (0,0,7,5:vw1); (0,0,7,6:vw1); (0,0,7,7:vw1); (0,1,7,0:vw1); (0,1,7,1:vw1); (0,1,7,2:vw1); (0,1,7,3:vw1); (0,1,7,4:vw1); (0,1,7,5:vw1); (0,1,7,6:vw1); (0,1,7,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
/* (d1,vc1,d0,vc0)=(0,0,0,4) */
/* (d1,vc1,d0,vc0)=(0,0,0,5) */
/* (d1,vc1,d0,vc0)=(0,0,0,6) */
/* (d1,vc1,d0,vc0)=(0,0,0,7) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
/* (d1,vc1,d0,vc0)=(0,0,1,4) */
/* (d1,vc1,d0,vc0)=(0,0,1,5) */
/* (d1,vc1,d0,vc0)=(0,0,1,6) */
/* (d1,vc1,d0,vc0)=(0,0,1,7) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
/* (d1,vc1,d0,vc0)=(0,1,0,4) */
/* (d1,vc1,d0,vc0)=(0,1,0,5) */
/* (d1,vc1,d0,vc0)=(0,1,0,6) */
/* (d1,vc1,d0,vc0)=(0,1,0,7) */
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
/* (d1,vc1,d0,vc0)=(0,1,1,1) */
/* (d1,vc1,d0,vc0)=(0,1,1,2) */
/* (d1,vc1,d0,vc0)=(0,1,1,3) */
/* (d1,vc1,d0,vc0)=(0,1,1,4) */
/* (d1,vc1,d0,vc0)=(0,1,1,5) */
/* (d1,vc1,d0,vc0)=(0,1,1,6) */
/* (d1,vc1,d0,vc0)=(0,1,1,7) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
/* (d1,vc1,d0,vc0)=(0,2,0,4) */
/* (d1,vc1,d0,vc0)=(0,2,0,5) */
/* (d1,vc1,d0,vc0)=(0,2,0,6) */
/* (d1,vc1,d0,vc0)=(0,2,0,7) */
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
/* (d1,vc1,d0,vc0)=(0,2,1,1) */
/* (d1,vc1,d0,vc0)=(0,2,1,2) */
/* (d1,vc1,d0,vc0)=(0,2,1,3) */
/* (d1,vc1,d0,vc0)=(0,2,1,4) */
/* (d1,vc1,d0,vc0)=(0,2,1,5) */
/* (d1,vc1,d0,vc0)=(0,2,1,6) */
/* (d1,vc1,d0,vc0)=(0,2,1,7) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
/* (d1,vc1,d0,vc0)=(0,3,0,4) */
/* (d1,vc1,d0,vc0)=(0,3,0,5) */
/* (d1,vc1,d0,vc0)=(0,3,0,6) */
/* (d1,vc1,d0,vc0)=(0,3,0,7) */
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
/* (d1,vc1,d0,vc0)=(0,3,1,1) */
/* (d1,vc1,d0,vc0)=(0,3,1,2) */
/* (d1,vc1,d0,vc0)=(0,3,1,3) */
/* (d1,vc1,d0,vc0)=(0,3,1,4) */
/* (d1,vc1,d0,vc0)=(0,3,1,5) */
/* (d1,vc1,d0,vc0)=(0,3,1,6) */
/* (d1,vc1,d0,vc0)=(0,3,1,7) */
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
/* (d1,vc1,d0,vc0)=(0,4,0,1) */
/* (d1,vc1,d0,vc0)=(0,4,0,2) */
/* (d1,vc1,d0,vc0)=(0,4,0,3) */
/* (d1,vc1,d0,vc0)=(0,4,0,4) */
/* (d1,vc1,d0,vc0)=(0,4,0,5) */
/* (d1,vc1,d0,vc0)=(0,4,0,6) */
/* (d1,vc1,d0,vc0)=(0,4,0,7) */
/* (d1,vc1,d0,vc0)=(0,4,1,0) */
/* (d1,vc1,d0,vc0)=(0,4,1,1) */
/* (d1,vc1,d0,vc0)=(0,4,1,2) */
/* (d1,vc1,d0,vc0)=(0,4,1,3) */
/* (d1,vc1,d0,vc0)=(0,4,1,4) */
/* (d1,vc1,d0,vc0)=(0,4,1,5) */
/* (d1,vc1,d0,vc0)=(0,4,1,6) */
/* (d1,vc1,d0,vc0)=(0,4,1,7) */
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
/* (d1,vc1,d0,vc0)=(0,5,0,1) */
/* (d1,vc1,d0,vc0)=(0,5,0,2) */
/* (d1,vc1,d0,vc0)=(0,5,0,3) */
/* (d1,vc1,d0,vc0)=(0,5,0,4) */
/* (d1,vc1,d0,vc0)=(0,5,0,5) */
/* (d1,vc1,d0,vc0)=(0,5,0,6) */
/* (d1,vc1,d0,vc0)=(0,5,0,7) */
/* (d1,vc1,d0,vc0)=(0,5,1,0) */
/* (d1,vc1,d0,vc0)=(0,5,1,1) */
/* (d1,vc1,d0,vc0)=(0,5,1,2) */
/* (d1,vc1,d0,vc0)=(0,5,1,3) */
/* (d1,vc1,d0,vc0)=(0,5,1,4) */
/* (d1,vc1,d0,vc0)=(0,5,1,5) */
/* (d1,vc1,d0,vc0)=(0,5,1,6) */
/* (d1,vc1,d0,vc0)=(0,5,1,7) */
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
/* (d1,vc1,d0,vc0)=(0,6,0,1) */
/* (d1,vc1,d0,vc0)=(0,6,0,2) */
/* (d1,vc1,d0,vc0)=(0,6,0,3) */
/* (d1,vc1,d0,vc0)=(0,6,0,4) */
/* (d1,vc1,d0,vc0)=(0,6,0,5) */
/* (d1,vc1,d0,vc0)=(0,6,0,6) */
/* (d1,vc1,d0,vc0)=(0,6,0,7) */
/* (d1,vc1,d0,vc0)=(0,6,1,0) */
/* (d1,vc1,d0,vc0)=(0,6,1,1) */
/* (d1,vc1,d0,vc0)=(0,6,1,2) */
/* (d1,vc1,d0,vc0)=(0,6,1,3) */
/* (d1,vc1,d0,vc0)=(0,6,1,4) */
/* (d1,vc1,d0,vc0)=(0,6,1,5) */
/* (d1,vc1,d0,vc0)=(0,6,1,6) */
/* (d1,vc1,d0,vc0)=(0,6,1,7) */
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
/* (d1,vc1,d0,vc0)=(0,7,0,1) */
/* (d1,vc1,d0,vc0)=(0,7,0,2) */
/* (d1,vc1,d0,vc0)=(0,7,0,3) */
/* (d1,vc1,d0,vc0)=(0,7,0,4) */
/* (d1,vc1,d0,vc0)=(0,7,0,5) */
/* (d1,vc1,d0,vc0)=(0,7,0,6) */
/* (d1,vc1,d0,vc0)=(0,7,0,7) */
/* (d1,vc1,d0,vc0)=(0,7,1,0) */
/* (d1,vc1,d0,vc0)=(0,7,1,1) */
/* (d1,vc1,d0,vc0)=(0,7,1,2) */
/* (d1,vc1,d0,vc0)=(0,7,1,3) */
/* (d1,vc1,d0,vc0)=(0,7,1,4) */
/* (d1,vc1,d0,vc0)=(0,7,1,5) */
/* (d1,vc1,d0,vc0)=(0,7,1,6) */
/* (d1,vc1,d0,vc0)=(0,7,1,7) */
_v_add_lshl_u32 v134, v131, v128, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

s_mov_b32 s66, 96
FP32_TO_BF16 vgprValuC+0     // convert C to bf16
buffer_store_short v0, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+1     // convert C to bf16
buffer_store_short v1, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+2     // convert C to bf16
buffer_store_short v2, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+3     // convert C to bf16
buffer_store_short v3, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+4     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v4, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+5     // convert C to bf16
buffer_store_short v5, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+6     // convert C to bf16
buffer_store_short v6, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+7     // convert C to bf16
buffer_store_short v7, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+8     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v8, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+9     // convert C to bf16
buffer_store_short v9, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+10     // convert C to bf16
buffer_store_short v10, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+11     // convert C to bf16
buffer_store_short v11, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+12     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v12, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+13     // convert C to bf16
buffer_store_short v13, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+14     // convert C to bf16
buffer_store_short v14, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+15     // convert C to bf16
buffer_store_short v15, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+16     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v16, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+17     // convert C to bf16
buffer_store_short v17, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+18     // convert C to bf16
buffer_store_short v18, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+19     // convert C to bf16
buffer_store_short v19, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+20     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v20, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+21     // convert C to bf16
buffer_store_short v21, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+22     // convert C to bf16
buffer_store_short v22, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+23     // convert C to bf16
buffer_store_short v23, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+24     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v24, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+25     // convert C to bf16
buffer_store_short v25, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+26     // convert C to bf16
buffer_store_short v26, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+27     // convert C to bf16
buffer_store_short v27, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+28     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v28, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+29     // convert C to bf16
buffer_store_short v29, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+30     // convert C to bf16
buffer_store_short v30, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+31     // convert C to bf16
buffer_store_short v31, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+32     // convert C to bf16
_v_add_lshl_u32 v134, v131, v128, 0x1              // inner add offset
s_mul_i32 s54, s[sgprStrideD1J], 32                // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_short v32, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+33     // convert C to bf16
buffer_store_short v33, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+34     // convert C to bf16
buffer_store_short v34, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+35     // convert C to bf16
buffer_store_short v35, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+36     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v36, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+37     // convert C to bf16
buffer_store_short v37, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+38     // convert C to bf16
buffer_store_short v38, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+39     // convert C to bf16
buffer_store_short v39, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+40     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v40, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+41     // convert C to bf16
buffer_store_short v41, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+42     // convert C to bf16
buffer_store_short v42, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+43     // convert C to bf16
buffer_store_short v43, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+44     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v44, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+45     // convert C to bf16
buffer_store_short v45, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+46     // convert C to bf16
buffer_store_short v46, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+47     // convert C to bf16
buffer_store_short v47, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+48     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v48, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+49     // convert C to bf16
buffer_store_short v49, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+50     // convert C to bf16
buffer_store_short v50, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+51     // convert C to bf16
buffer_store_short v51, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+52     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v52, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+53     // convert C to bf16
buffer_store_short v53, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+54     // convert C to bf16
buffer_store_short v54, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+55     // convert C to bf16
buffer_store_short v55, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+56     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v56, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+57     // convert C to bf16
buffer_store_short v57, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+58     // convert C to bf16
buffer_store_short v58, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+59     // convert C to bf16
buffer_store_short v59, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+60     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v60, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+61     // convert C to bf16
buffer_store_short v61, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+62     // convert C to bf16
buffer_store_short v62, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+63     // convert C to bf16
buffer_store_short v63, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+64     // convert C to bf16
_v_add_lshl_u32 v134, v131, v128, 0x1              // inner add offset
s_mov_b32 s67, 224
s_mul_i32 s54, s[sgprStrideD1J], s67                // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_short v64, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+65     // convert C to bf16
buffer_store_short v65, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+66     // convert C to bf16
buffer_store_short v66, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+67     // convert C to bf16
buffer_store_short v67, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+68     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v68, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+69     // convert C to bf16
buffer_store_short v69, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+70     // convert C to bf16
buffer_store_short v70, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+71     // convert C to bf16
buffer_store_short v71, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+72     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v72, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+73     // convert C to bf16
buffer_store_short v73, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+74     // convert C to bf16
buffer_store_short v74, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+75     // convert C to bf16
buffer_store_short v75, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+76     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v76, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+77     // convert C to bf16
buffer_store_short v77, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+78     // convert C to bf16
buffer_store_short v78, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+79     // convert C to bf16
buffer_store_short v79, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+80     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v80, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+81     // convert C to bf16
buffer_store_short v81, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+82     // convert C to bf16
buffer_store_short v82, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+83     // convert C to bf16
buffer_store_short v83, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+84     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v84, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+85     // convert C to bf16
buffer_store_short v85, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+86     // convert C to bf16
buffer_store_short v86, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+87     // convert C to bf16
buffer_store_short v87, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+88     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v88, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+89     // convert C to bf16
buffer_store_short v89, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+90     // convert C to bf16
buffer_store_short v90, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+91     // convert C to bf16
buffer_store_short v91, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+92     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v92, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+93     // convert C to bf16
buffer_store_short v93, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+94     // convert C to bf16
buffer_store_short v94, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+95     // convert C to bf16
buffer_store_short v95, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+96     // convert C to bf16
_v_add_lshl_u32 v134, v131, v128, 0x1              // inner add offset
s_mul_i32 s54, s[sgprStrideD1J], 32                // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_short v96, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+97     // convert C to bf16
buffer_store_short v97, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+98     // convert C to bf16
buffer_store_short v98, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+99     // convert C to bf16
buffer_store_short v99, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+100     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v100, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+101     // convert C to bf16
buffer_store_short v101, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+102     // convert C to bf16
buffer_store_short v102, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+103     // convert C to bf16
buffer_store_short v103, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+104     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v104, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+105     // convert C to bf16
buffer_store_short v105, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+106     // convert C to bf16
buffer_store_short v106, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+107     // convert C to bf16
buffer_store_short v107, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+108     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v108, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+109     // convert C to bf16
buffer_store_short v109, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+110     // convert C to bf16
buffer_store_short v110, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+111     // convert C to bf16
buffer_store_short v111, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+112     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v112, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+113     // convert C to bf16
buffer_store_short v113, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+114     // convert C to bf16
buffer_store_short v114, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+115     // convert C to bf16
buffer_store_short v115, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+116     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v116, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+117     // convert C to bf16
buffer_store_short v117, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+118     // convert C to bf16
buffer_store_short v118, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+119     // convert C to bf16
buffer_store_short v119, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+120     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v120, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+121     // convert C to bf16
buffer_store_short v121, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+122     // convert C to bf16
buffer_store_short v122, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+123     // convert C to bf16
buffer_store_short v123, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+124     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v124, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+125     // convert C to bf16
buffer_store_short v125, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+126     // convert C to bf16
buffer_store_short v126, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+127     // convert C to bf16
buffer_store_short v127, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
s_branch label_GW_End_20                           // jump to end
label_GW_End_20:

s_endpgm                                           // Kernel End
OptNLL_End_14:


/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/


PrefetchGlobalLastIterEnd_4:


/******************************************/
/* Tail Loop                              */
/******************************************/

s_cmp_eq_u32 s[sgprShadowLimitB_forTailLoop+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB_forTailLoop+0]    // Move shadow to real if we are within 2^32

/* local write reset offsets a */



/* local write reset offsets b */



//numIterL = (((sizeL % LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 15, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 16
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0              // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 SkipTailLoopL_7                     // skip to end of tail loop b/c numIter==0
s_sub_i32 s66, s[sgprLoopCounterL], 0              // 
s_cmp_gt_u32 s66, 16                               // 
s_cselect_b32 s66, 16, s66                         // 
s_mul_i32 s66, 32, s66                             // 
v_cmp_lt_u32 s[68:69], v[vgprSerial_forLdsWrapOptA], s66 // 
v_cndmask_b32 v[vgprGlobalReadOffsetA+0], -0x1, v[vgprGlobalReadOffsetA+0], s[68:69] // 

v_mov_b32 v[vgprGlobalReadOffsetB+0], v[vgprGlobalReadOffsetB1]

v_and_b32 v[vgprG2LB+0], v[vgprSerial], 1          // 
v_lshlrev_b32 v[vgprG2LB+0], 3, v[vgprG2LB+0]      // 
v_add_co_u32 v[vgprG2LB+0], vcc, 0, v[vgprG2LB+0]  // 
v_cmp_lt_u32 s[68:69], v[vgprG2LB+0], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+0], vcc, 0, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v[vgprG2LB+0], -0x1, v[vgprG2LB+0], s[68:69] // 
v_and_b32 v173, v[vgprSerial], 1                   // 
v_lshlrev_b32 v173, 3, v173                        // 
v_add_co_u32 v173, vcc, 1, v173                    // 
v_cmp_lt_u32 s[68:69], v173, s[sgprLoopCounterL]   // 
v_add_co_u32 v173, vcc, 2, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v173, -0x1, v173, s[68:69]           // 
v_and_b32 v[vgprG2LB+1], v[vgprSerial], 1          // 
v_lshlrev_b32 v[vgprG2LB+1], 3, v[vgprG2LB+1]      // 
v_add_co_u32 v[vgprG2LB+1], vcc, 2, v[vgprG2LB+1]  // 
v_cmp_lt_u32 s[68:69], v[vgprG2LB+1], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+1], vcc, 4, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v[vgprG2LB+1], -0x1, v[vgprG2LB+1], s[68:69] // 
v_and_b32 v174, v[vgprSerial], 1                   // 
v_lshlrev_b32 v174, 3, v174                        // 
v_add_co_u32 v174, vcc, 3, v174                    // 
v_cmp_lt_u32 s[68:69], v174, s[sgprLoopCounterL]   // 
v_add_co_u32 v174, vcc, 6, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v174, -0x1, v174, s[68:69]           // 
v_and_b32 v[vgprG2LB+2], v[vgprSerial], 1          // 
v_lshlrev_b32 v[vgprG2LB+2], 3, v[vgprG2LB+2]      // 
v_add_co_u32 v[vgprG2LB+2], vcc, 4, v[vgprG2LB+2]  // 
v_cmp_lt_u32 s[68:69], v[vgprG2LB+2], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+2], vcc, 8, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v[vgprG2LB+2], -0x1, v[vgprG2LB+2], s[68:69] // 
v_and_b32 v175, v[vgprSerial], 1                   // 
v_lshlrev_b32 v175, 3, v175                        // 
v_add_co_u32 v175, vcc, 5, v175                    // 
v_cmp_lt_u32 s[68:69], v175, s[sgprLoopCounterL]   // 
v_add_co_u32 v175, vcc, 10, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v175, -0x1, v175, s[68:69]           // 
v_and_b32 v[vgprG2LB+3], v[vgprSerial], 1          // 
v_lshlrev_b32 v[vgprG2LB+3], 3, v[vgprG2LB+3]      // 
v_add_co_u32 v[vgprG2LB+3], vcc, 6, v[vgprG2LB+3]  // 
v_cmp_lt_u32 s[68:69], v[vgprG2LB+3], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+3], vcc, 12, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v[vgprG2LB+3], -0x1, v[vgprG2LB+3], s[68:69] // 
v_and_b32 v176, v[vgprSerial], 1                   // 
v_lshlrev_b32 v176, 3, v176                        // 
v_add_co_u32 v176, vcc, 7, v176                    // 
v_cmp_lt_u32 s[68:69], v176, s[sgprLoopCounterL]   // 
v_add_co_u32 v176, vcc, 14, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v176, -0x1, v176, s[68:69]           // 
s_add_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 15 // tail add
s_lshr_b32 s[sgprLoopCounterL], s[sgprLoopCounterL], 4 // 


/* Update M0 for DTLDS */


/* global read a */

s_add_u32 m0, s[sgprLocalWriteAddrA], 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 lds// load one buffer value


/* Update M0 for DTLDS */


/* global read b */

/* g2l=0, load component 0 */
buffer_load_short_d16 v[vgprG2LB+0+0], v[vgprG2LB+0+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_short_d16_hi v173, v173, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v173 // HasEccHalf: pack
/* g2l=0, load component 2 */
buffer_load_short_d16 v[vgprG2LB+0+1], v[vgprG2LB+0+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_short_d16_hi v174, v174, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v174 // HasEccHalf: pack
/* g2l=0, load component 4 */
buffer_load_short_d16 v[vgprG2LB+0+2], v[vgprG2LB+0+2], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_short_d16_hi v175, v175, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+2], v[vgprG2LB+0+2], v175 // HasEccHalf: pack
/* g2l=0, load component 6 */
buffer_load_short_d16 v[vgprG2LB+0+3], v[vgprG2LB+0+3], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_short_d16_hi v176, v176, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+3], v[vgprG2LB+0+3], v176 // HasEccHalf: pack

s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=02wait for global read

s_barrier //




/* local write a */


/* local write b */

ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0


/* Recalc local read offsets */


s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-15wait for local write

s_barrier //


/* TailLoop: Check VGPR.checkin for INT8 LW */


/* local read reset offsets a */


/* localReadResetOffsets */
/* handled internally */
v_mov_b32 v[vgprLocalReadAddrA], v[vgprLocalReadAddrA_ori]
v_mov_b32 v[vgprLocalReadAddrASwap], v[vgprLocalReadAddrASwap_ori]



/* local read reset offsets b */


/* localReadResetOffsets */
/* handled internally */
v_mov_b32 v[vgprLocalReadAddrB], v[vgprLocalReadAddrB_ori]
v_mov_b32 v[vgprLocalReadAddrBSwap], v[vgprLocalReadAddrBSwap_ori]


/* local read init pointers a */


/* localReadInitPointers */


/* local read init pointers b */


/* localReadInitPointers */


/* tail loop: macs */

TailLoopBeginL_5:


/* local read a */

ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x80 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x100 // read data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x180 // read data


/* local read b */

ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB_forLdsWrapTailLoop] offset:0 // read data
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB_forLdsWrapTailLoop] offset:0x1000 // read data
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB_forLdsWrapTailLoop] offset:0x200 // read data
ds_read_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB_forLdsWrapTailLoop] offset:0x1200 // read data


/* local read inc a */

s_mov_b32 s62, 0x2000                              // inc
_v_add_co_u32 v[vgprLocalReadAddrA], vcc, s62, v[vgprLocalReadAddrA] // lrA += 512 (LSU*(MT+PAD)*bpe)
_v_add_co_u32 v[vgprLocalReadAddrASwap], vcc, s62, v[vgprLocalReadAddrASwap] // lrA += 512 (LSU*(MT+PAD)*bpe)

/* local read inc b */

s_mov_b32 s62, 0x20                                // inc
_v_add_co_u32 v[vgprLocalReadAddrB], vcc, s62, v[vgprLocalReadAddrB] // lrB += 32 (LSU*(MT+PAD)*bpe)
_v_add_co_u32 v[vgprLocalReadAddrBSwap], vcc, s62, v[vgprLocalReadAddrBSwap] // lrB += 32 (LSU*(MT+PAD)*bpe)

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-14wait for local read


.align32 8, 0xbf800001                             // align v_pk_fma
MAC_16x8_X0

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x1 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x1 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 TailLoopBeginL_5                    // restart LoopL
TailLoopEndL_6:

SkipTailLoopL_7:

Summation_End_23:
/* endSummation: add vgpr [128...188) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
.set ShadowLimitA, UNDEF
.set ShadowLimitB, UNDEF
.set LocalWriteAddrA, UNDEF
.set LocalWriteAddrB, UNDEF


/* shift vector components d0 */

v_mov_b32 v130, s[sgprWorkGroup0]                  // 
v_mul_i32_i24 v130, -0x100, v130                   // wg*MT
_v_add_co_u32 v130, vcc, s[sgprSizesFree+0], v130  // wgMT = Size - wg*MT
v_mov_b32 v128, 0x100                              // MT
v_cmp_lt_u32 s[54:55], v130, v128                  // wgMT < MT
v_cndmask_b32 v130, v128, v130, s[54:55]           // wgMT = (wgMT < MT) ? wgMT : MT
v_lshrrev_b32 v132, 7, v130                        // v132 = v130 / 128
v_lshrrev_b32 v132, 5, v130                        // v132 = v130 / 32
v_and_b32 v132, 1, v132                            // v132 = v132 % 2
v_and_b32 v133, 127, v130                          // v133 = v130 % 128
v_and_b32 v133, 255, v130                          // v133 = v130 % 256
v_lshrrev_b32 v133, 0, v133                        // v[rReg] = v[rReg] / 1
v_lshrrev_b32 v134, 4, v133                        // vectorIdx: v[qReg_temp1] = v[rReg] / 16
v_and_b32 v135, 15, v133                           // threadIdx: v[qReg_temp2] = v[rReg] % 16 
v_mul_i32_i24 v137, 0x10, v132                     // waveIdx*16
v_and_b32 v133, 7, v130                            // v[rReg] = v[wgMT]%8
_v_add_co_u32 v145, vcc, v135, v137                // waveIdx*16 + threadIdx
_v_sub_co_u32 v145, vcc, v145, v133                // waveIdx*16 + threadIdx-1/2/3/4/5/6/7
v_lshrrev_b32 v134, 6, v[vgprSerial]               // v[qReg_temp2] = v[vgprSerial] / 64
v_and_b32 v135, 1, v134                            // waveIdx
v_and_b32 v137, 15, v[vgprSerial]                  // v[qReg_temp3] = v[qReg_temp1] % 16
v_mul_i32_i24 v135, 0x10, v135                     // waveIdx*16
_v_add_co_u32 v147, vcc, v135, v137                // waveIdx*16 + threadIdx
v_mov_b32 v135, v130                               // save origin wgMT for LDSWrap
v_and_b32 v130, 127, v130                          // v130 = v130 % 128
v_lshrrev_b32 v134, 4, v130                        // v134 = v130 / 16
v_mov_b32 v130, v135                               // fetch origin wgMT for LDSWrap
v_and_b32 v130, 255, v130                          // v130 = v130 % 256
v_lshrrev_b32 v134, 6, v130                        // v134 = v130 / 64
v_lshlrev_b32 v134, 1, v134                        // ULDSWrapMethodA-subGroupIndex
v_and_b32 v135, 31, v130                           // v135 = v130 % 32
v_lshrrev_b32 v135, 4, v135                        // v135 = v135 / 16
_v_add_co_u32 v134, vcc, v134, v135                // vId = 2 components
v_and_b32 v135, 1, v130                            // v135 = v130 % 2
v_mov_b32 v136, v135                               // duplicate
v_lshrrev_b32 v135, 1, v136                        // v135 = v136 / 2
_v_add_co_u32 v135, vcc, v134, v135                // vId = 2 components
v_cmp_eq_u32 vcc, v133, 0x1                        // wgMT%VW == 1
s_cbranch_vccnz label_0024                         // shift d0 r=1
v_cmp_eq_u32 vcc, v133, 0x2                        // wgMT%VW == 2
s_cbranch_vccnz label_0033                         // shift d0 r=2
v_cmp_eq_u32 vcc, v133, 0x3                        // wgMT%VW == 3
s_cbranch_vccnz label_0042                         // shift d0 r=3
v_cmp_eq_u32 vcc, v133, 0x4                        // wgMT%VW == 4
s_cbranch_vccnz label_0051                         // shift d0 r=4
v_cmp_eq_u32 vcc, v133, 0x5                        // wgMT%VW == 5
s_cbranch_vccnz label_0060                         // shift d0 r=5
v_cmp_eq_u32 vcc, v133, 0x6                        // wgMT%VW == 6
s_cbranch_vccnz label_0069                         // shift d0 r=6
v_cmp_eq_u32 vcc, v133, 0x7                        // wgMT%VW == 7
s_cbranch_vccnz label_0078                         // shift d0 r=7
s_branch label_0087                                // no shifting

/******************************************/
/* shift d0 r=1                           */
/******************************************/
label_0024:
v_cmp_eq_u32 vcc, v135, 0x0                        // wgMT/(SG*VW) == 0
s_cbranch_vccnz label_0025                         // shift d0, r=1, v=0
v_cmp_eq_u32 vcc, v135, 0x1                        // wgMT/(SG*VW) == 1
s_cbranch_vccnz label_0026                         // shift d0, r=1, v=1
v_cmp_eq_u32 vcc, v135, 0x2                        // wgMT/(SG*VW) == 2
s_cbranch_vccnz label_0027                         // shift d0, r=1, v=2
v_cmp_eq_u32 vcc, v135, 0x3                        // wgMT/(SG*VW) == 3
s_cbranch_vccnz label_0028                         // shift d0, r=1, v=3
v_cmp_eq_u32 vcc, v135, 0x4                        // wgMT/(SG*VW) == 4
s_cbranch_vccnz label_0029                         // shift d0, r=1, v=4
v_cmp_eq_u32 vcc, v135, 0x5                        // wgMT/(SG*VW) == 5
s_cbranch_vccnz label_0030                         // shift d0, r=1, v=5
v_cmp_eq_u32 vcc, v135, 0x6                        // wgMT/(SG*VW) == 6
s_cbranch_vccnz label_0031                         // shift d0, r=1, v=6
v_cmp_eq_u32 vcc, v135, 0x7                        // wgMT/(SG*VW) == 7
s_cbranch_vccnz label_0032                         // shift d0, r=1, v=7

/* shift d0 r=1 v=0 */
label_0025:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v0, v134, v0, offset:28            // permute edge values
ds_bpermute_b32 v1, v134, v1, offset:28            // permute edge values
ds_bpermute_b32 v2, v134, v2, offset:28            // permute edge values
ds_bpermute_b32 v3, v134, v3, offset:28            // permute edge values
ds_bpermute_b32 v32, v134, v32, offset:28          // permute edge values
ds_bpermute_b32 v33, v134, v33, offset:28          // permute edge values
ds_bpermute_b32 v34, v134, v34, offset:28          // permute edge values
ds_bpermute_b32 v35, v134, v35, offset:28          // permute edge values
ds_bpermute_b32 v64, v134, v64, offset:28          // permute edge values
ds_bpermute_b32 v65, v134, v65, offset:28          // permute edge values
ds_bpermute_b32 v66, v134, v66, offset:28          // permute edge values
ds_bpermute_b32 v67, v134, v67, offset:28          // permute edge values
ds_bpermute_b32 v96, v134, v96, offset:28          // permute edge values
ds_bpermute_b32 v97, v134, v97, offset:28          // permute edge values
ds_bpermute_b32 v98, v134, v98, offset:28          // permute edge values
ds_bpermute_b32 v99, v134, v99, offset:28          // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=1 v=1 */
label_0026:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v4, v134, v4, offset:28            // permute edge values
ds_bpermute_b32 v5, v134, v5, offset:28            // permute edge values
ds_bpermute_b32 v6, v134, v6, offset:28            // permute edge values
ds_bpermute_b32 v7, v134, v7, offset:28            // permute edge values
ds_bpermute_b32 v36, v134, v36, offset:28          // permute edge values
ds_bpermute_b32 v37, v134, v37, offset:28          // permute edge values
ds_bpermute_b32 v38, v134, v38, offset:28          // permute edge values
ds_bpermute_b32 v39, v134, v39, offset:28          // permute edge values
ds_bpermute_b32 v68, v134, v68, offset:28          // permute edge values
ds_bpermute_b32 v69, v134, v69, offset:28          // permute edge values
ds_bpermute_b32 v70, v134, v70, offset:28          // permute edge values
ds_bpermute_b32 v71, v134, v71, offset:28          // permute edge values
ds_bpermute_b32 v100, v134, v100, offset:28        // permute edge values
ds_bpermute_b32 v101, v134, v101, offset:28        // permute edge values
ds_bpermute_b32 v102, v134, v102, offset:28        // permute edge values
ds_bpermute_b32 v103, v134, v103, offset:28        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=1 v=2 */
label_0027:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v8, v134, v8, offset:28            // permute edge values
ds_bpermute_b32 v9, v134, v9, offset:28            // permute edge values
ds_bpermute_b32 v10, v134, v10, offset:28          // permute edge values
ds_bpermute_b32 v11, v134, v11, offset:28          // permute edge values
ds_bpermute_b32 v40, v134, v40, offset:28          // permute edge values
ds_bpermute_b32 v41, v134, v41, offset:28          // permute edge values
ds_bpermute_b32 v42, v134, v42, offset:28          // permute edge values
ds_bpermute_b32 v43, v134, v43, offset:28          // permute edge values
ds_bpermute_b32 v72, v134, v72, offset:28          // permute edge values
ds_bpermute_b32 v73, v134, v73, offset:28          // permute edge values
ds_bpermute_b32 v74, v134, v74, offset:28          // permute edge values
ds_bpermute_b32 v75, v134, v75, offset:28          // permute edge values
ds_bpermute_b32 v104, v134, v104, offset:28        // permute edge values
ds_bpermute_b32 v105, v134, v105, offset:28        // permute edge values
ds_bpermute_b32 v106, v134, v106, offset:28        // permute edge values
ds_bpermute_b32 v107, v134, v107, offset:28        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=1 v=3 */
label_0028:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v12, v134, v12, offset:28          // permute edge values
ds_bpermute_b32 v13, v134, v13, offset:28          // permute edge values
ds_bpermute_b32 v14, v134, v14, offset:28          // permute edge values
ds_bpermute_b32 v15, v134, v15, offset:28          // permute edge values
ds_bpermute_b32 v44, v134, v44, offset:28          // permute edge values
ds_bpermute_b32 v45, v134, v45, offset:28          // permute edge values
ds_bpermute_b32 v46, v134, v46, offset:28          // permute edge values
ds_bpermute_b32 v47, v134, v47, offset:28          // permute edge values
ds_bpermute_b32 v76, v134, v76, offset:28          // permute edge values
ds_bpermute_b32 v77, v134, v77, offset:28          // permute edge values
ds_bpermute_b32 v78, v134, v78, offset:28          // permute edge values
ds_bpermute_b32 v79, v134, v79, offset:28          // permute edge values
ds_bpermute_b32 v108, v134, v108, offset:28        // permute edge values
ds_bpermute_b32 v109, v134, v109, offset:28        // permute edge values
ds_bpermute_b32 v110, v134, v110, offset:28        // permute edge values
ds_bpermute_b32 v111, v134, v111, offset:28        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=1 v=4 */
label_0029:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v16, v134, v16, offset:28          // permute edge values
ds_bpermute_b32 v17, v134, v17, offset:28          // permute edge values
ds_bpermute_b32 v18, v134, v18, offset:28          // permute edge values
ds_bpermute_b32 v19, v134, v19, offset:28          // permute edge values
ds_bpermute_b32 v48, v134, v48, offset:28          // permute edge values
ds_bpermute_b32 v49, v134, v49, offset:28          // permute edge values
ds_bpermute_b32 v50, v134, v50, offset:28          // permute edge values
ds_bpermute_b32 v51, v134, v51, offset:28          // permute edge values
ds_bpermute_b32 v80, v134, v80, offset:28          // permute edge values
ds_bpermute_b32 v81, v134, v81, offset:28          // permute edge values
ds_bpermute_b32 v82, v134, v82, offset:28          // permute edge values
ds_bpermute_b32 v83, v134, v83, offset:28          // permute edge values
ds_bpermute_b32 v112, v134, v112, offset:28        // permute edge values
ds_bpermute_b32 v113, v134, v113, offset:28        // permute edge values
ds_bpermute_b32 v114, v134, v114, offset:28        // permute edge values
ds_bpermute_b32 v115, v134, v115, offset:28        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=1 v=5 */
label_0030:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v20, v134, v20, offset:28          // permute edge values
ds_bpermute_b32 v21, v134, v21, offset:28          // permute edge values
ds_bpermute_b32 v22, v134, v22, offset:28          // permute edge values
ds_bpermute_b32 v23, v134, v23, offset:28          // permute edge values
ds_bpermute_b32 v52, v134, v52, offset:28          // permute edge values
ds_bpermute_b32 v53, v134, v53, offset:28          // permute edge values
ds_bpermute_b32 v54, v134, v54, offset:28          // permute edge values
ds_bpermute_b32 v55, v134, v55, offset:28          // permute edge values
ds_bpermute_b32 v84, v134, v84, offset:28          // permute edge values
ds_bpermute_b32 v85, v134, v85, offset:28          // permute edge values
ds_bpermute_b32 v86, v134, v86, offset:28          // permute edge values
ds_bpermute_b32 v87, v134, v87, offset:28          // permute edge values
ds_bpermute_b32 v116, v134, v116, offset:28        // permute edge values
ds_bpermute_b32 v117, v134, v117, offset:28        // permute edge values
ds_bpermute_b32 v118, v134, v118, offset:28        // permute edge values
ds_bpermute_b32 v119, v134, v119, offset:28        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=1 v=6 */
label_0031:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v24, v134, v24, offset:28          // permute edge values
ds_bpermute_b32 v25, v134, v25, offset:28          // permute edge values
ds_bpermute_b32 v26, v134, v26, offset:28          // permute edge values
ds_bpermute_b32 v27, v134, v27, offset:28          // permute edge values
ds_bpermute_b32 v56, v134, v56, offset:28          // permute edge values
ds_bpermute_b32 v57, v134, v57, offset:28          // permute edge values
ds_bpermute_b32 v58, v134, v58, offset:28          // permute edge values
ds_bpermute_b32 v59, v134, v59, offset:28          // permute edge values
ds_bpermute_b32 v88, v134, v88, offset:28          // permute edge values
ds_bpermute_b32 v89, v134, v89, offset:28          // permute edge values
ds_bpermute_b32 v90, v134, v90, offset:28          // permute edge values
ds_bpermute_b32 v91, v134, v91, offset:28          // permute edge values
ds_bpermute_b32 v120, v134, v120, offset:28        // permute edge values
ds_bpermute_b32 v121, v134, v121, offset:28        // permute edge values
ds_bpermute_b32 v122, v134, v122, offset:28        // permute edge values
ds_bpermute_b32 v123, v134, v123, offset:28        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=1 v=7 */
label_0032:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v28, v134, v28, offset:28          // permute edge values
ds_bpermute_b32 v29, v134, v29, offset:28          // permute edge values
ds_bpermute_b32 v30, v134, v30, offset:28          // permute edge values
ds_bpermute_b32 v31, v134, v31, offset:28          // permute edge values
ds_bpermute_b32 v60, v134, v60, offset:28          // permute edge values
ds_bpermute_b32 v61, v134, v61, offset:28          // permute edge values
ds_bpermute_b32 v62, v134, v62, offset:28          // permute edge values
ds_bpermute_b32 v63, v134, v63, offset:28          // permute edge values
ds_bpermute_b32 v92, v134, v92, offset:28          // permute edge values
ds_bpermute_b32 v93, v134, v93, offset:28          // permute edge values
ds_bpermute_b32 v94, v134, v94, offset:28          // permute edge values
ds_bpermute_b32 v95, v134, v95, offset:28          // permute edge values
ds_bpermute_b32 v124, v134, v124, offset:28        // permute edge values
ds_bpermute_b32 v125, v134, v125, offset:28        // permute edge values
ds_bpermute_b32 v126, v134, v126, offset:28        // permute edge values
ds_bpermute_b32 v127, v134, v127, offset:28        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/******************************************/
/* shift d0 r=2                           */
/******************************************/
label_0033:
v_cmp_eq_u32 vcc, v135, 0x0                        // wgMT/(SG*VW) == 0
s_cbranch_vccnz label_0034                         // shift d0, r=2, v=0
v_cmp_eq_u32 vcc, v135, 0x1                        // wgMT/(SG*VW) == 1
s_cbranch_vccnz label_0035                         // shift d0, r=2, v=1
v_cmp_eq_u32 vcc, v135, 0x2                        // wgMT/(SG*VW) == 2
s_cbranch_vccnz label_0036                         // shift d0, r=2, v=2
v_cmp_eq_u32 vcc, v135, 0x3                        // wgMT/(SG*VW) == 3
s_cbranch_vccnz label_0037                         // shift d0, r=2, v=3
v_cmp_eq_u32 vcc, v135, 0x4                        // wgMT/(SG*VW) == 4
s_cbranch_vccnz label_0038                         // shift d0, r=2, v=4
v_cmp_eq_u32 vcc, v135, 0x5                        // wgMT/(SG*VW) == 5
s_cbranch_vccnz label_0039                         // shift d0, r=2, v=5
v_cmp_eq_u32 vcc, v135, 0x6                        // wgMT/(SG*VW) == 6
s_cbranch_vccnz label_0040                         // shift d0, r=2, v=6
v_cmp_eq_u32 vcc, v135, 0x7                        // wgMT/(SG*VW) == 7
s_cbranch_vccnz label_0041                         // shift d0, r=2, v=7

/* shift d0 r=2 v=0 */
label_0034:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v0, v134, v0, offset:24            // permute edge values
ds_bpermute_b32 v1, v134, v1, offset:24            // permute edge values
ds_bpermute_b32 v2, v134, v2, offset:24            // permute edge values
ds_bpermute_b32 v3, v134, v3, offset:24            // permute edge values
ds_bpermute_b32 v32, v134, v32, offset:24          // permute edge values
ds_bpermute_b32 v33, v134, v33, offset:24          // permute edge values
ds_bpermute_b32 v34, v134, v34, offset:24          // permute edge values
ds_bpermute_b32 v35, v134, v35, offset:24          // permute edge values
ds_bpermute_b32 v64, v134, v64, offset:24          // permute edge values
ds_bpermute_b32 v65, v134, v65, offset:24          // permute edge values
ds_bpermute_b32 v66, v134, v66, offset:24          // permute edge values
ds_bpermute_b32 v67, v134, v67, offset:24          // permute edge values
ds_bpermute_b32 v96, v134, v96, offset:24          // permute edge values
ds_bpermute_b32 v97, v134, v97, offset:24          // permute edge values
ds_bpermute_b32 v98, v134, v98, offset:24          // permute edge values
ds_bpermute_b32 v99, v134, v99, offset:24          // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=2 v=1 */
label_0035:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v4, v134, v4, offset:24            // permute edge values
ds_bpermute_b32 v5, v134, v5, offset:24            // permute edge values
ds_bpermute_b32 v6, v134, v6, offset:24            // permute edge values
ds_bpermute_b32 v7, v134, v7, offset:24            // permute edge values
ds_bpermute_b32 v36, v134, v36, offset:24          // permute edge values
ds_bpermute_b32 v37, v134, v37, offset:24          // permute edge values
ds_bpermute_b32 v38, v134, v38, offset:24          // permute edge values
ds_bpermute_b32 v39, v134, v39, offset:24          // permute edge values
ds_bpermute_b32 v68, v134, v68, offset:24          // permute edge values
ds_bpermute_b32 v69, v134, v69, offset:24          // permute edge values
ds_bpermute_b32 v70, v134, v70, offset:24          // permute edge values
ds_bpermute_b32 v71, v134, v71, offset:24          // permute edge values
ds_bpermute_b32 v100, v134, v100, offset:24        // permute edge values
ds_bpermute_b32 v101, v134, v101, offset:24        // permute edge values
ds_bpermute_b32 v102, v134, v102, offset:24        // permute edge values
ds_bpermute_b32 v103, v134, v103, offset:24        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=2 v=2 */
label_0036:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v8, v134, v8, offset:24            // permute edge values
ds_bpermute_b32 v9, v134, v9, offset:24            // permute edge values
ds_bpermute_b32 v10, v134, v10, offset:24          // permute edge values
ds_bpermute_b32 v11, v134, v11, offset:24          // permute edge values
ds_bpermute_b32 v40, v134, v40, offset:24          // permute edge values
ds_bpermute_b32 v41, v134, v41, offset:24          // permute edge values
ds_bpermute_b32 v42, v134, v42, offset:24          // permute edge values
ds_bpermute_b32 v43, v134, v43, offset:24          // permute edge values
ds_bpermute_b32 v72, v134, v72, offset:24          // permute edge values
ds_bpermute_b32 v73, v134, v73, offset:24          // permute edge values
ds_bpermute_b32 v74, v134, v74, offset:24          // permute edge values
ds_bpermute_b32 v75, v134, v75, offset:24          // permute edge values
ds_bpermute_b32 v104, v134, v104, offset:24        // permute edge values
ds_bpermute_b32 v105, v134, v105, offset:24        // permute edge values
ds_bpermute_b32 v106, v134, v106, offset:24        // permute edge values
ds_bpermute_b32 v107, v134, v107, offset:24        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=2 v=3 */
label_0037:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v12, v134, v12, offset:24          // permute edge values
ds_bpermute_b32 v13, v134, v13, offset:24          // permute edge values
ds_bpermute_b32 v14, v134, v14, offset:24          // permute edge values
ds_bpermute_b32 v15, v134, v15, offset:24          // permute edge values
ds_bpermute_b32 v44, v134, v44, offset:24          // permute edge values
ds_bpermute_b32 v45, v134, v45, offset:24          // permute edge values
ds_bpermute_b32 v46, v134, v46, offset:24          // permute edge values
ds_bpermute_b32 v47, v134, v47, offset:24          // permute edge values
ds_bpermute_b32 v76, v134, v76, offset:24          // permute edge values
ds_bpermute_b32 v77, v134, v77, offset:24          // permute edge values
ds_bpermute_b32 v78, v134, v78, offset:24          // permute edge values
ds_bpermute_b32 v79, v134, v79, offset:24          // permute edge values
ds_bpermute_b32 v108, v134, v108, offset:24        // permute edge values
ds_bpermute_b32 v109, v134, v109, offset:24        // permute edge values
ds_bpermute_b32 v110, v134, v110, offset:24        // permute edge values
ds_bpermute_b32 v111, v134, v111, offset:24        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=2 v=4 */
label_0038:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v16, v134, v16, offset:24          // permute edge values
ds_bpermute_b32 v17, v134, v17, offset:24          // permute edge values
ds_bpermute_b32 v18, v134, v18, offset:24          // permute edge values
ds_bpermute_b32 v19, v134, v19, offset:24          // permute edge values
ds_bpermute_b32 v48, v134, v48, offset:24          // permute edge values
ds_bpermute_b32 v49, v134, v49, offset:24          // permute edge values
ds_bpermute_b32 v50, v134, v50, offset:24          // permute edge values
ds_bpermute_b32 v51, v134, v51, offset:24          // permute edge values
ds_bpermute_b32 v80, v134, v80, offset:24          // permute edge values
ds_bpermute_b32 v81, v134, v81, offset:24          // permute edge values
ds_bpermute_b32 v82, v134, v82, offset:24          // permute edge values
ds_bpermute_b32 v83, v134, v83, offset:24          // permute edge values
ds_bpermute_b32 v112, v134, v112, offset:24        // permute edge values
ds_bpermute_b32 v113, v134, v113, offset:24        // permute edge values
ds_bpermute_b32 v114, v134, v114, offset:24        // permute edge values
ds_bpermute_b32 v115, v134, v115, offset:24        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=2 v=5 */
label_0039:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v20, v134, v20, offset:24          // permute edge values
ds_bpermute_b32 v21, v134, v21, offset:24          // permute edge values
ds_bpermute_b32 v22, v134, v22, offset:24          // permute edge values
ds_bpermute_b32 v23, v134, v23, offset:24          // permute edge values
ds_bpermute_b32 v52, v134, v52, offset:24          // permute edge values
ds_bpermute_b32 v53, v134, v53, offset:24          // permute edge values
ds_bpermute_b32 v54, v134, v54, offset:24          // permute edge values
ds_bpermute_b32 v55, v134, v55, offset:24          // permute edge values
ds_bpermute_b32 v84, v134, v84, offset:24          // permute edge values
ds_bpermute_b32 v85, v134, v85, offset:24          // permute edge values
ds_bpermute_b32 v86, v134, v86, offset:24          // permute edge values
ds_bpermute_b32 v87, v134, v87, offset:24          // permute edge values
ds_bpermute_b32 v116, v134, v116, offset:24        // permute edge values
ds_bpermute_b32 v117, v134, v117, offset:24        // permute edge values
ds_bpermute_b32 v118, v134, v118, offset:24        // permute edge values
ds_bpermute_b32 v119, v134, v119, offset:24        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=2 v=6 */
label_0040:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v24, v134, v24, offset:24          // permute edge values
ds_bpermute_b32 v25, v134, v25, offset:24          // permute edge values
ds_bpermute_b32 v26, v134, v26, offset:24          // permute edge values
ds_bpermute_b32 v27, v134, v27, offset:24          // permute edge values
ds_bpermute_b32 v56, v134, v56, offset:24          // permute edge values
ds_bpermute_b32 v57, v134, v57, offset:24          // permute edge values
ds_bpermute_b32 v58, v134, v58, offset:24          // permute edge values
ds_bpermute_b32 v59, v134, v59, offset:24          // permute edge values
ds_bpermute_b32 v88, v134, v88, offset:24          // permute edge values
ds_bpermute_b32 v89, v134, v89, offset:24          // permute edge values
ds_bpermute_b32 v90, v134, v90, offset:24          // permute edge values
ds_bpermute_b32 v91, v134, v91, offset:24          // permute edge values
ds_bpermute_b32 v120, v134, v120, offset:24        // permute edge values
ds_bpermute_b32 v121, v134, v121, offset:24        // permute edge values
ds_bpermute_b32 v122, v134, v122, offset:24        // permute edge values
ds_bpermute_b32 v123, v134, v123, offset:24        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=2 v=7 */
label_0041:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v28, v134, v28, offset:24          // permute edge values
ds_bpermute_b32 v29, v134, v29, offset:24          // permute edge values
ds_bpermute_b32 v30, v134, v30, offset:24          // permute edge values
ds_bpermute_b32 v31, v134, v31, offset:24          // permute edge values
ds_bpermute_b32 v60, v134, v60, offset:24          // permute edge values
ds_bpermute_b32 v61, v134, v61, offset:24          // permute edge values
ds_bpermute_b32 v62, v134, v62, offset:24          // permute edge values
ds_bpermute_b32 v63, v134, v63, offset:24          // permute edge values
ds_bpermute_b32 v92, v134, v92, offset:24          // permute edge values
ds_bpermute_b32 v93, v134, v93, offset:24          // permute edge values
ds_bpermute_b32 v94, v134, v94, offset:24          // permute edge values
ds_bpermute_b32 v95, v134, v95, offset:24          // permute edge values
ds_bpermute_b32 v124, v134, v124, offset:24        // permute edge values
ds_bpermute_b32 v125, v134, v125, offset:24        // permute edge values
ds_bpermute_b32 v126, v134, v126, offset:24        // permute edge values
ds_bpermute_b32 v127, v134, v127, offset:24        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/******************************************/
/* shift d0 r=3                           */
/******************************************/
label_0042:
v_cmp_eq_u32 vcc, v135, 0x0                        // wgMT/(SG*VW) == 0
s_cbranch_vccnz label_0043                         // shift d0, r=3, v=0
v_cmp_eq_u32 vcc, v135, 0x1                        // wgMT/(SG*VW) == 1
s_cbranch_vccnz label_0044                         // shift d0, r=3, v=1
v_cmp_eq_u32 vcc, v135, 0x2                        // wgMT/(SG*VW) == 2
s_cbranch_vccnz label_0045                         // shift d0, r=3, v=2
v_cmp_eq_u32 vcc, v135, 0x3                        // wgMT/(SG*VW) == 3
s_cbranch_vccnz label_0046                         // shift d0, r=3, v=3
v_cmp_eq_u32 vcc, v135, 0x4                        // wgMT/(SG*VW) == 4
s_cbranch_vccnz label_0047                         // shift d0, r=3, v=4
v_cmp_eq_u32 vcc, v135, 0x5                        // wgMT/(SG*VW) == 5
s_cbranch_vccnz label_0048                         // shift d0, r=3, v=5
v_cmp_eq_u32 vcc, v135, 0x6                        // wgMT/(SG*VW) == 6
s_cbranch_vccnz label_0049                         // shift d0, r=3, v=6
v_cmp_eq_u32 vcc, v135, 0x7                        // wgMT/(SG*VW) == 7
s_cbranch_vccnz label_0050                         // shift d0, r=3, v=7

/* shift d0 r=3 v=0 */
label_0043:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v0, v134, v0, offset:20            // permute edge values
ds_bpermute_b32 v1, v134, v1, offset:20            // permute edge values
ds_bpermute_b32 v2, v134, v2, offset:20            // permute edge values
ds_bpermute_b32 v3, v134, v3, offset:20            // permute edge values
ds_bpermute_b32 v32, v134, v32, offset:20          // permute edge values
ds_bpermute_b32 v33, v134, v33, offset:20          // permute edge values
ds_bpermute_b32 v34, v134, v34, offset:20          // permute edge values
ds_bpermute_b32 v35, v134, v35, offset:20          // permute edge values
ds_bpermute_b32 v64, v134, v64, offset:20          // permute edge values
ds_bpermute_b32 v65, v134, v65, offset:20          // permute edge values
ds_bpermute_b32 v66, v134, v66, offset:20          // permute edge values
ds_bpermute_b32 v67, v134, v67, offset:20          // permute edge values
ds_bpermute_b32 v96, v134, v96, offset:20          // permute edge values
ds_bpermute_b32 v97, v134, v97, offset:20          // permute edge values
ds_bpermute_b32 v98, v134, v98, offset:20          // permute edge values
ds_bpermute_b32 v99, v134, v99, offset:20          // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=3 v=1 */
label_0044:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v4, v134, v4, offset:20            // permute edge values
ds_bpermute_b32 v5, v134, v5, offset:20            // permute edge values
ds_bpermute_b32 v6, v134, v6, offset:20            // permute edge values
ds_bpermute_b32 v7, v134, v7, offset:20            // permute edge values
ds_bpermute_b32 v36, v134, v36, offset:20          // permute edge values
ds_bpermute_b32 v37, v134, v37, offset:20          // permute edge values
ds_bpermute_b32 v38, v134, v38, offset:20          // permute edge values
ds_bpermute_b32 v39, v134, v39, offset:20          // permute edge values
ds_bpermute_b32 v68, v134, v68, offset:20          // permute edge values
ds_bpermute_b32 v69, v134, v69, offset:20          // permute edge values
ds_bpermute_b32 v70, v134, v70, offset:20          // permute edge values
ds_bpermute_b32 v71, v134, v71, offset:20          // permute edge values
ds_bpermute_b32 v100, v134, v100, offset:20        // permute edge values
ds_bpermute_b32 v101, v134, v101, offset:20        // permute edge values
ds_bpermute_b32 v102, v134, v102, offset:20        // permute edge values
ds_bpermute_b32 v103, v134, v103, offset:20        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=3 v=2 */
label_0045:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v8, v134, v8, offset:20            // permute edge values
ds_bpermute_b32 v9, v134, v9, offset:20            // permute edge values
ds_bpermute_b32 v10, v134, v10, offset:20          // permute edge values
ds_bpermute_b32 v11, v134, v11, offset:20          // permute edge values
ds_bpermute_b32 v40, v134, v40, offset:20          // permute edge values
ds_bpermute_b32 v41, v134, v41, offset:20          // permute edge values
ds_bpermute_b32 v42, v134, v42, offset:20          // permute edge values
ds_bpermute_b32 v43, v134, v43, offset:20          // permute edge values
ds_bpermute_b32 v72, v134, v72, offset:20          // permute edge values
ds_bpermute_b32 v73, v134, v73, offset:20          // permute edge values
ds_bpermute_b32 v74, v134, v74, offset:20          // permute edge values
ds_bpermute_b32 v75, v134, v75, offset:20          // permute edge values
ds_bpermute_b32 v104, v134, v104, offset:20        // permute edge values
ds_bpermute_b32 v105, v134, v105, offset:20        // permute edge values
ds_bpermute_b32 v106, v134, v106, offset:20        // permute edge values
ds_bpermute_b32 v107, v134, v107, offset:20        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=3 v=3 */
label_0046:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v12, v134, v12, offset:20          // permute edge values
ds_bpermute_b32 v13, v134, v13, offset:20          // permute edge values
ds_bpermute_b32 v14, v134, v14, offset:20          // permute edge values
ds_bpermute_b32 v15, v134, v15, offset:20          // permute edge values
ds_bpermute_b32 v44, v134, v44, offset:20          // permute edge values
ds_bpermute_b32 v45, v134, v45, offset:20          // permute edge values
ds_bpermute_b32 v46, v134, v46, offset:20          // permute edge values
ds_bpermute_b32 v47, v134, v47, offset:20          // permute edge values
ds_bpermute_b32 v76, v134, v76, offset:20          // permute edge values
ds_bpermute_b32 v77, v134, v77, offset:20          // permute edge values
ds_bpermute_b32 v78, v134, v78, offset:20          // permute edge values
ds_bpermute_b32 v79, v134, v79, offset:20          // permute edge values
ds_bpermute_b32 v108, v134, v108, offset:20        // permute edge values
ds_bpermute_b32 v109, v134, v109, offset:20        // permute edge values
ds_bpermute_b32 v110, v134, v110, offset:20        // permute edge values
ds_bpermute_b32 v111, v134, v111, offset:20        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=3 v=4 */
label_0047:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v16, v134, v16, offset:20          // permute edge values
ds_bpermute_b32 v17, v134, v17, offset:20          // permute edge values
ds_bpermute_b32 v18, v134, v18, offset:20          // permute edge values
ds_bpermute_b32 v19, v134, v19, offset:20          // permute edge values
ds_bpermute_b32 v48, v134, v48, offset:20          // permute edge values
ds_bpermute_b32 v49, v134, v49, offset:20          // permute edge values
ds_bpermute_b32 v50, v134, v50, offset:20          // permute edge values
ds_bpermute_b32 v51, v134, v51, offset:20          // permute edge values
ds_bpermute_b32 v80, v134, v80, offset:20          // permute edge values
ds_bpermute_b32 v81, v134, v81, offset:20          // permute edge values
ds_bpermute_b32 v82, v134, v82, offset:20          // permute edge values
ds_bpermute_b32 v83, v134, v83, offset:20          // permute edge values
ds_bpermute_b32 v112, v134, v112, offset:20        // permute edge values
ds_bpermute_b32 v113, v134, v113, offset:20        // permute edge values
ds_bpermute_b32 v114, v134, v114, offset:20        // permute edge values
ds_bpermute_b32 v115, v134, v115, offset:20        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=3 v=5 */
label_0048:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v20, v134, v20, offset:20          // permute edge values
ds_bpermute_b32 v21, v134, v21, offset:20          // permute edge values
ds_bpermute_b32 v22, v134, v22, offset:20          // permute edge values
ds_bpermute_b32 v23, v134, v23, offset:20          // permute edge values
ds_bpermute_b32 v52, v134, v52, offset:20          // permute edge values
ds_bpermute_b32 v53, v134, v53, offset:20          // permute edge values
ds_bpermute_b32 v54, v134, v54, offset:20          // permute edge values
ds_bpermute_b32 v55, v134, v55, offset:20          // permute edge values
ds_bpermute_b32 v84, v134, v84, offset:20          // permute edge values
ds_bpermute_b32 v85, v134, v85, offset:20          // permute edge values
ds_bpermute_b32 v86, v134, v86, offset:20          // permute edge values
ds_bpermute_b32 v87, v134, v87, offset:20          // permute edge values
ds_bpermute_b32 v116, v134, v116, offset:20        // permute edge values
ds_bpermute_b32 v117, v134, v117, offset:20        // permute edge values
ds_bpermute_b32 v118, v134, v118, offset:20        // permute edge values
ds_bpermute_b32 v119, v134, v119, offset:20        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=3 v=6 */
label_0049:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v24, v134, v24, offset:20          // permute edge values
ds_bpermute_b32 v25, v134, v25, offset:20          // permute edge values
ds_bpermute_b32 v26, v134, v26, offset:20          // permute edge values
ds_bpermute_b32 v27, v134, v27, offset:20          // permute edge values
ds_bpermute_b32 v56, v134, v56, offset:20          // permute edge values
ds_bpermute_b32 v57, v134, v57, offset:20          // permute edge values
ds_bpermute_b32 v58, v134, v58, offset:20          // permute edge values
ds_bpermute_b32 v59, v134, v59, offset:20          // permute edge values
ds_bpermute_b32 v88, v134, v88, offset:20          // permute edge values
ds_bpermute_b32 v89, v134, v89, offset:20          // permute edge values
ds_bpermute_b32 v90, v134, v90, offset:20          // permute edge values
ds_bpermute_b32 v91, v134, v91, offset:20          // permute edge values
ds_bpermute_b32 v120, v134, v120, offset:20        // permute edge values
ds_bpermute_b32 v121, v134, v121, offset:20        // permute edge values
ds_bpermute_b32 v122, v134, v122, offset:20        // permute edge values
ds_bpermute_b32 v123, v134, v123, offset:20        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=3 v=7 */
label_0050:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v28, v134, v28, offset:20          // permute edge values
ds_bpermute_b32 v29, v134, v29, offset:20          // permute edge values
ds_bpermute_b32 v30, v134, v30, offset:20          // permute edge values
ds_bpermute_b32 v31, v134, v31, offset:20          // permute edge values
ds_bpermute_b32 v60, v134, v60, offset:20          // permute edge values
ds_bpermute_b32 v61, v134, v61, offset:20          // permute edge values
ds_bpermute_b32 v62, v134, v62, offset:20          // permute edge values
ds_bpermute_b32 v63, v134, v63, offset:20          // permute edge values
ds_bpermute_b32 v92, v134, v92, offset:20          // permute edge values
ds_bpermute_b32 v93, v134, v93, offset:20          // permute edge values
ds_bpermute_b32 v94, v134, v94, offset:20          // permute edge values
ds_bpermute_b32 v95, v134, v95, offset:20          // permute edge values
ds_bpermute_b32 v124, v134, v124, offset:20        // permute edge values
ds_bpermute_b32 v125, v134, v125, offset:20        // permute edge values
ds_bpermute_b32 v126, v134, v126, offset:20        // permute edge values
ds_bpermute_b32 v127, v134, v127, offset:20        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/******************************************/
/* shift d0 r=4                           */
/******************************************/
label_0051:
v_cmp_eq_u32 vcc, v135, 0x0                        // wgMT/(SG*VW) == 0
s_cbranch_vccnz label_0052                         // shift d0, r=4, v=0
v_cmp_eq_u32 vcc, v135, 0x1                        // wgMT/(SG*VW) == 1
s_cbranch_vccnz label_0053                         // shift d0, r=4, v=1
v_cmp_eq_u32 vcc, v135, 0x2                        // wgMT/(SG*VW) == 2
s_cbranch_vccnz label_0054                         // shift d0, r=4, v=2
v_cmp_eq_u32 vcc, v135, 0x3                        // wgMT/(SG*VW) == 3
s_cbranch_vccnz label_0055                         // shift d0, r=4, v=3
v_cmp_eq_u32 vcc, v135, 0x4                        // wgMT/(SG*VW) == 4
s_cbranch_vccnz label_0056                         // shift d0, r=4, v=4
v_cmp_eq_u32 vcc, v135, 0x5                        // wgMT/(SG*VW) == 5
s_cbranch_vccnz label_0057                         // shift d0, r=4, v=5
v_cmp_eq_u32 vcc, v135, 0x6                        // wgMT/(SG*VW) == 6
s_cbranch_vccnz label_0058                         // shift d0, r=4, v=6
v_cmp_eq_u32 vcc, v135, 0x7                        // wgMT/(SG*VW) == 7
s_cbranch_vccnz label_0059                         // shift d0, r=4, v=7

/* shift d0 r=4 v=0 */
label_0052:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v0, v134, v0, offset:16            // permute edge values
ds_bpermute_b32 v1, v134, v1, offset:16            // permute edge values
ds_bpermute_b32 v2, v134, v2, offset:16            // permute edge values
ds_bpermute_b32 v3, v134, v3, offset:16            // permute edge values
ds_bpermute_b32 v32, v134, v32, offset:16          // permute edge values
ds_bpermute_b32 v33, v134, v33, offset:16          // permute edge values
ds_bpermute_b32 v34, v134, v34, offset:16          // permute edge values
ds_bpermute_b32 v35, v134, v35, offset:16          // permute edge values
ds_bpermute_b32 v64, v134, v64, offset:16          // permute edge values
ds_bpermute_b32 v65, v134, v65, offset:16          // permute edge values
ds_bpermute_b32 v66, v134, v66, offset:16          // permute edge values
ds_bpermute_b32 v67, v134, v67, offset:16          // permute edge values
ds_bpermute_b32 v96, v134, v96, offset:16          // permute edge values
ds_bpermute_b32 v97, v134, v97, offset:16          // permute edge values
ds_bpermute_b32 v98, v134, v98, offset:16          // permute edge values
ds_bpermute_b32 v99, v134, v99, offset:16          // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=4 v=1 */
label_0053:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v4, v134, v4, offset:16            // permute edge values
ds_bpermute_b32 v5, v134, v5, offset:16            // permute edge values
ds_bpermute_b32 v6, v134, v6, offset:16            // permute edge values
ds_bpermute_b32 v7, v134, v7, offset:16            // permute edge values
ds_bpermute_b32 v36, v134, v36, offset:16          // permute edge values
ds_bpermute_b32 v37, v134, v37, offset:16          // permute edge values
ds_bpermute_b32 v38, v134, v38, offset:16          // permute edge values
ds_bpermute_b32 v39, v134, v39, offset:16          // permute edge values
ds_bpermute_b32 v68, v134, v68, offset:16          // permute edge values
ds_bpermute_b32 v69, v134, v69, offset:16          // permute edge values
ds_bpermute_b32 v70, v134, v70, offset:16          // permute edge values
ds_bpermute_b32 v71, v134, v71, offset:16          // permute edge values
ds_bpermute_b32 v100, v134, v100, offset:16        // permute edge values
ds_bpermute_b32 v101, v134, v101, offset:16        // permute edge values
ds_bpermute_b32 v102, v134, v102, offset:16        // permute edge values
ds_bpermute_b32 v103, v134, v103, offset:16        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=4 v=2 */
label_0054:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v8, v134, v8, offset:16            // permute edge values
ds_bpermute_b32 v9, v134, v9, offset:16            // permute edge values
ds_bpermute_b32 v10, v134, v10, offset:16          // permute edge values
ds_bpermute_b32 v11, v134, v11, offset:16          // permute edge values
ds_bpermute_b32 v40, v134, v40, offset:16          // permute edge values
ds_bpermute_b32 v41, v134, v41, offset:16          // permute edge values
ds_bpermute_b32 v42, v134, v42, offset:16          // permute edge values
ds_bpermute_b32 v43, v134, v43, offset:16          // permute edge values
ds_bpermute_b32 v72, v134, v72, offset:16          // permute edge values
ds_bpermute_b32 v73, v134, v73, offset:16          // permute edge values
ds_bpermute_b32 v74, v134, v74, offset:16          // permute edge values
ds_bpermute_b32 v75, v134, v75, offset:16          // permute edge values
ds_bpermute_b32 v104, v134, v104, offset:16        // permute edge values
ds_bpermute_b32 v105, v134, v105, offset:16        // permute edge values
ds_bpermute_b32 v106, v134, v106, offset:16        // permute edge values
ds_bpermute_b32 v107, v134, v107, offset:16        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=4 v=3 */
label_0055:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v12, v134, v12, offset:16          // permute edge values
ds_bpermute_b32 v13, v134, v13, offset:16          // permute edge values
ds_bpermute_b32 v14, v134, v14, offset:16          // permute edge values
ds_bpermute_b32 v15, v134, v15, offset:16          // permute edge values
ds_bpermute_b32 v44, v134, v44, offset:16          // permute edge values
ds_bpermute_b32 v45, v134, v45, offset:16          // permute edge values
ds_bpermute_b32 v46, v134, v46, offset:16          // permute edge values
ds_bpermute_b32 v47, v134, v47, offset:16          // permute edge values
ds_bpermute_b32 v76, v134, v76, offset:16          // permute edge values
ds_bpermute_b32 v77, v134, v77, offset:16          // permute edge values
ds_bpermute_b32 v78, v134, v78, offset:16          // permute edge values
ds_bpermute_b32 v79, v134, v79, offset:16          // permute edge values
ds_bpermute_b32 v108, v134, v108, offset:16        // permute edge values
ds_bpermute_b32 v109, v134, v109, offset:16        // permute edge values
ds_bpermute_b32 v110, v134, v110, offset:16        // permute edge values
ds_bpermute_b32 v111, v134, v111, offset:16        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=4 v=4 */
label_0056:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v16, v134, v16, offset:16          // permute edge values
ds_bpermute_b32 v17, v134, v17, offset:16          // permute edge values
ds_bpermute_b32 v18, v134, v18, offset:16          // permute edge values
ds_bpermute_b32 v19, v134, v19, offset:16          // permute edge values
ds_bpermute_b32 v48, v134, v48, offset:16          // permute edge values
ds_bpermute_b32 v49, v134, v49, offset:16          // permute edge values
ds_bpermute_b32 v50, v134, v50, offset:16          // permute edge values
ds_bpermute_b32 v51, v134, v51, offset:16          // permute edge values
ds_bpermute_b32 v80, v134, v80, offset:16          // permute edge values
ds_bpermute_b32 v81, v134, v81, offset:16          // permute edge values
ds_bpermute_b32 v82, v134, v82, offset:16          // permute edge values
ds_bpermute_b32 v83, v134, v83, offset:16          // permute edge values
ds_bpermute_b32 v112, v134, v112, offset:16        // permute edge values
ds_bpermute_b32 v113, v134, v113, offset:16        // permute edge values
ds_bpermute_b32 v114, v134, v114, offset:16        // permute edge values
ds_bpermute_b32 v115, v134, v115, offset:16        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=4 v=5 */
label_0057:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v20, v134, v20, offset:16          // permute edge values
ds_bpermute_b32 v21, v134, v21, offset:16          // permute edge values
ds_bpermute_b32 v22, v134, v22, offset:16          // permute edge values
ds_bpermute_b32 v23, v134, v23, offset:16          // permute edge values
ds_bpermute_b32 v52, v134, v52, offset:16          // permute edge values
ds_bpermute_b32 v53, v134, v53, offset:16          // permute edge values
ds_bpermute_b32 v54, v134, v54, offset:16          // permute edge values
ds_bpermute_b32 v55, v134, v55, offset:16          // permute edge values
ds_bpermute_b32 v84, v134, v84, offset:16          // permute edge values
ds_bpermute_b32 v85, v134, v85, offset:16          // permute edge values
ds_bpermute_b32 v86, v134, v86, offset:16          // permute edge values
ds_bpermute_b32 v87, v134, v87, offset:16          // permute edge values
ds_bpermute_b32 v116, v134, v116, offset:16        // permute edge values
ds_bpermute_b32 v117, v134, v117, offset:16        // permute edge values
ds_bpermute_b32 v118, v134, v118, offset:16        // permute edge values
ds_bpermute_b32 v119, v134, v119, offset:16        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=4 v=6 */
label_0058:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v24, v134, v24, offset:16          // permute edge values
ds_bpermute_b32 v25, v134, v25, offset:16          // permute edge values
ds_bpermute_b32 v26, v134, v26, offset:16          // permute edge values
ds_bpermute_b32 v27, v134, v27, offset:16          // permute edge values
ds_bpermute_b32 v56, v134, v56, offset:16          // permute edge values
ds_bpermute_b32 v57, v134, v57, offset:16          // permute edge values
ds_bpermute_b32 v58, v134, v58, offset:16          // permute edge values
ds_bpermute_b32 v59, v134, v59, offset:16          // permute edge values
ds_bpermute_b32 v88, v134, v88, offset:16          // permute edge values
ds_bpermute_b32 v89, v134, v89, offset:16          // permute edge values
ds_bpermute_b32 v90, v134, v90, offset:16          // permute edge values
ds_bpermute_b32 v91, v134, v91, offset:16          // permute edge values
ds_bpermute_b32 v120, v134, v120, offset:16        // permute edge values
ds_bpermute_b32 v121, v134, v121, offset:16        // permute edge values
ds_bpermute_b32 v122, v134, v122, offset:16        // permute edge values
ds_bpermute_b32 v123, v134, v123, offset:16        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=4 v=7 */
label_0059:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v28, v134, v28, offset:16          // permute edge values
ds_bpermute_b32 v29, v134, v29, offset:16          // permute edge values
ds_bpermute_b32 v30, v134, v30, offset:16          // permute edge values
ds_bpermute_b32 v31, v134, v31, offset:16          // permute edge values
ds_bpermute_b32 v60, v134, v60, offset:16          // permute edge values
ds_bpermute_b32 v61, v134, v61, offset:16          // permute edge values
ds_bpermute_b32 v62, v134, v62, offset:16          // permute edge values
ds_bpermute_b32 v63, v134, v63, offset:16          // permute edge values
ds_bpermute_b32 v92, v134, v92, offset:16          // permute edge values
ds_bpermute_b32 v93, v134, v93, offset:16          // permute edge values
ds_bpermute_b32 v94, v134, v94, offset:16          // permute edge values
ds_bpermute_b32 v95, v134, v95, offset:16          // permute edge values
ds_bpermute_b32 v124, v134, v124, offset:16        // permute edge values
ds_bpermute_b32 v125, v134, v125, offset:16        // permute edge values
ds_bpermute_b32 v126, v134, v126, offset:16        // permute edge values
ds_bpermute_b32 v127, v134, v127, offset:16        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/******************************************/
/* shift d0 r=5                           */
/******************************************/
label_0060:
v_cmp_eq_u32 vcc, v135, 0x0                        // wgMT/(SG*VW) == 0
s_cbranch_vccnz label_0061                         // shift d0, r=5, v=0
v_cmp_eq_u32 vcc, v135, 0x1                        // wgMT/(SG*VW) == 1
s_cbranch_vccnz label_0062                         // shift d0, r=5, v=1
v_cmp_eq_u32 vcc, v135, 0x2                        // wgMT/(SG*VW) == 2
s_cbranch_vccnz label_0063                         // shift d0, r=5, v=2
v_cmp_eq_u32 vcc, v135, 0x3                        // wgMT/(SG*VW) == 3
s_cbranch_vccnz label_0064                         // shift d0, r=5, v=3
v_cmp_eq_u32 vcc, v135, 0x4                        // wgMT/(SG*VW) == 4
s_cbranch_vccnz label_0065                         // shift d0, r=5, v=4
v_cmp_eq_u32 vcc, v135, 0x5                        // wgMT/(SG*VW) == 5
s_cbranch_vccnz label_0066                         // shift d0, r=5, v=5
v_cmp_eq_u32 vcc, v135, 0x6                        // wgMT/(SG*VW) == 6
s_cbranch_vccnz label_0067                         // shift d0, r=5, v=6
v_cmp_eq_u32 vcc, v135, 0x7                        // wgMT/(SG*VW) == 7
s_cbranch_vccnz label_0068                         // shift d0, r=5, v=7

/* shift d0 r=5 v=0 */
label_0061:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v0, v134, v0, offset:12            // permute edge values
ds_bpermute_b32 v1, v134, v1, offset:12            // permute edge values
ds_bpermute_b32 v2, v134, v2, offset:12            // permute edge values
ds_bpermute_b32 v3, v134, v3, offset:12            // permute edge values
ds_bpermute_b32 v32, v134, v32, offset:12          // permute edge values
ds_bpermute_b32 v33, v134, v33, offset:12          // permute edge values
ds_bpermute_b32 v34, v134, v34, offset:12          // permute edge values
ds_bpermute_b32 v35, v134, v35, offset:12          // permute edge values
ds_bpermute_b32 v64, v134, v64, offset:12          // permute edge values
ds_bpermute_b32 v65, v134, v65, offset:12          // permute edge values
ds_bpermute_b32 v66, v134, v66, offset:12          // permute edge values
ds_bpermute_b32 v67, v134, v67, offset:12          // permute edge values
ds_bpermute_b32 v96, v134, v96, offset:12          // permute edge values
ds_bpermute_b32 v97, v134, v97, offset:12          // permute edge values
ds_bpermute_b32 v98, v134, v98, offset:12          // permute edge values
ds_bpermute_b32 v99, v134, v99, offset:12          // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=5 v=1 */
label_0062:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v4, v134, v4, offset:12            // permute edge values
ds_bpermute_b32 v5, v134, v5, offset:12            // permute edge values
ds_bpermute_b32 v6, v134, v6, offset:12            // permute edge values
ds_bpermute_b32 v7, v134, v7, offset:12            // permute edge values
ds_bpermute_b32 v36, v134, v36, offset:12          // permute edge values
ds_bpermute_b32 v37, v134, v37, offset:12          // permute edge values
ds_bpermute_b32 v38, v134, v38, offset:12          // permute edge values
ds_bpermute_b32 v39, v134, v39, offset:12          // permute edge values
ds_bpermute_b32 v68, v134, v68, offset:12          // permute edge values
ds_bpermute_b32 v69, v134, v69, offset:12          // permute edge values
ds_bpermute_b32 v70, v134, v70, offset:12          // permute edge values
ds_bpermute_b32 v71, v134, v71, offset:12          // permute edge values
ds_bpermute_b32 v100, v134, v100, offset:12        // permute edge values
ds_bpermute_b32 v101, v134, v101, offset:12        // permute edge values
ds_bpermute_b32 v102, v134, v102, offset:12        // permute edge values
ds_bpermute_b32 v103, v134, v103, offset:12        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=5 v=2 */
label_0063:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v8, v134, v8, offset:12            // permute edge values
ds_bpermute_b32 v9, v134, v9, offset:12            // permute edge values
ds_bpermute_b32 v10, v134, v10, offset:12          // permute edge values
ds_bpermute_b32 v11, v134, v11, offset:12          // permute edge values
ds_bpermute_b32 v40, v134, v40, offset:12          // permute edge values
ds_bpermute_b32 v41, v134, v41, offset:12          // permute edge values
ds_bpermute_b32 v42, v134, v42, offset:12          // permute edge values
ds_bpermute_b32 v43, v134, v43, offset:12          // permute edge values
ds_bpermute_b32 v72, v134, v72, offset:12          // permute edge values
ds_bpermute_b32 v73, v134, v73, offset:12          // permute edge values
ds_bpermute_b32 v74, v134, v74, offset:12          // permute edge values
ds_bpermute_b32 v75, v134, v75, offset:12          // permute edge values
ds_bpermute_b32 v104, v134, v104, offset:12        // permute edge values
ds_bpermute_b32 v105, v134, v105, offset:12        // permute edge values
ds_bpermute_b32 v106, v134, v106, offset:12        // permute edge values
ds_bpermute_b32 v107, v134, v107, offset:12        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=5 v=3 */
label_0064:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v12, v134, v12, offset:12          // permute edge values
ds_bpermute_b32 v13, v134, v13, offset:12          // permute edge values
ds_bpermute_b32 v14, v134, v14, offset:12          // permute edge values
ds_bpermute_b32 v15, v134, v15, offset:12          // permute edge values
ds_bpermute_b32 v44, v134, v44, offset:12          // permute edge values
ds_bpermute_b32 v45, v134, v45, offset:12          // permute edge values
ds_bpermute_b32 v46, v134, v46, offset:12          // permute edge values
ds_bpermute_b32 v47, v134, v47, offset:12          // permute edge values
ds_bpermute_b32 v76, v134, v76, offset:12          // permute edge values
ds_bpermute_b32 v77, v134, v77, offset:12          // permute edge values
ds_bpermute_b32 v78, v134, v78, offset:12          // permute edge values
ds_bpermute_b32 v79, v134, v79, offset:12          // permute edge values
ds_bpermute_b32 v108, v134, v108, offset:12        // permute edge values
ds_bpermute_b32 v109, v134, v109, offset:12        // permute edge values
ds_bpermute_b32 v110, v134, v110, offset:12        // permute edge values
ds_bpermute_b32 v111, v134, v111, offset:12        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=5 v=4 */
label_0065:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v16, v134, v16, offset:12          // permute edge values
ds_bpermute_b32 v17, v134, v17, offset:12          // permute edge values
ds_bpermute_b32 v18, v134, v18, offset:12          // permute edge values
ds_bpermute_b32 v19, v134, v19, offset:12          // permute edge values
ds_bpermute_b32 v48, v134, v48, offset:12          // permute edge values
ds_bpermute_b32 v49, v134, v49, offset:12          // permute edge values
ds_bpermute_b32 v50, v134, v50, offset:12          // permute edge values
ds_bpermute_b32 v51, v134, v51, offset:12          // permute edge values
ds_bpermute_b32 v80, v134, v80, offset:12          // permute edge values
ds_bpermute_b32 v81, v134, v81, offset:12          // permute edge values
ds_bpermute_b32 v82, v134, v82, offset:12          // permute edge values
ds_bpermute_b32 v83, v134, v83, offset:12          // permute edge values
ds_bpermute_b32 v112, v134, v112, offset:12        // permute edge values
ds_bpermute_b32 v113, v134, v113, offset:12        // permute edge values
ds_bpermute_b32 v114, v134, v114, offset:12        // permute edge values
ds_bpermute_b32 v115, v134, v115, offset:12        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=5 v=5 */
label_0066:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v20, v134, v20, offset:12          // permute edge values
ds_bpermute_b32 v21, v134, v21, offset:12          // permute edge values
ds_bpermute_b32 v22, v134, v22, offset:12          // permute edge values
ds_bpermute_b32 v23, v134, v23, offset:12          // permute edge values
ds_bpermute_b32 v52, v134, v52, offset:12          // permute edge values
ds_bpermute_b32 v53, v134, v53, offset:12          // permute edge values
ds_bpermute_b32 v54, v134, v54, offset:12          // permute edge values
ds_bpermute_b32 v55, v134, v55, offset:12          // permute edge values
ds_bpermute_b32 v84, v134, v84, offset:12          // permute edge values
ds_bpermute_b32 v85, v134, v85, offset:12          // permute edge values
ds_bpermute_b32 v86, v134, v86, offset:12          // permute edge values
ds_bpermute_b32 v87, v134, v87, offset:12          // permute edge values
ds_bpermute_b32 v116, v134, v116, offset:12        // permute edge values
ds_bpermute_b32 v117, v134, v117, offset:12        // permute edge values
ds_bpermute_b32 v118, v134, v118, offset:12        // permute edge values
ds_bpermute_b32 v119, v134, v119, offset:12        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=5 v=6 */
label_0067:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v24, v134, v24, offset:12          // permute edge values
ds_bpermute_b32 v25, v134, v25, offset:12          // permute edge values
ds_bpermute_b32 v26, v134, v26, offset:12          // permute edge values
ds_bpermute_b32 v27, v134, v27, offset:12          // permute edge values
ds_bpermute_b32 v56, v134, v56, offset:12          // permute edge values
ds_bpermute_b32 v57, v134, v57, offset:12          // permute edge values
ds_bpermute_b32 v58, v134, v58, offset:12          // permute edge values
ds_bpermute_b32 v59, v134, v59, offset:12          // permute edge values
ds_bpermute_b32 v88, v134, v88, offset:12          // permute edge values
ds_bpermute_b32 v89, v134, v89, offset:12          // permute edge values
ds_bpermute_b32 v90, v134, v90, offset:12          // permute edge values
ds_bpermute_b32 v91, v134, v91, offset:12          // permute edge values
ds_bpermute_b32 v120, v134, v120, offset:12        // permute edge values
ds_bpermute_b32 v121, v134, v121, offset:12        // permute edge values
ds_bpermute_b32 v122, v134, v122, offset:12        // permute edge values
ds_bpermute_b32 v123, v134, v123, offset:12        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=5 v=7 */
label_0068:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v28, v134, v28, offset:12          // permute edge values
ds_bpermute_b32 v29, v134, v29, offset:12          // permute edge values
ds_bpermute_b32 v30, v134, v30, offset:12          // permute edge values
ds_bpermute_b32 v31, v134, v31, offset:12          // permute edge values
ds_bpermute_b32 v60, v134, v60, offset:12          // permute edge values
ds_bpermute_b32 v61, v134, v61, offset:12          // permute edge values
ds_bpermute_b32 v62, v134, v62, offset:12          // permute edge values
ds_bpermute_b32 v63, v134, v63, offset:12          // permute edge values
ds_bpermute_b32 v92, v134, v92, offset:12          // permute edge values
ds_bpermute_b32 v93, v134, v93, offset:12          // permute edge values
ds_bpermute_b32 v94, v134, v94, offset:12          // permute edge values
ds_bpermute_b32 v95, v134, v95, offset:12          // permute edge values
ds_bpermute_b32 v124, v134, v124, offset:12        // permute edge values
ds_bpermute_b32 v125, v134, v125, offset:12        // permute edge values
ds_bpermute_b32 v126, v134, v126, offset:12        // permute edge values
ds_bpermute_b32 v127, v134, v127, offset:12        // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/******************************************/
/* shift d0 r=6                           */
/******************************************/
label_0069:
v_cmp_eq_u32 vcc, v135, 0x0                        // wgMT/(SG*VW) == 0
s_cbranch_vccnz label_0070                         // shift d0, r=6, v=0
v_cmp_eq_u32 vcc, v135, 0x1                        // wgMT/(SG*VW) == 1
s_cbranch_vccnz label_0071                         // shift d0, r=6, v=1
v_cmp_eq_u32 vcc, v135, 0x2                        // wgMT/(SG*VW) == 2
s_cbranch_vccnz label_0072                         // shift d0, r=6, v=2
v_cmp_eq_u32 vcc, v135, 0x3                        // wgMT/(SG*VW) == 3
s_cbranch_vccnz label_0073                         // shift d0, r=6, v=3
v_cmp_eq_u32 vcc, v135, 0x4                        // wgMT/(SG*VW) == 4
s_cbranch_vccnz label_0074                         // shift d0, r=6, v=4
v_cmp_eq_u32 vcc, v135, 0x5                        // wgMT/(SG*VW) == 5
s_cbranch_vccnz label_0075                         // shift d0, r=6, v=5
v_cmp_eq_u32 vcc, v135, 0x6                        // wgMT/(SG*VW) == 6
s_cbranch_vccnz label_0076                         // shift d0, r=6, v=6
v_cmp_eq_u32 vcc, v135, 0x7                        // wgMT/(SG*VW) == 7
s_cbranch_vccnz label_0077                         // shift d0, r=6, v=7

/* shift d0 r=6 v=0 */
label_0070:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v0, v134, v0, offset:8             // permute edge values
ds_bpermute_b32 v1, v134, v1, offset:8             // permute edge values
ds_bpermute_b32 v2, v134, v2, offset:8             // permute edge values
ds_bpermute_b32 v3, v134, v3, offset:8             // permute edge values
ds_bpermute_b32 v32, v134, v32, offset:8           // permute edge values
ds_bpermute_b32 v33, v134, v33, offset:8           // permute edge values
ds_bpermute_b32 v34, v134, v34, offset:8           // permute edge values
ds_bpermute_b32 v35, v134, v35, offset:8           // permute edge values
ds_bpermute_b32 v64, v134, v64, offset:8           // permute edge values
ds_bpermute_b32 v65, v134, v65, offset:8           // permute edge values
ds_bpermute_b32 v66, v134, v66, offset:8           // permute edge values
ds_bpermute_b32 v67, v134, v67, offset:8           // permute edge values
ds_bpermute_b32 v96, v134, v96, offset:8           // permute edge values
ds_bpermute_b32 v97, v134, v97, offset:8           // permute edge values
ds_bpermute_b32 v98, v134, v98, offset:8           // permute edge values
ds_bpermute_b32 v99, v134, v99, offset:8           // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=6 v=1 */
label_0071:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v4, v134, v4, offset:8             // permute edge values
ds_bpermute_b32 v5, v134, v5, offset:8             // permute edge values
ds_bpermute_b32 v6, v134, v6, offset:8             // permute edge values
ds_bpermute_b32 v7, v134, v7, offset:8             // permute edge values
ds_bpermute_b32 v36, v134, v36, offset:8           // permute edge values
ds_bpermute_b32 v37, v134, v37, offset:8           // permute edge values
ds_bpermute_b32 v38, v134, v38, offset:8           // permute edge values
ds_bpermute_b32 v39, v134, v39, offset:8           // permute edge values
ds_bpermute_b32 v68, v134, v68, offset:8           // permute edge values
ds_bpermute_b32 v69, v134, v69, offset:8           // permute edge values
ds_bpermute_b32 v70, v134, v70, offset:8           // permute edge values
ds_bpermute_b32 v71, v134, v71, offset:8           // permute edge values
ds_bpermute_b32 v100, v134, v100, offset:8         // permute edge values
ds_bpermute_b32 v101, v134, v101, offset:8         // permute edge values
ds_bpermute_b32 v102, v134, v102, offset:8         // permute edge values
ds_bpermute_b32 v103, v134, v103, offset:8         // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=6 v=2 */
label_0072:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v8, v134, v8, offset:8             // permute edge values
ds_bpermute_b32 v9, v134, v9, offset:8             // permute edge values
ds_bpermute_b32 v10, v134, v10, offset:8           // permute edge values
ds_bpermute_b32 v11, v134, v11, offset:8           // permute edge values
ds_bpermute_b32 v40, v134, v40, offset:8           // permute edge values
ds_bpermute_b32 v41, v134, v41, offset:8           // permute edge values
ds_bpermute_b32 v42, v134, v42, offset:8           // permute edge values
ds_bpermute_b32 v43, v134, v43, offset:8           // permute edge values
ds_bpermute_b32 v72, v134, v72, offset:8           // permute edge values
ds_bpermute_b32 v73, v134, v73, offset:8           // permute edge values
ds_bpermute_b32 v74, v134, v74, offset:8           // permute edge values
ds_bpermute_b32 v75, v134, v75, offset:8           // permute edge values
ds_bpermute_b32 v104, v134, v104, offset:8         // permute edge values
ds_bpermute_b32 v105, v134, v105, offset:8         // permute edge values
ds_bpermute_b32 v106, v134, v106, offset:8         // permute edge values
ds_bpermute_b32 v107, v134, v107, offset:8         // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=6 v=3 */
label_0073:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v12, v134, v12, offset:8           // permute edge values
ds_bpermute_b32 v13, v134, v13, offset:8           // permute edge values
ds_bpermute_b32 v14, v134, v14, offset:8           // permute edge values
ds_bpermute_b32 v15, v134, v15, offset:8           // permute edge values
ds_bpermute_b32 v44, v134, v44, offset:8           // permute edge values
ds_bpermute_b32 v45, v134, v45, offset:8           // permute edge values
ds_bpermute_b32 v46, v134, v46, offset:8           // permute edge values
ds_bpermute_b32 v47, v134, v47, offset:8           // permute edge values
ds_bpermute_b32 v76, v134, v76, offset:8           // permute edge values
ds_bpermute_b32 v77, v134, v77, offset:8           // permute edge values
ds_bpermute_b32 v78, v134, v78, offset:8           // permute edge values
ds_bpermute_b32 v79, v134, v79, offset:8           // permute edge values
ds_bpermute_b32 v108, v134, v108, offset:8         // permute edge values
ds_bpermute_b32 v109, v134, v109, offset:8         // permute edge values
ds_bpermute_b32 v110, v134, v110, offset:8         // permute edge values
ds_bpermute_b32 v111, v134, v111, offset:8         // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=6 v=4 */
label_0074:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v16, v134, v16, offset:8           // permute edge values
ds_bpermute_b32 v17, v134, v17, offset:8           // permute edge values
ds_bpermute_b32 v18, v134, v18, offset:8           // permute edge values
ds_bpermute_b32 v19, v134, v19, offset:8           // permute edge values
ds_bpermute_b32 v48, v134, v48, offset:8           // permute edge values
ds_bpermute_b32 v49, v134, v49, offset:8           // permute edge values
ds_bpermute_b32 v50, v134, v50, offset:8           // permute edge values
ds_bpermute_b32 v51, v134, v51, offset:8           // permute edge values
ds_bpermute_b32 v80, v134, v80, offset:8           // permute edge values
ds_bpermute_b32 v81, v134, v81, offset:8           // permute edge values
ds_bpermute_b32 v82, v134, v82, offset:8           // permute edge values
ds_bpermute_b32 v83, v134, v83, offset:8           // permute edge values
ds_bpermute_b32 v112, v134, v112, offset:8         // permute edge values
ds_bpermute_b32 v113, v134, v113, offset:8         // permute edge values
ds_bpermute_b32 v114, v134, v114, offset:8         // permute edge values
ds_bpermute_b32 v115, v134, v115, offset:8         // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=6 v=5 */
label_0075:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v20, v134, v20, offset:8           // permute edge values
ds_bpermute_b32 v21, v134, v21, offset:8           // permute edge values
ds_bpermute_b32 v22, v134, v22, offset:8           // permute edge values
ds_bpermute_b32 v23, v134, v23, offset:8           // permute edge values
ds_bpermute_b32 v52, v134, v52, offset:8           // permute edge values
ds_bpermute_b32 v53, v134, v53, offset:8           // permute edge values
ds_bpermute_b32 v54, v134, v54, offset:8           // permute edge values
ds_bpermute_b32 v55, v134, v55, offset:8           // permute edge values
ds_bpermute_b32 v84, v134, v84, offset:8           // permute edge values
ds_bpermute_b32 v85, v134, v85, offset:8           // permute edge values
ds_bpermute_b32 v86, v134, v86, offset:8           // permute edge values
ds_bpermute_b32 v87, v134, v87, offset:8           // permute edge values
ds_bpermute_b32 v116, v134, v116, offset:8         // permute edge values
ds_bpermute_b32 v117, v134, v117, offset:8         // permute edge values
ds_bpermute_b32 v118, v134, v118, offset:8         // permute edge values
ds_bpermute_b32 v119, v134, v119, offset:8         // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=6 v=6 */
label_0076:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v24, v134, v24, offset:8           // permute edge values
ds_bpermute_b32 v25, v134, v25, offset:8           // permute edge values
ds_bpermute_b32 v26, v134, v26, offset:8           // permute edge values
ds_bpermute_b32 v27, v134, v27, offset:8           // permute edge values
ds_bpermute_b32 v56, v134, v56, offset:8           // permute edge values
ds_bpermute_b32 v57, v134, v57, offset:8           // permute edge values
ds_bpermute_b32 v58, v134, v58, offset:8           // permute edge values
ds_bpermute_b32 v59, v134, v59, offset:8           // permute edge values
ds_bpermute_b32 v88, v134, v88, offset:8           // permute edge values
ds_bpermute_b32 v89, v134, v89, offset:8           // permute edge values
ds_bpermute_b32 v90, v134, v90, offset:8           // permute edge values
ds_bpermute_b32 v91, v134, v91, offset:8           // permute edge values
ds_bpermute_b32 v120, v134, v120, offset:8         // permute edge values
ds_bpermute_b32 v121, v134, v121, offset:8         // permute edge values
ds_bpermute_b32 v122, v134, v122, offset:8         // permute edge values
ds_bpermute_b32 v123, v134, v123, offset:8         // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=6 v=7 */
label_0077:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v28, v134, v28, offset:8           // permute edge values
ds_bpermute_b32 v29, v134, v29, offset:8           // permute edge values
ds_bpermute_b32 v30, v134, v30, offset:8           // permute edge values
ds_bpermute_b32 v31, v134, v31, offset:8           // permute edge values
ds_bpermute_b32 v60, v134, v60, offset:8           // permute edge values
ds_bpermute_b32 v61, v134, v61, offset:8           // permute edge values
ds_bpermute_b32 v62, v134, v62, offset:8           // permute edge values
ds_bpermute_b32 v63, v134, v63, offset:8           // permute edge values
ds_bpermute_b32 v92, v134, v92, offset:8           // permute edge values
ds_bpermute_b32 v93, v134, v93, offset:8           // permute edge values
ds_bpermute_b32 v94, v134, v94, offset:8           // permute edge values
ds_bpermute_b32 v95, v134, v95, offset:8           // permute edge values
ds_bpermute_b32 v124, v134, v124, offset:8         // permute edge values
ds_bpermute_b32 v125, v134, v125, offset:8         // permute edge values
ds_bpermute_b32 v126, v134, v126, offset:8         // permute edge values
ds_bpermute_b32 v127, v134, v127, offset:8         // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/******************************************/
/* shift d0 r=7                           */
/******************************************/
label_0078:
v_cmp_eq_u32 vcc, v135, 0x0                        // wgMT/(SG*VW) == 0
s_cbranch_vccnz label_0079                         // shift d0, r=7, v=0
v_cmp_eq_u32 vcc, v135, 0x1                        // wgMT/(SG*VW) == 1
s_cbranch_vccnz label_0080                         // shift d0, r=7, v=1
v_cmp_eq_u32 vcc, v135, 0x2                        // wgMT/(SG*VW) == 2
s_cbranch_vccnz label_0081                         // shift d0, r=7, v=2
v_cmp_eq_u32 vcc, v135, 0x3                        // wgMT/(SG*VW) == 3
s_cbranch_vccnz label_0082                         // shift d0, r=7, v=3
v_cmp_eq_u32 vcc, v135, 0x4                        // wgMT/(SG*VW) == 4
s_cbranch_vccnz label_0083                         // shift d0, r=7, v=4
v_cmp_eq_u32 vcc, v135, 0x5                        // wgMT/(SG*VW) == 5
s_cbranch_vccnz label_0084                         // shift d0, r=7, v=5
v_cmp_eq_u32 vcc, v135, 0x6                        // wgMT/(SG*VW) == 6
s_cbranch_vccnz label_0085                         // shift d0, r=7, v=6
v_cmp_eq_u32 vcc, v135, 0x7                        // wgMT/(SG*VW) == 7
s_cbranch_vccnz label_0086                         // shift d0, r=7, v=7

/* shift d0 r=7 v=0 */
label_0079:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v0, v134, v0, offset:4             // permute edge values
ds_bpermute_b32 v1, v134, v1, offset:4             // permute edge values
ds_bpermute_b32 v2, v134, v2, offset:4             // permute edge values
ds_bpermute_b32 v3, v134, v3, offset:4             // permute edge values
ds_bpermute_b32 v32, v134, v32, offset:4           // permute edge values
ds_bpermute_b32 v33, v134, v33, offset:4           // permute edge values
ds_bpermute_b32 v34, v134, v34, offset:4           // permute edge values
ds_bpermute_b32 v35, v134, v35, offset:4           // permute edge values
ds_bpermute_b32 v64, v134, v64, offset:4           // permute edge values
ds_bpermute_b32 v65, v134, v65, offset:4           // permute edge values
ds_bpermute_b32 v66, v134, v66, offset:4           // permute edge values
ds_bpermute_b32 v67, v134, v67, offset:4           // permute edge values
ds_bpermute_b32 v96, v134, v96, offset:4           // permute edge values
ds_bpermute_b32 v97, v134, v97, offset:4           // permute edge values
ds_bpermute_b32 v98, v134, v98, offset:4           // permute edge values
ds_bpermute_b32 v99, v134, v99, offset:4           // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=7 v=1 */
label_0080:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v4, v134, v4, offset:4             // permute edge values
ds_bpermute_b32 v5, v134, v5, offset:4             // permute edge values
ds_bpermute_b32 v6, v134, v6, offset:4             // permute edge values
ds_bpermute_b32 v7, v134, v7, offset:4             // permute edge values
ds_bpermute_b32 v36, v134, v36, offset:4           // permute edge values
ds_bpermute_b32 v37, v134, v37, offset:4           // permute edge values
ds_bpermute_b32 v38, v134, v38, offset:4           // permute edge values
ds_bpermute_b32 v39, v134, v39, offset:4           // permute edge values
ds_bpermute_b32 v68, v134, v68, offset:4           // permute edge values
ds_bpermute_b32 v69, v134, v69, offset:4           // permute edge values
ds_bpermute_b32 v70, v134, v70, offset:4           // permute edge values
ds_bpermute_b32 v71, v134, v71, offset:4           // permute edge values
ds_bpermute_b32 v100, v134, v100, offset:4         // permute edge values
ds_bpermute_b32 v101, v134, v101, offset:4         // permute edge values
ds_bpermute_b32 v102, v134, v102, offset:4         // permute edge values
ds_bpermute_b32 v103, v134, v103, offset:4         // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=7 v=2 */
label_0081:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v8, v134, v8, offset:4             // permute edge values
ds_bpermute_b32 v9, v134, v9, offset:4             // permute edge values
ds_bpermute_b32 v10, v134, v10, offset:4           // permute edge values
ds_bpermute_b32 v11, v134, v11, offset:4           // permute edge values
ds_bpermute_b32 v40, v134, v40, offset:4           // permute edge values
ds_bpermute_b32 v41, v134, v41, offset:4           // permute edge values
ds_bpermute_b32 v42, v134, v42, offset:4           // permute edge values
ds_bpermute_b32 v43, v134, v43, offset:4           // permute edge values
ds_bpermute_b32 v72, v134, v72, offset:4           // permute edge values
ds_bpermute_b32 v73, v134, v73, offset:4           // permute edge values
ds_bpermute_b32 v74, v134, v74, offset:4           // permute edge values
ds_bpermute_b32 v75, v134, v75, offset:4           // permute edge values
ds_bpermute_b32 v104, v134, v104, offset:4         // permute edge values
ds_bpermute_b32 v105, v134, v105, offset:4         // permute edge values
ds_bpermute_b32 v106, v134, v106, offset:4         // permute edge values
ds_bpermute_b32 v107, v134, v107, offset:4         // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=7 v=3 */
label_0082:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v12, v134, v12, offset:4           // permute edge values
ds_bpermute_b32 v13, v134, v13, offset:4           // permute edge values
ds_bpermute_b32 v14, v134, v14, offset:4           // permute edge values
ds_bpermute_b32 v15, v134, v15, offset:4           // permute edge values
ds_bpermute_b32 v44, v134, v44, offset:4           // permute edge values
ds_bpermute_b32 v45, v134, v45, offset:4           // permute edge values
ds_bpermute_b32 v46, v134, v46, offset:4           // permute edge values
ds_bpermute_b32 v47, v134, v47, offset:4           // permute edge values
ds_bpermute_b32 v76, v134, v76, offset:4           // permute edge values
ds_bpermute_b32 v77, v134, v77, offset:4           // permute edge values
ds_bpermute_b32 v78, v134, v78, offset:4           // permute edge values
ds_bpermute_b32 v79, v134, v79, offset:4           // permute edge values
ds_bpermute_b32 v108, v134, v108, offset:4         // permute edge values
ds_bpermute_b32 v109, v134, v109, offset:4         // permute edge values
ds_bpermute_b32 v110, v134, v110, offset:4         // permute edge values
ds_bpermute_b32 v111, v134, v111, offset:4         // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=7 v=4 */
label_0083:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v16, v134, v16, offset:4           // permute edge values
ds_bpermute_b32 v17, v134, v17, offset:4           // permute edge values
ds_bpermute_b32 v18, v134, v18, offset:4           // permute edge values
ds_bpermute_b32 v19, v134, v19, offset:4           // permute edge values
ds_bpermute_b32 v48, v134, v48, offset:4           // permute edge values
ds_bpermute_b32 v49, v134, v49, offset:4           // permute edge values
ds_bpermute_b32 v50, v134, v50, offset:4           // permute edge values
ds_bpermute_b32 v51, v134, v51, offset:4           // permute edge values
ds_bpermute_b32 v80, v134, v80, offset:4           // permute edge values
ds_bpermute_b32 v81, v134, v81, offset:4           // permute edge values
ds_bpermute_b32 v82, v134, v82, offset:4           // permute edge values
ds_bpermute_b32 v83, v134, v83, offset:4           // permute edge values
ds_bpermute_b32 v112, v134, v112, offset:4         // permute edge values
ds_bpermute_b32 v113, v134, v113, offset:4         // permute edge values
ds_bpermute_b32 v114, v134, v114, offset:4         // permute edge values
ds_bpermute_b32 v115, v134, v115, offset:4         // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=7 v=5 */
label_0084:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v20, v134, v20, offset:4           // permute edge values
ds_bpermute_b32 v21, v134, v21, offset:4           // permute edge values
ds_bpermute_b32 v22, v134, v22, offset:4           // permute edge values
ds_bpermute_b32 v23, v134, v23, offset:4           // permute edge values
ds_bpermute_b32 v52, v134, v52, offset:4           // permute edge values
ds_bpermute_b32 v53, v134, v53, offset:4           // permute edge values
ds_bpermute_b32 v54, v134, v54, offset:4           // permute edge values
ds_bpermute_b32 v55, v134, v55, offset:4           // permute edge values
ds_bpermute_b32 v84, v134, v84, offset:4           // permute edge values
ds_bpermute_b32 v85, v134, v85, offset:4           // permute edge values
ds_bpermute_b32 v86, v134, v86, offset:4           // permute edge values
ds_bpermute_b32 v87, v134, v87, offset:4           // permute edge values
ds_bpermute_b32 v116, v134, v116, offset:4         // permute edge values
ds_bpermute_b32 v117, v134, v117, offset:4         // permute edge values
ds_bpermute_b32 v118, v134, v118, offset:4         // permute edge values
ds_bpermute_b32 v119, v134, v119, offset:4         // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=7 v=6 */
label_0085:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v24, v134, v24, offset:4           // permute edge values
ds_bpermute_b32 v25, v134, v25, offset:4           // permute edge values
ds_bpermute_b32 v26, v134, v26, offset:4           // permute edge values
ds_bpermute_b32 v27, v134, v27, offset:4           // permute edge values
ds_bpermute_b32 v56, v134, v56, offset:4           // permute edge values
ds_bpermute_b32 v57, v134, v57, offset:4           // permute edge values
ds_bpermute_b32 v58, v134, v58, offset:4           // permute edge values
ds_bpermute_b32 v59, v134, v59, offset:4           // permute edge values
ds_bpermute_b32 v88, v134, v88, offset:4           // permute edge values
ds_bpermute_b32 v89, v134, v89, offset:4           // permute edge values
ds_bpermute_b32 v90, v134, v90, offset:4           // permute edge values
ds_bpermute_b32 v91, v134, v91, offset:4           // permute edge values
ds_bpermute_b32 v120, v134, v120, offset:4         // permute edge values
ds_bpermute_b32 v121, v134, v121, offset:4         // permute edge values
ds_bpermute_b32 v122, v134, v122, offset:4         // permute edge values
ds_bpermute_b32 v123, v134, v123, offset:4         // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting

/* shift d0 r=7 v=7 */
label_0086:
_v_cmpx_ge_u32 s[54:55], v147, v145                // serial % SG == (wgMT/VECTOR_WIDTH)%SG
v_mov_b32 v134, v[vgprSerial]                      // permute register between threads
v_lshlrev_b32 v134, 2, v134                        // permute register between threads
ds_bpermute_b32 v28, v134, v28, offset:4           // permute edge values
ds_bpermute_b32 v29, v134, v29, offset:4           // permute edge values
ds_bpermute_b32 v30, v134, v30, offset:4           // permute edge values
ds_bpermute_b32 v31, v134, v31, offset:4           // permute edge values
ds_bpermute_b32 v60, v134, v60, offset:4           // permute edge values
ds_bpermute_b32 v61, v134, v61, offset:4           // permute edge values
ds_bpermute_b32 v62, v134, v62, offset:4           // permute edge values
ds_bpermute_b32 v63, v134, v63, offset:4           // permute edge values
ds_bpermute_b32 v92, v134, v92, offset:4           // permute edge values
ds_bpermute_b32 v93, v134, v93, offset:4           // permute edge values
ds_bpermute_b32 v94, v134, v94, offset:4           // permute edge values
ds_bpermute_b32 v95, v134, v95, offset:4           // permute edge values
ds_bpermute_b32 v124, v134, v124, offset:4         // permute edge values
ds_bpermute_b32 v125, v134, v125, offset:4         // permute edge values
ds_bpermute_b32 v126, v134, v126, offset:4         // permute edge values
ds_bpermute_b32 v127, v134, v127, offset:4         // permute edge values
s_waitcnt 0                                        // (Wait all)
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_0087                                // done shifting
label_0087: // end shift0



/* not-LocalSplitU: global write indices */

/* computeStoreVgprs */
v_and_b32 v128, 63, v[vgprSerial]                  // v0 = v[vgprSerial] & 63
v_lshrrev_b32 v129, 6, v[vgprSerial]               // v4 = v[vgprSerial] / 64
v_and_b32 v132, 15, v128                           // v5 = v0 & 15
v_lshrrev_b32 v133, 4, v128                        // v3 = v0 / 16
v_and_b32 v134, 1, v129                            // v4&1
s_mov_b32 s54, 32                                 //  
v_mul_lo_u32 v134, s54, v134                       //  
v_lshlrev_b32 v128, 0, v132                        // ((v[vgprSerial]&63)&15)*2
_v_add_co_u32 v128, vcc, v128, v134                // row
v_lshrrev_b32 v134, 1, v129                        // v4>>1
s_mov_b32 s54, 32                                  //  
v_mul_lo_u32 v134, s54, v134                       //  
v_lshlrev_b32 v129, 0, v133                        // ((v[vgprSerial]&63)&15)*2
_v_add_co_u32 v129, vcc, v129, v134                // col
s_mul_i32 s[sgprStrideoffC+0], 8, s[sgprStridesC]  // 
s_mul_i32 s[sgprStrideoffC+1], 16, s[sgprStridesC] // 
s_mul_i32 s[sgprStrideoffC+2], 24, s[sgprStridesC] // 
s_mul_i32 s[sgprStrideoffD+0], 8, s[sgprStridesD]  // 
s_mul_i32 s[sgprStrideoffD+1], 16, s[sgprStridesD] // 
s_mul_i32 s[sgprStrideoffD+2], 24, s[sgprStridesD] // 
v_mul_lo_u32 v130, v129, s[sgprStrideC1J]          // rowStart vgpr
v_mul_lo_u32 v131, v129, s[sgprStrideD1J]          // rowStart vgpr

s_mul_i32 s54, 0x100, s[sgprWorkGroup0]            // s54 = wg0*MT0
_v_add_co_u32 v128, vcc, s54, v128                 // coord0 = tid0*VW + wg0*MT0
s_mul_i32 s56, 0x100, s[sgprWorkGroup1]            // <- wg1*MT1
_v_add_co_u32 v129, vcc, s56, v129                 // coord1 = tid1*VW + wg1*MT1


/* not-LocalSplitU: global write */









s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 GW_Beta_110                         // Branch if Beta is not zero
s_branch GW_B0_DIRECT_101                          // ABI fixes beta=0: always use rowmapped direct output

s_and_b32 s54, 255, s[sgprSizeI]                   // s54 = s[sgprSizeI] % 256
s_add_u32 s56, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s56                // wg0 >= nwg0-1 ?
s_cselect_b32 s54, s54, 0                          // set rMT0
s_cmpk_gt_u32 s54, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B0_E1_101                        // jump if edges required
s_and_b32 s54, 255, s[sgprSizeJ]                   // s54 = s[sgprSizeJ] % 256
s_add_u32 s56, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s56                // wg1 >= nwg1-1
s_cselect_b32 s54, s54, 0                          // set rMT1
s_cmpk_gt_u32 s54, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B0_E1_101                        // jump if edges required
GW_B0_E0_98:

/* edge=0, allocate 2 sgpr. perBatch=2 perElement=0 elementsPerBatch=128 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSharedMask=1 optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,0,4:vw1); (0,0,0,5:vw1); (0,0,0,6:vw1); (0,0,0,7:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1); (0,1,0,4:vw1); (0,1,0,5:vw1); (0,1,0,6:vw1); (0,1,0,7:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,0,1,4:vw1); (0,0,1,5:vw1); (0,0,1,6:vw1); (0,0,1,7:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,1,1,2:vw1); (0,1,1,3:vw1); (0,1,1,4:vw1); (0,1,1,5:vw1); (0,1,1,6:vw1); (0,1,1,7:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,2,4:vw1); (0,0,2,5:vw1); (0,0,2,6:vw1); (0,0,2,7:vw1); (0,1,2,0:vw1); (0,1,2,1:vw1); (0,1,2,2:vw1); (0,1,2,3:vw1); (0,1,2,4:vw1); (0,1,2,5:vw1); (0,1,2,6:vw1); (0,1,2,7:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,0,3,4:vw1); (0,0,3,5:vw1); (0,0,3,6:vw1); (0,0,3,7:vw1); (0,1,3,0:vw1); (0,1,3,1:vw1); (0,1,3,2:vw1); (0,1,3,3:vw1); (0,1,3,4:vw1); (0,1,3,5:vw1); (0,1,3,6:vw1); (0,1,3,7:vw1); (0,0,4,0:vw1); (0,0,4,1:vw1); (0,0,4,2:vw1); (0,0,4,3:vw1); (0,0,4,4:vw1); (0,0,4,5:vw1); (0,0,4,6:vw1); (0,0,4,7:vw1); (0,1,4,0:vw1); (0,1,4,1:vw1); (0,1,4,2:vw1); (0,1,4,3:vw1); (0,1,4,4:vw1); (0,1,4,5:vw1); (0,1,4,6:vw1); (0,1,4,7:vw1); (0,0,5,0:vw1); (0,0,5,1:vw1); (0,0,5,2:vw1); (0,0,5,3:vw1); (0,0,5,4:vw1); (0,0,5,5:vw1); (0,0,5,6:vw1); (0,0,5,7:vw1); (0,1,5,0:vw1); (0,1,5,1:vw1); (0,1,5,2:vw1); (0,1,5,3:vw1); (0,1,5,4:vw1); (0,1,5,5:vw1); (0,1,5,6:vw1); (0,1,5,7:vw1); (0,0,6,0:vw1); (0,0,6,1:vw1); (0,0,6,2:vw1); (0,0,6,3:vw1); (0,0,6,4:vw1); (0,0,6,5:vw1); (0,0,6,6:vw1); (0,0,6,7:vw1); (0,1,6,0:vw1); (0,1,6,1:vw1); (0,1,6,2:vw1); (0,1,6,3:vw1); (0,1,6,4:vw1); (0,1,6,5:vw1); (0,1,6,6:vw1); (0,1,6,7:vw1); (0,0,7,0:vw1); (0,0,7,1:vw1); (0,0,7,2:vw1); (0,0,7,3:vw1); (0,0,7,4:vw1); (0,0,7,5:vw1); (0,0,7,6:vw1); (0,0,7,7:vw1); (0,1,7,0:vw1); (0,1,7,1:vw1); (0,1,7,2:vw1); (0,1,7,3:vw1); (0,1,7,4:vw1); (0,1,7,5:vw1); (0,1,7,6:vw1); (0,1,7,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
/* (d1,vc1,d0,vc0)=(0,0,0,4) */
/* (d1,vc1,d0,vc0)=(0,0,0,5) */
/* (d1,vc1,d0,vc0)=(0,0,0,6) */
/* (d1,vc1,d0,vc0)=(0,0,0,7) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
/* (d1,vc1,d0,vc0)=(0,0,1,4) */
/* (d1,vc1,d0,vc0)=(0,0,1,5) */
/* (d1,vc1,d0,vc0)=(0,0,1,6) */
/* (d1,vc1,d0,vc0)=(0,0,1,7) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
/* (d1,vc1,d0,vc0)=(0,1,0,4) */
/* (d1,vc1,d0,vc0)=(0,1,0,5) */
/* (d1,vc1,d0,vc0)=(0,1,0,6) */
/* (d1,vc1,d0,vc0)=(0,1,0,7) */
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
/* (d1,vc1,d0,vc0)=(0,1,1,1) */
/* (d1,vc1,d0,vc0)=(0,1,1,2) */
/* (d1,vc1,d0,vc0)=(0,1,1,3) */
/* (d1,vc1,d0,vc0)=(0,1,1,4) */
/* (d1,vc1,d0,vc0)=(0,1,1,5) */
/* (d1,vc1,d0,vc0)=(0,1,1,6) */
/* (d1,vc1,d0,vc0)=(0,1,1,7) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
/* (d1,vc1,d0,vc0)=(0,2,0,4) */
/* (d1,vc1,d0,vc0)=(0,2,0,5) */
/* (d1,vc1,d0,vc0)=(0,2,0,6) */
/* (d1,vc1,d0,vc0)=(0,2,0,7) */
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
/* (d1,vc1,d0,vc0)=(0,2,1,1) */
/* (d1,vc1,d0,vc0)=(0,2,1,2) */
/* (d1,vc1,d0,vc0)=(0,2,1,3) */
/* (d1,vc1,d0,vc0)=(0,2,1,4) */
/* (d1,vc1,d0,vc0)=(0,2,1,5) */
/* (d1,vc1,d0,vc0)=(0,2,1,6) */
/* (d1,vc1,d0,vc0)=(0,2,1,7) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
/* (d1,vc1,d0,vc0)=(0,3,0,4) */
/* (d1,vc1,d0,vc0)=(0,3,0,5) */
/* (d1,vc1,d0,vc0)=(0,3,0,6) */
/* (d1,vc1,d0,vc0)=(0,3,0,7) */
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
/* (d1,vc1,d0,vc0)=(0,3,1,1) */
/* (d1,vc1,d0,vc0)=(0,3,1,2) */
/* (d1,vc1,d0,vc0)=(0,3,1,3) */
/* (d1,vc1,d0,vc0)=(0,3,1,4) */
/* (d1,vc1,d0,vc0)=(0,3,1,5) */
/* (d1,vc1,d0,vc0)=(0,3,1,6) */
/* (d1,vc1,d0,vc0)=(0,3,1,7) */
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
/* (d1,vc1,d0,vc0)=(0,4,0,1) */
/* (d1,vc1,d0,vc0)=(0,4,0,2) */
/* (d1,vc1,d0,vc0)=(0,4,0,3) */
/* (d1,vc1,d0,vc0)=(0,4,0,4) */
/* (d1,vc1,d0,vc0)=(0,4,0,5) */
/* (d1,vc1,d0,vc0)=(0,4,0,6) */
/* (d1,vc1,d0,vc0)=(0,4,0,7) */
/* (d1,vc1,d0,vc0)=(0,4,1,0) */
/* (d1,vc1,d0,vc0)=(0,4,1,1) */
/* (d1,vc1,d0,vc0)=(0,4,1,2) */
/* (d1,vc1,d0,vc0)=(0,4,1,3) */
/* (d1,vc1,d0,vc0)=(0,4,1,4) */
/* (d1,vc1,d0,vc0)=(0,4,1,5) */
/* (d1,vc1,d0,vc0)=(0,4,1,6) */
/* (d1,vc1,d0,vc0)=(0,4,1,7) */
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
/* (d1,vc1,d0,vc0)=(0,5,0,1) */
/* (d1,vc1,d0,vc0)=(0,5,0,2) */
/* (d1,vc1,d0,vc0)=(0,5,0,3) */
/* (d1,vc1,d0,vc0)=(0,5,0,4) */
/* (d1,vc1,d0,vc0)=(0,5,0,5) */
/* (d1,vc1,d0,vc0)=(0,5,0,6) */
/* (d1,vc1,d0,vc0)=(0,5,0,7) */
/* (d1,vc1,d0,vc0)=(0,5,1,0) */
/* (d1,vc1,d0,vc0)=(0,5,1,1) */
/* (d1,vc1,d0,vc0)=(0,5,1,2) */
/* (d1,vc1,d0,vc0)=(0,5,1,3) */
/* (d1,vc1,d0,vc0)=(0,5,1,4) */
/* (d1,vc1,d0,vc0)=(0,5,1,5) */
/* (d1,vc1,d0,vc0)=(0,5,1,6) */
/* (d1,vc1,d0,vc0)=(0,5,1,7) */
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
/* (d1,vc1,d0,vc0)=(0,6,0,1) */
/* (d1,vc1,d0,vc0)=(0,6,0,2) */
/* (d1,vc1,d0,vc0)=(0,6,0,3) */
/* (d1,vc1,d0,vc0)=(0,6,0,4) */
/* (d1,vc1,d0,vc0)=(0,6,0,5) */
/* (d1,vc1,d0,vc0)=(0,6,0,6) */
/* (d1,vc1,d0,vc0)=(0,6,0,7) */
/* (d1,vc1,d0,vc0)=(0,6,1,0) */
/* (d1,vc1,d0,vc0)=(0,6,1,1) */
/* (d1,vc1,d0,vc0)=(0,6,1,2) */
/* (d1,vc1,d0,vc0)=(0,6,1,3) */
/* (d1,vc1,d0,vc0)=(0,6,1,4) */
/* (d1,vc1,d0,vc0)=(0,6,1,5) */
/* (d1,vc1,d0,vc0)=(0,6,1,6) */
/* (d1,vc1,d0,vc0)=(0,6,1,7) */
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
/* (d1,vc1,d0,vc0)=(0,7,0,1) */
/* (d1,vc1,d0,vc0)=(0,7,0,2) */
/* (d1,vc1,d0,vc0)=(0,7,0,3) */
/* (d1,vc1,d0,vc0)=(0,7,0,4) */
/* (d1,vc1,d0,vc0)=(0,7,0,5) */
/* (d1,vc1,d0,vc0)=(0,7,0,6) */
/* (d1,vc1,d0,vc0)=(0,7,0,7) */
/* (d1,vc1,d0,vc0)=(0,7,1,0) */
/* (d1,vc1,d0,vc0)=(0,7,1,1) */
/* (d1,vc1,d0,vc0)=(0,7,1,2) */
/* (d1,vc1,d0,vc0)=(0,7,1,3) */
/* (d1,vc1,d0,vc0)=(0,7,1,4) */
/* (d1,vc1,d0,vc0)=(0,7,1,5) */
/* (d1,vc1,d0,vc0)=(0,7,1,6) */
/* (d1,vc1,d0,vc0)=(0,7,1,7) */
_v_add_lshl_u32 v134, v131, v128, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128

/* rC *= alpha batchEements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 0, 4), (0, 0, 0, 5), (0, 0, 0, 6), (0, 0, 0, 7), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (0, 1, 0, 4), (0, 1, 0, 5), (0, 1, 0, 6), (0, 1, 0, 7), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 0, 1, 4), (0, 0, 1, 5), (0, 0, 1, 6), (0, 0, 1, 7), (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 1, 3), (0, 1, 1, 4), (0, 1, 1, 5), (0, 1, 1, 6), (0, 1, 1, 7), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 2, 4), (0, 0, 2, 5), (0, 0, 2, 6), (0, 0, 2, 7), (0, 1, 2, 0), (0, 1, 2, 1), (0, 1, 2, 2), (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 5), (0, 1, 2, 6), (0, 1, 2, 7), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 0, 3, 4), (0, 0, 3, 5), (0, 0, 3, 6), (0, 0, 3, 7), (0, 1, 3, 0), (0, 1, 3, 1), (0, 1, 3, 2), (0, 1, 3, 3), (0, 1, 3, 4), (0, 1, 3, 5), (0, 1, 3, 6), (0, 1, 3, 7), (0, 0, 4, 0), (0, 0, 4, 1), (0, 0, 4, 2), (0, 0, 4, 3), (0, 0, 4, 4), (0, 0, 4, 5), (0, 0, 4, 6), (0, 0, 4, 7), (0, 1, 4, 0), (0, 1, 4, 1), (0, 1, 4, 2), (0, 1, 4, 3), (0, 1, 4, 4), (0, 1, 4, 5), (0, 1, 4, 6), (0, 1, 4, 7), (0, 0, 5, 0), (0, 0, 5, 1), (0, 0, 5, 2), (0, 0, 5, 3), (0, 0, 5, 4), (0, 0, 5, 5), (0, 0, 5, 6), (0, 0, 5, 7), (0, 1, 5, 0), (0, 1, 5, 1), (0, 1, 5, 2), (0, 1, 5, 3), (0, 1, 5, 4), (0, 1, 5, 5), (0, 1, 5, 6), (0, 1, 5, 7), (0, 0, 6, 0), (0, 0, 6, 1), (0, 0, 6, 2), (0, 0, 6, 3), (0, 0, 6, 4), (0, 0, 6, 5), (0, 0, 6, 6), (0, 0, 6, 7), (0, 1, 6, 0), (0, 1, 6, 1), (0, 1, 6, 2), (0, 1, 6, 3), (0, 1, 6, 4), (0, 1, 6, 5), (0, 1, 6, 6), (0, 1, 6, 7), (0, 0, 7, 0), (0, 0, 7, 1), (0, 0, 7, 2), (0, 0, 7, 3), (0, 0, 7, 4), (0, 0, 7, 5), (0, 0, 7, 6), (0, 0, 7, 7), (0, 1, 7, 0), (0, 1, 7, 1), (0, 1, 7, 2), (0, 1, 7, 3), (0, 1, 7, 4), (0, 1, 7, 5), (0, 1, 7, 6), (0, 1, 7, 7)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // *= alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // *= alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // *= alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // *= alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // *= alpha
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // *= alpha
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // *= alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // *= alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // *= alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // *= alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // *= alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // *= alpha
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // *= alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // *= alpha
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // *= alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // *= alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // *= alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // *= alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // *= alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // *= alpha
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // *= alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // *= alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // *= alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // *= alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // *= alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // *= alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // *= alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // *= alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // *= alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // *= alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // *= alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // *= alpha
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // *= alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // *= alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // *= alpha
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // *= alpha
v_mul_f32 v[vgprValuC+64], s[sgprAlpha], v[vgprValuC+64] // *= alpha
v_mul_f32 v[vgprValuC+65], s[sgprAlpha], v[vgprValuC+65] // *= alpha
v_mul_f32 v[vgprValuC+66], s[sgprAlpha], v[vgprValuC+66] // *= alpha
v_mul_f32 v[vgprValuC+67], s[sgprAlpha], v[vgprValuC+67] // *= alpha
v_mul_f32 v[vgprValuC+68], s[sgprAlpha], v[vgprValuC+68] // *= alpha
v_mul_f32 v[vgprValuC+69], s[sgprAlpha], v[vgprValuC+69] // *= alpha
v_mul_f32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+70] // *= alpha
v_mul_f32 v[vgprValuC+71], s[sgprAlpha], v[vgprValuC+71] // *= alpha
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+72] // *= alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+73] // *= alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+74] // *= alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+75] // *= alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+76] // *= alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+77] // *= alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+78] // *= alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+79] // *= alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+80] // *= alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+81] // *= alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+82] // *= alpha
v_mul_f32 v[vgprValuC+83], s[sgprAlpha], v[vgprValuC+83] // *= alpha
v_mul_f32 v[vgprValuC+84], s[sgprAlpha], v[vgprValuC+84] // *= alpha
v_mul_f32 v[vgprValuC+85], s[sgprAlpha], v[vgprValuC+85] // *= alpha
v_mul_f32 v[vgprValuC+86], s[sgprAlpha], v[vgprValuC+86] // *= alpha
v_mul_f32 v[vgprValuC+87], s[sgprAlpha], v[vgprValuC+87] // *= alpha
v_mul_f32 v[vgprValuC+88], s[sgprAlpha], v[vgprValuC+88] // *= alpha
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+89] // *= alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+90] // *= alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+91] // *= alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+92] // *= alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+93] // *= alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+94] // *= alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+95] // *= alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+96] // *= alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+97] // *= alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+98] // *= alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+99] // *= alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+100] // *= alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+101] // *= alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+102] // *= alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+103] // *= alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+104] // *= alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+105] // *= alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+106] // *= alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+107] // *= alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+108] // *= alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+109] // *= alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+110] // *= alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+111] // *= alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+112] // *= alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+113] // *= alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+114] // *= alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+115] // *= alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+116] // *= alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+117] // *= alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+118] // *= alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+119] // *= alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+120] // *= alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+121] // *= alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+122] // *= alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+123] // *= alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+124] // *= alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+125] // *= alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+126] // *= alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+127] // *= alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

s_mov_b32 s66,96
FP32_TO_BF16 vgprValuC+0     // convert C to bf16
buffer_store_short v0, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+1     // convert C to bf16
buffer_store_short v1, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+2     // convert C to bf16
buffer_store_short v2, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+3     // convert C to bf16
buffer_store_short v3, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+4     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v4, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+5     // convert C to bf16
buffer_store_short v5, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+6     // convert C to bf16
buffer_store_short v6, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+7     // convert C to bf16
buffer_store_short v7, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+8     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v8, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+9     // convert C to bf16
buffer_store_short v9, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+10     // convert C to bf16
buffer_store_short v10, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+11     // convert C to bf16
buffer_store_short v11, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+12     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v12, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+13     // convert C to bf16
buffer_store_short v13, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+14     // convert C to bf16
buffer_store_short v14, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+15     // convert C to bf16
buffer_store_short v15, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+16     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v16, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+17     // convert C to bf16
buffer_store_short v17, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+18     // convert C to bf16
buffer_store_short v18, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+19     // convert C to bf16
buffer_store_short v19, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+20     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v20, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+21     // convert C to bf16
buffer_store_short v21, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+22     // convert C to bf16
buffer_store_short v22, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+23     // convert C to bf16
buffer_store_short v23, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+24     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v24, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+25     // convert C to bf16
buffer_store_short v25, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+26     // convert C to bf16
buffer_store_short v26, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+27     // convert C to bf16
buffer_store_short v27, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+28     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v28, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+29     // convert C to bf16
buffer_store_short v29, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+30     // convert C to bf16
buffer_store_short v30, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+31     // convert C to bf16
buffer_store_short v31, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+32     // convert C to bf16
_v_add_lshl_u32 v134, v131, v128, 0x1              // inner add offset
s_mul_i32 s54, s[sgprStrideD1J], 32                // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_short v32, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+33     // convert C to bf16
buffer_store_short v33, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+34     // convert C to bf16
buffer_store_short v34, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+35     // convert C to bf16
buffer_store_short v35, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+36     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v36, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+37     // convert C to bf16
buffer_store_short v37, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+38     // convert C to bf16
buffer_store_short v38, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+39     // convert C to bf16
buffer_store_short v39, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+40     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v40, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+41     // convert C to bf16
buffer_store_short v41, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+42     // convert C to bf16
buffer_store_short v42, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+43     // convert C to bf16
buffer_store_short v43, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+44     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v44, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+45     // convert C to bf16
buffer_store_short v45, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+46     // convert C to bf16
buffer_store_short v46, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+47     // convert C to bf16
buffer_store_short v47, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+48     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v48, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+49     // convert C to bf16
buffer_store_short v49, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+50     // convert C to bf16
buffer_store_short v50, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+51     // convert C to bf16
buffer_store_short v51, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+52     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v52, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+53     // convert C to bf16
buffer_store_short v53, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+54     // convert C to bf16
buffer_store_short v54, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+55     // convert C to bf16
buffer_store_short v55, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+56     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v56, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+57     // convert C to bf16
buffer_store_short v57, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+58     // convert C to bf16
buffer_store_short v58, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+59     // convert C to bf16
buffer_store_short v59, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+60     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v60, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+61     // convert C to bf16
buffer_store_short v61, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+62     // convert C to bf16
buffer_store_short v62, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+63     // convert C to bf16
buffer_store_short v63, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+64     // convert C to bf16
_v_add_lshl_u32 v134, v131, v128, 0x1              // inner add offset
s_mov_b32 s67,224
s_mul_i32 s54, s[sgprStrideD1J], s67                // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_short v64, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+65     // convert C to bf16
buffer_store_short v65, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+66     // convert C to bf16
buffer_store_short v66, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+67     // convert C to bf16
buffer_store_short v67, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+68     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v68, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+69     // convert C to bf16
buffer_store_short v69, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+70     // convert C to bf16
buffer_store_short v70, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+71     // convert C to bf16
buffer_store_short v71, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+72     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v72, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+73     // convert C to bf16
buffer_store_short v73, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+74     // convert C to bf16
buffer_store_short v74, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+75     // convert C to bf16
buffer_store_short v75, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+76     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v76, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+77     // convert C to bf16
buffer_store_short v77, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+78     // convert C to bf16
buffer_store_short v78, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+79     // convert C to bf16
buffer_store_short v79, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+80     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v80, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+81     // convert C to bf16
buffer_store_short v81, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+82     // convert C to bf16
buffer_store_short v82, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+83     // convert C to bf16
buffer_store_short v83, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+84     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v84, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+85     // convert C to bf16
buffer_store_short v85, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+86     // convert C to bf16
buffer_store_short v86, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+87     // convert C to bf16
buffer_store_short v87, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+88     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v88, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+89     // convert C to bf16
buffer_store_short v89, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+90     // convert C to bf16
buffer_store_short v90, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+91     // convert C to bf16
buffer_store_short v91, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+92     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v92, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+93     // convert C to bf16
buffer_store_short v93, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+94     // convert C to bf16
buffer_store_short v94, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+95     // convert C to bf16
buffer_store_short v95, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+96     // convert C to bf16
_v_add_lshl_u32 v134, v131, v128, 0x1              // inner add offset
s_mul_i32 s54, s[sgprStrideD1J], 32                // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_short v96, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+97     // convert C to bf16
buffer_store_short v97, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+98     // convert C to bf16
buffer_store_short v98, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+99     // convert C to bf16
buffer_store_short v99, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+100     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v100, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+101     // convert C to bf16
buffer_store_short v101, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+102     // convert C to bf16
buffer_store_short v102, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+103     // convert C to bf16
buffer_store_short v103, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+104     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v104, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+105     // convert C to bf16
buffer_store_short v105, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+106     // convert C to bf16
buffer_store_short v106, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+107     // convert C to bf16
buffer_store_short v107, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+108     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v108, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+109     // convert C to bf16
buffer_store_short v109, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+110     // convert C to bf16
buffer_store_short v110, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+111     // convert C to bf16
buffer_store_short v111, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+112     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v112, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+113     // convert C to bf16
buffer_store_short v113, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+114     // convert C to bf16
buffer_store_short v114, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+115     // convert C to bf16
buffer_store_short v115, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+116     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v116, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+117     // convert C to bf16
buffer_store_short v117, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+118     // convert C to bf16
buffer_store_short v118, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+119     // convert C to bf16
buffer_store_short v119, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+120     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v120, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+121     // convert C to bf16
buffer_store_short v121, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+122     // convert C to bf16
buffer_store_short v122, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+123     // convert C to bf16
buffer_store_short v123, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+124     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v124, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+125     // convert C to bf16
buffer_store_short v125, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+126     // convert C to bf16
buffer_store_short v126, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+127     // convert C to bf16
buffer_store_short v127, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
s_branch label_GW_End_109                          // jump to end
GW_B0_DIRECT_101:

/* FP32_TO_BF16 consumes these two generated epilogue constants.  Both the
 * optimized and ordinary beta-zero paths branch here before their native
 * writers initialize them, so establish them at the shared direct entry. */
v_mov_b32 v206, 0x7fff0000                         // fp32 NaN
v_mov_b32 v207, 0x7fff                             // round-to-nearest-even bias

/* The generated C SRD is not part of the grouped ABI.  Rebuild it here as
 * the sorted_token_ids SRD and recover this problem's sorted row start from
 * the uniform VGPR preserved by the v36 scheduler. */
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

/* Group 0: local routed rows 0,4,8,12. */
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

/* Group 1: local routed rows 16,20,24,28. */
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

/* Group 2: local routed rows 128,132,136,140. */
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

/* Group 3: local routed rows 144,148,152,156. */
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

GW_B0_E1_101:

/* edge=1, allocate 46 sgpr. perBatch=6 perElement=2 elementsPerBatch=20 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,0,4:vw1); (0,0,0,5:vw1); (0,0,0,6:vw1); (0,0,0,7:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1); (0,1,0,4:vw1); (0,1,0,5:vw1); (0,1,0,6:vw1); (0,1,0,7:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v135, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v135, vcc, s[sgprStrideoffD+0], v135  // add 4* loop offset
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v136, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+1], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v137, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v137, vcc, s[sgprStrideoffD+2], v137  // add 4* loop offset
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,4) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,5) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v139, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v139, vcc, s[sgprStrideoffD+0], v139  // add 4* loop offset
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,6) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+1], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,7) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v141, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v141, vcc, s[sgprStrideoffD+2], v141  // add 4* loop offset
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v143, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v143, vcc, s[sgprStrideoffD+0], v143  // add 4* loop offset
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+1], v144  // add 4* loop offset
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v145, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v145, vcc, s[sgprStrideoffD+2], v145  // add 4* loop offset
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,4) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,5) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v147, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v147, vcc, s[sgprStrideoffD+0], v147  // add 4* loop offset
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,6) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+1], v148  // add 4* loop offset
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,7) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v149, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v149, vcc, s[sgprStrideoffD+2], v149  // add 4* loop offset
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v151, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v151, vcc, s[sgprStrideoffD+0], v151  // add 4* loop offset
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+1], v152  // add 4* loop offset
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v153, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v153, vcc, s[sgprStrideoffD+2], v153  // add 4* loop offset
v_cndmask_b32 v153, -1, v153, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 0, 4), (0, 0, 0, 5), (0, 0, 0, 6), (0, 0, 0, 7), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (0, 1, 0, 4), (0, 1, 0, 5), (0, 1, 0, 6), (0, 1, 0, 7), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // *= alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // *= alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

FP32_TO_BF16 vgprValuC+0     // convert C to bf16
buffer_store_short v0, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+1     // convert C to bf16
buffer_store_short v1, v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+2     // convert C to bf16
buffer_store_short v2, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+3     // convert C to bf16
buffer_store_short v3, v137, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+4     // convert C to bf16
buffer_store_short v4, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+5     // convert C to bf16
buffer_store_short v5, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+6     // convert C to bf16
buffer_store_short v6, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+7     // convert C to bf16
buffer_store_short v7, v141, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+8     // convert C to bf16
buffer_store_short v8, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+9     // convert C to bf16
buffer_store_short v9, v143, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+10     // convert C to bf16
buffer_store_short v10, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+11     // convert C to bf16
buffer_store_short v11, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+12     // convert C to bf16
buffer_store_short v12, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+13     // convert C to bf16
buffer_store_short v13, v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+14     // convert C to bf16
buffer_store_short v14, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+15     // convert C to bf16
buffer_store_short v15, v149, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+16     // convert C to bf16
buffer_store_short v16, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+17     // convert C to bf16
buffer_store_short v17, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+18     // convert C to bf16
buffer_store_short v18, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+19     // convert C to bf16
buffer_store_short v19, v153, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,1,4:vw1); (0,0,1,5:vw1); (0,0,1,6:vw1); (0,0,1,7:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,1,1,2:vw1); (0,1,1,3:vw1); (0,1,1,4:vw1); (0,1,1,5:vw1); (0,1,1,6:vw1); (0,1,1,7:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,2,4:vw1); (0,0,2,5:vw1); (0,0,2,6:vw1); (0,0,2,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,1,0,4) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,5) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v135, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v135, vcc, s[sgprStrideoffD+0], v135  // add 4* loop offset
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,6) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+1], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,7) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v137, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v137, vcc, s[sgprStrideoffD+2], v137  // add 4* loop offset
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,1) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=1
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v139, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v139, vcc, s[sgprStrideoffD+0], v139  // add 4* loop offset
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,2) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=2
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+1], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,3) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=3
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v141, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v141, vcc, s[sgprStrideoffD+2], v141  // add 4* loop offset
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,4) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,5) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v143, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v143, vcc, s[sgprStrideoffD+0], v143  // add 4* loop offset
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,6) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+1], v144  // add 4* loop offset
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,7) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v145, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v145, vcc, s[sgprStrideoffD+2], v145  // add 4* loop offset
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v146, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
s_mul_i32 s[sgprEdgeCheck+1], s[sgprStrideD1J], 32 // add offset stride
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v147, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v147, vcc, s[sgprStrideoffD+0], v147  // add 4* loop offset
v_add_co_u32 v147, vcc, s[sgprEdgeCheck+1], v147   // 
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v148, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+1], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v149, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v149, vcc, s[sgprStrideoffD+2], v149  // add 4* loop offset
v_add_co_u32 v149, vcc, s[sgprEdgeCheck+1], v149   // 
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,4) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck+1], v150   // 
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,5) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v151, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v151, vcc, s[sgprStrideoffD+0], v151  // add 4* loop offset
v_add_co_u32 v151, vcc, s[sgprEdgeCheck+1], v151   // 
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,6) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+1], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck+1], v152   // 
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,7) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v153, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v153, vcc, s[sgprStrideoffD+2], v153  // add 4* loop offset
v_add_co_u32 v153, vcc, s[sgprEdgeCheck+1], v153   // 
v_cndmask_b32 v153, -1, v153, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 0, 1, 4), (0, 0, 1, 5), (0, 0, 1, 6), (0, 0, 1, 7), (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 1, 3), (0, 1, 1, 4), (0, 1, 1, 5), (0, 1, 1, 6), (0, 1, 1, 7), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 2, 4), (0, 0, 2, 5), (0, 0, 2, 6), (0, 0, 2, 7)] */
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // *= alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // *= alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // *= alpha
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // *= alpha
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // *= alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // *= alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // *= alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // *= alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // *= alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // *= alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

FP32_TO_BF16 vgprValuC+20     // convert C to bf16
buffer_store_short v20, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+21     // convert C to bf16
buffer_store_short v21, v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+22     // convert C to bf16
buffer_store_short v22, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+23     // convert C to bf16
buffer_store_short v23, v137, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+24     // convert C to bf16
buffer_store_short v24, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+25     // convert C to bf16
buffer_store_short v25, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+26     // convert C to bf16
buffer_store_short v26, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+27     // convert C to bf16
buffer_store_short v27, v141, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+28     // convert C to bf16
buffer_store_short v28, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+29     // convert C to bf16
buffer_store_short v29, v143, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+30     // convert C to bf16
buffer_store_short v30, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+31     // convert C to bf16
buffer_store_short v31, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+32     // convert C to bf16
buffer_store_short v32, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+33     // convert C to bf16
buffer_store_short v33, v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+34     // convert C to bf16
buffer_store_short v34, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+35     // convert C to bf16
buffer_store_short v35, v149, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+36     // convert C to bf16
buffer_store_short v36, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+37     // convert C to bf16
buffer_store_short v37, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+38     // convert C to bf16
buffer_store_short v38, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+39     // convert C to bf16
buffer_store_short v39, v153, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,1,2,0:vw1); (0,1,2,1:vw1); (0,1,2,2:vw1); (0,1,2,3:vw1); (0,1,2,4:vw1); (0,1,2,5:vw1); (0,1,2,6:vw1); (0,1,2,7:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,0,3,4:vw1); (0,0,3,5:vw1); (0,0,3,6:vw1); (0,0,3,7:vw1); (0,1,3,0:vw1); (0,1,3,1:vw1); (0,1,3,2:vw1); (0,1,3,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,1) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v135, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v135, vcc, s[sgprStrideoffD+0], v135  // add 4* loop offset
v_add_co_u32 v135, vcc, s[sgprEdgeCheck+1], v135   // 
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,2) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+1], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,3) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v137, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v137, vcc, s[sgprStrideoffD+2], v137  // add 4* loop offset
v_add_co_u32 v137, vcc, s[sgprEdgeCheck+1], v137   // 
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,4) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,5) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v139, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v139, vcc, s[sgprStrideoffD+0], v139  // add 4* loop offset
v_add_co_u32 v139, vcc, s[sgprEdgeCheck+1], v139   // 
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,6) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+1], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,7) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v141, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v141, vcc, s[sgprStrideoffD+2], v141  // add 4* loop offset
v_add_co_u32 v141, vcc, s[sgprEdgeCheck+1], v141   // 
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v143, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v143, vcc, s[sgprStrideoffD+0], v143  // add 4* loop offset
v_add_co_u32 v143, vcc, s[sgprEdgeCheck+1], v143   // 
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+1], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v145, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v145, vcc, s[sgprStrideoffD+2], v145  // add 4* loop offset
v_add_co_u32 v145, vcc, s[sgprEdgeCheck+1], v145   // 
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,4) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,5) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v147, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v147, vcc, s[sgprStrideoffD+0], v147  // add 4* loop offset
v_add_co_u32 v147, vcc, s[sgprEdgeCheck+1], v147   // 
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,6) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+1], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,7) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v149, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v149, vcc, s[sgprStrideoffD+2], v149  // add 4* loop offset
v_add_co_u32 v149, vcc, s[sgprEdgeCheck+1], v149   // 
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck+1], v150   // 
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,1) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=1
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v151, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v151, vcc, s[sgprStrideoffD+0], v151  // add 4* loop offset
v_add_co_u32 v151, vcc, s[sgprEdgeCheck+1], v151   // 
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,2) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=2
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+1], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck+1], v152   // 
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,3) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=3
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v153, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v153, vcc, s[sgprStrideoffD+2], v153  // add 4* loop offset
v_add_co_u32 v153, vcc, s[sgprEdgeCheck+1], v153   // 
v_cndmask_b32 v153, -1, v153, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 1, 2, 0), (0, 1, 2, 1), (0, 1, 2, 2), (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 5), (0, 1, 2, 6), (0, 1, 2, 7), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 0, 3, 4), (0, 0, 3, 5), (0, 0, 3, 6), (0, 0, 3, 7), (0, 1, 3, 0), (0, 1, 3, 1), (0, 1, 3, 2), (0, 1, 3, 3)] */
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // *= alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // *= alpha
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // *= alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // *= alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // *= alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // *= alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // *= alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // *= alpha
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // *= alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // *= alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // *= alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // *= alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // *= alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // *= alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // *= alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // *= alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // *= alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // *= alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // *= alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // *= alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

FP32_TO_BF16 vgprValuC+40     // convert C to bf16
buffer_store_short v40, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+41     // convert C to bf16
buffer_store_short v41, v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+42     // convert C to bf16
buffer_store_short v42, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+43     // convert C to bf16
buffer_store_short v43, v137, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+44     // convert C to bf16
buffer_store_short v44, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+45     // convert C to bf16
buffer_store_short v45, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+46     // convert C to bf16
buffer_store_short v46, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+47     // convert C to bf16
buffer_store_short v47, v141, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+48     // convert C to bf16
buffer_store_short v48, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+49     // convert C to bf16
buffer_store_short v49, v143, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+50     // convert C to bf16
buffer_store_short v50, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+51     // convert C to bf16
buffer_store_short v51, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+52     // convert C to bf16
buffer_store_short v52, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+53     // convert C to bf16
buffer_store_short v53, v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+54     // convert C to bf16
buffer_store_short v54, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+55     // convert C to bf16
buffer_store_short v55, v149, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+56     // convert C to bf16
buffer_store_short v56, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+57     // convert C to bf16
buffer_store_short v57, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+58     // convert C to bf16
buffer_store_short v58, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+59     // convert C to bf16
buffer_store_short v59, v153, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (0,1,3,4:vw1); (0,1,3,5:vw1); (0,1,3,6:vw1); (0,1,3,7:vw1); (0,0,4,0:vw1); (0,0,4,1:vw1); (0,0,4,2:vw1); (0,0,4,3:vw1); (0,0,4,4:vw1); (0,0,4,5:vw1); (0,0,4,6:vw1); (0,0,4,7:vw1); (0,1,4,0:vw1); (0,1,4,1:vw1); (0,1,4,2:vw1); (0,1,4,3:vw1); (0,1,4,4:vw1); (0,1,4,5:vw1); (0,1,4,6:vw1); (0,1,4,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,3,1,4) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,5) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v135, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v135, vcc, s[sgprStrideoffD+0], v135  // add 4* loop offset
v_add_co_u32 v135, vcc, s[sgprEdgeCheck+1], v135   // 
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,6) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+1], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,7) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v137, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v137, vcc, s[sgprStrideoffD+2], v137  // add 4* loop offset
v_add_co_u32 v137, vcc, s[sgprEdgeCheck+1], v137   // 
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
s_mov_b32 s54, 128
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v138, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
s_mul_i32 s[sgprEdgeCheck+1], s[sgprStrideD1J], 256 // add offset stride
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,1) */
s_mov_b32 s54, 132
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v139, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v139, vcc, s[sgprStrideoffD+0], v139  // add 4* loop offset
v_add_co_u32 v139, vcc, s[sgprEdgeCheck+1], v139   // 
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,2) */
s_mov_b32 s54, 136
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v140, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+1], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,3) */
s_mov_b32 s54, 140
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v141, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v141, vcc, s[sgprStrideoffD+2], v141  // add 4* loop offset
v_add_co_u32 v141, vcc, s[sgprEdgeCheck+1], v141   // 
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,4) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,5) */
s_mov_b32 s54, 132
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v143, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v143, vcc, s[sgprStrideoffD+0], v143  // add 4* loop offset
v_add_co_u32 v143, vcc, s[sgprEdgeCheck+1], v143   // 
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,6) */
s_mov_b32 s54, 136
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+1], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,7) */
s_mov_b32 s54, 140
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v145, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v145, vcc, s[sgprStrideoffD+2], v145  // add 4* loop offset
v_add_co_u32 v145, vcc, s[sgprEdgeCheck+1], v145   // 
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,0) */
s_mov_b32 s54, 128
s_mov_b32 s55, 64
_v_add_co_u32 v132, vcc, v128, s55                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,1) */
s_mov_b32 s54, 132
s_mov_b32 s55, 64
_v_add_co_u32 v132, vcc, v128, s55                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v147, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v147, vcc, s[sgprStrideoffD+0], v147  // add 4* loop offset
v_add_co_u32 v147, vcc, s[sgprEdgeCheck+1], v147   // 
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,2) */
s_mov_b32 s54, 136
s_mov_b32 s55, 64
_v_add_co_u32 v132, vcc, v128, s55                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+1], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,3) */
s_mov_b32 s54, 140
s_mov_b32 s55, 64
_v_add_co_u32 v132, vcc, v128, s55                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v149, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v149, vcc, s[sgprStrideoffD+2], v149  // add 4* loop offset
v_add_co_u32 v149, vcc, s[sgprEdgeCheck+1], v149   // 
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,4) */
s_mov_b32 s54, 128
s_mov_b32 s55, 80
_v_add_co_u32 v132, vcc, v128, s55                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck+1], v150   // 
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,5) */
s_mov_b32 s54, 132
s_mov_b32 s55, 80
_v_add_co_u32 v132, vcc, v128, s55                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v151, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v151, vcc, s[sgprStrideoffD+0], v151  // add 4* loop offset
v_add_co_u32 v151, vcc, s[sgprEdgeCheck+1], v151   // 
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,6) */
s_mov_b32 s54, 136
s_mov_b32 s55, 80
_v_add_co_u32 v132, vcc, v128, s55                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+1], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck+1], v152   // 
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,7) */
s_mov_b32 s54, 140
s_mov_b32 s55, 80
_v_add_co_u32 v132, vcc, v128, s55                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v153, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v153, vcc, s[sgprStrideoffD+2], v153  // add 4* loop offset
v_add_co_u32 v153, vcc, s[sgprEdgeCheck+1], v153   // 
v_cndmask_b32 v153, -1, v153, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 1, 3, 4), (0, 1, 3, 5), (0, 1, 3, 6), (0, 1, 3, 7), (0, 0, 4, 0), (0, 0, 4, 1), (0, 0, 4, 2), (0, 0, 4, 3), (0, 0, 4, 4), (0, 0, 4, 5), (0, 0, 4, 6), (0, 0, 4, 7), (0, 1, 4, 0), (0, 1, 4, 1), (0, 1, 4, 2), (0, 1, 4, 3), (0, 1, 4, 4), (0, 1, 4, 5), (0, 1, 4, 6), (0, 1, 4, 7)] */
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // *= alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // *= alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // *= alpha
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // *= alpha
v_mul_f32 v[vgprValuC+64], s[sgprAlpha], v[vgprValuC+64] // *= alpha
v_mul_f32 v[vgprValuC+65], s[sgprAlpha], v[vgprValuC+65] // *= alpha
v_mul_f32 v[vgprValuC+66], s[sgprAlpha], v[vgprValuC+66] // *= alpha
v_mul_f32 v[vgprValuC+67], s[sgprAlpha], v[vgprValuC+67] // *= alpha
v_mul_f32 v[vgprValuC+68], s[sgprAlpha], v[vgprValuC+68] // *= alpha
v_mul_f32 v[vgprValuC+69], s[sgprAlpha], v[vgprValuC+69] // *= alpha
v_mul_f32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+70] // *= alpha
v_mul_f32 v[vgprValuC+71], s[sgprAlpha], v[vgprValuC+71] // *= alpha
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+72] // *= alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+73] // *= alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+74] // *= alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+75] // *= alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+76] // *= alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+77] // *= alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+78] // *= alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+79] // *= alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

FP32_TO_BF16 vgprValuC+60     // convert C to bf16
buffer_store_short v60, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+61     // convert C to bf16
buffer_store_short v61, v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+62     // convert C to bf16
buffer_store_short v62, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+63     // convert C to bf16
buffer_store_short v63, v137, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+64     // convert C to bf16
buffer_store_short v64, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+65     // convert C to bf16
buffer_store_short v65, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+66     // convert C to bf16
buffer_store_short v66, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+67     // convert C to bf16
buffer_store_short v67, v141, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+68     // convert C to bf16
buffer_store_short v68, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+69     // convert C to bf16
buffer_store_short v69, v143, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+70     // convert C to bf16
buffer_store_short v70, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+71     // convert C to bf16
buffer_store_short v71, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+72     // convert C to bf16
buffer_store_short v72, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+73     // convert C to bf16
buffer_store_short v73, v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+74     // convert C to bf16
buffer_store_short v74, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+75     // convert C to bf16
buffer_store_short v75, v149, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+76     // convert C to bf16
buffer_store_short v76, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+77     // convert C to bf16
buffer_store_short v77, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+78     // convert C to bf16
buffer_store_short v78, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+79     // convert C to bf16
buffer_store_short v79, v153, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #4 (d1,d0,vc1,vc0) = */
/*    (0,0,5,0:vw1); (0,0,5,1:vw1); (0,0,5,2:vw1); (0,0,5,3:vw1); (0,0,5,4:vw1); (0,0,5,5:vw1); (0,0,5,6:vw1); (0,0,5,7:vw1); (0,1,5,0:vw1); (0,1,5,1:vw1); (0,1,5,2:vw1); (0,1,5,3:vw1); (0,1,5,4:vw1); (0,1,5,5:vw1); (0,1,5,6:vw1); (0,1,5,7:vw1); (0,0,6,0:vw1); (0,0,6,1:vw1); (0,0,6,2:vw1); (0,0,6,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
s_mov_b32 s54, 128
s_mov_b32 s55, 128
_v_add_co_u32 v132, vcc, v128, s55                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,1) */
s_mov_b32 s54, 132
s_mov_b32 s55, 128
_v_add_co_u32 v132, vcc, v128, s55                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v135, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v135, vcc, s[sgprStrideoffD+0], v135  // add 4* loop offset
v_add_co_u32 v135, vcc, s[sgprEdgeCheck+1], v135   // 
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,2) */
s_mov_b32 s54, 136
s_mov_b32 s55, 128
_v_add_co_u32 v132, vcc, v128, s55                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+1], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,3) */
s_mov_b32 s54, 140
s_mov_b32 s55, 128
_v_add_co_u32 v132, vcc, v128, s55                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s54                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v137, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v137, vcc, s[sgprStrideoffD+2], v137  // add 4* loop offset
v_add_co_u32 v137, vcc, s[sgprEdgeCheck+1], v137   // 
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,4) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 128
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,5) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 132
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v139, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v139, vcc, s[sgprStrideoffD+0], v139  // add 4* loop offset
v_add_co_u32 v139, vcc, s[sgprEdgeCheck+1], v139   // 
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,6) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 136
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+1], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,7) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 140
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v141, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v141, vcc, s[sgprStrideoffD+2], v141  // add 4* loop offset
v_add_co_u32 v141, vcc, s[sgprEdgeCheck+1], v141   // 
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,0) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 128
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,1) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 132
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v143, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v143, vcc, s[sgprStrideoffD+0], v143  // add 4* loop offset
v_add_co_u32 v143, vcc, s[sgprEdgeCheck+1], v143   // 
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,2) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 136
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+1], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,3) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 140
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v145, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v145, vcc, s[sgprStrideoffD+2], v145  // add 4* loop offset
v_add_co_u32 v145, vcc, s[sgprEdgeCheck+1], v145   // 
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,4) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 128
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,5) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 132
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v147, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v147, vcc, s[sgprStrideoffD+0], v147  // add 4* loop offset
v_add_co_u32 v147, vcc, s[sgprEdgeCheck+1], v147   // 
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,6) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 136
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+1], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,7) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 140
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v149, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v149, vcc, s[sgprStrideoffD+2], v149  // add 4* loop offset
v_add_co_u32 v149, vcc, s[sgprEdgeCheck+1], v149   // 
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
s_mov_b32 s55, 144
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v150, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
s_mov_b32 s54, 288
s_mul_i32 s[sgprEdgeCheck+1], s[sgprStrideD1J], s54 // add offset stride
v_add_co_u32 v150, vcc, s[sgprEdgeCheck+1], v150   // 
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,1) */
s_mov_b32 s55, 148
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v151, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v151, vcc, s[sgprStrideoffD+0], v151  // add 4* loop offset
v_add_co_u32 v151, vcc, s[sgprEdgeCheck+1], v151   // 
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,2) */
s_mov_b32 s55, 152
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v152, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+1], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck+1], v152   // 
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,3) */
s_mov_b32 s55, 156
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v153, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v153, vcc, s[sgprStrideoffD+2], v153  // add 4* loop offset
v_add_co_u32 v153, vcc, s[sgprEdgeCheck+1], v153   // 
v_cndmask_b32 v153, -1, v153, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 0, 5, 0), (0, 0, 5, 1), (0, 0, 5, 2), (0, 0, 5, 3), (0, 0, 5, 4), (0, 0, 5, 5), (0, 0, 5, 6), (0, 0, 5, 7), (0, 1, 5, 0), (0, 1, 5, 1), (0, 1, 5, 2), (0, 1, 5, 3), (0, 1, 5, 4), (0, 1, 5, 5), (0, 1, 5, 6), (0, 1, 5, 7), (0, 0, 6, 0), (0, 0, 6, 1), (0, 0, 6, 2), (0, 0, 6, 3)] */
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+80] // *= alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+81] // *= alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+82] // *= alpha
v_mul_f32 v[vgprValuC+83], s[sgprAlpha], v[vgprValuC+83] // *= alpha
v_mul_f32 v[vgprValuC+84], s[sgprAlpha], v[vgprValuC+84] // *= alpha
v_mul_f32 v[vgprValuC+85], s[sgprAlpha], v[vgprValuC+85] // *= alpha
v_mul_f32 v[vgprValuC+86], s[sgprAlpha], v[vgprValuC+86] // *= alpha
v_mul_f32 v[vgprValuC+87], s[sgprAlpha], v[vgprValuC+87] // *= alpha
v_mul_f32 v[vgprValuC+88], s[sgprAlpha], v[vgprValuC+88] // *= alpha
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+89] // *= alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+90] // *= alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+91] // *= alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+92] // *= alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+93] // *= alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+94] // *= alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+95] // *= alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+96] // *= alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+97] // *= alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+98] // *= alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+99] // *= alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

FP32_TO_BF16 vgprValuC+80     // convert C to bf16
buffer_store_short v80, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+81     // convert C to bf16
buffer_store_short v81, v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+82     // convert C to bf16
buffer_store_short v82, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+83     // convert C to bf16
buffer_store_short v83, v137, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+84     // convert C to bf16
buffer_store_short v84, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+85     // convert C to bf16
buffer_store_short v85, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+86     // convert C to bf16
buffer_store_short v86, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+87     // convert C to bf16
buffer_store_short v87, v141, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+88     // convert C to bf16
buffer_store_short v88, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+89     // convert C to bf16
buffer_store_short v89, v143, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+90     // convert C to bf16
buffer_store_short v90, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+91     // convert C to bf16
buffer_store_short v91, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+92     // convert C to bf16
buffer_store_short v92, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+93     // convert C to bf16
buffer_store_short v93, v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+94     // convert C to bf16
buffer_store_short v94, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+95     // convert C to bf16
buffer_store_short v95, v149, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+96     // convert C to bf16
buffer_store_short v96, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+97     // convert C to bf16
buffer_store_short v97, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+98     // convert C to bf16
buffer_store_short v98, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+99     // convert C to bf16
buffer_store_short v99, v153, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #5 (d1,d0,vc1,vc0) = */
/*    (0,0,6,4:vw1); (0,0,6,5:vw1); (0,0,6,6:vw1); (0,0,6,7:vw1); (0,1,6,0:vw1); (0,1,6,1:vw1); (0,1,6,2:vw1); (0,1,6,3:vw1); (0,1,6,4:vw1); (0,1,6,5:vw1); (0,1,6,6:vw1); (0,1,6,7:vw1); (0,0,7,0:vw1); (0,0,7,1:vw1); (0,0,7,2:vw1); (0,0,7,3:vw1); (0,0,7,4:vw1); (0,0,7,5:vw1); (0,0,7,6:vw1); (0,0,7,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,6,0,4) */
s_mov_b32 s55, 144
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,5) */
s_mov_b32 s55, 148
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v135, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v135, vcc, s[sgprStrideoffD+0], v135  // add 4* loop offset
v_add_co_u32 v135, vcc, s[sgprEdgeCheck+1], v135   // 
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,6) */
s_mov_b32 s55, 152
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+1], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,7) */
s_mov_b32 s55, 156
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v137, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v137, vcc, s[sgprStrideoffD+2], v137  // add 4* loop offset
v_add_co_u32 v137, vcc, s[sgprEdgeCheck+1], v137   // 
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,0) */
s_mov_b32 s54, 64
s_mov_b32 s55, 144
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,1) */
s_mov_b32 s54, 64
s_mov_b32 s55, 148
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v139, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v139, vcc, s[sgprStrideoffD+0], v139  // add 4* loop offset
v_add_co_u32 v139, vcc, s[sgprEdgeCheck+1], v139   // 
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,2) */
s_mov_b32 s54, 64
s_mov_b32 s55, 152
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+1], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,3) */
s_mov_b32 s54, 64
s_mov_b32 s55, 156
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v141, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v141, vcc, s[sgprStrideoffD+2], v141  // add 4* loop offset
v_add_co_u32 v141, vcc, s[sgprEdgeCheck+1], v141   // 
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,4) */
s_mov_b32 s54, 80
s_mov_b32 s55, 144
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,5) */
s_mov_b32 s54, 80
s_mov_b32 s55, 148
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v143, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v143, vcc, s[sgprStrideoffD+0], v143  // add 4* loop offset
v_add_co_u32 v143, vcc, s[sgprEdgeCheck+1], v143   // 
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,6) */
s_mov_b32 s54, 80
s_mov_b32 s55, 152
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+1], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,7) */
s_mov_b32 s54, 80
s_mov_b32 s55, 156
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v145, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v145, vcc, s[sgprStrideoffD+2], v145  // add 4* loop offset
v_add_co_u32 v145, vcc, s[sgprEdgeCheck+1], v145   // 
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
s_mov_b32 s54, 128
s_mov_b32 s55, 144
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,1) */
s_mov_b32 s54, 128
s_mov_b32 s55, 148
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v147, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v147, vcc, s[sgprStrideoffD+0], v147  // add 4* loop offset
v_add_co_u32 v147, vcc, s[sgprEdgeCheck+1], v147   // 
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,2) */
s_mov_b32 s54, 128
s_mov_b32 s55, 152
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+1], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,3) */
s_mov_b32 s54, 128
s_mov_b32 s55, 156
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v149, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v149, vcc, s[sgprStrideoffD+2], v149  // add 4* loop offset
v_add_co_u32 v149, vcc, s[sgprEdgeCheck+1], v149   // 
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,4) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55,144
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck+1], v150   // 
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,5) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55,148
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v151, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v151, vcc, s[sgprStrideoffD+0], v151  // add 4* loop offset
v_add_co_u32 v151, vcc, s[sgprEdgeCheck+1], v151   // 
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,6) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55,152
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+1], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck+1], v152   // 
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,7) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55,156
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v153, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v153, vcc, s[sgprStrideoffD+2], v153  // add 4* loop offset
v_add_co_u32 v153, vcc, s[sgprEdgeCheck+1], v153   // 
v_cndmask_b32 v153, -1, v153, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 0, 6, 4), (0, 0, 6, 5), (0, 0, 6, 6), (0, 0, 6, 7), (0, 1, 6, 0), (0, 1, 6, 1), (0, 1, 6, 2), (0, 1, 6, 3), (0, 1, 6, 4), (0, 1, 6, 5), (0, 1, 6, 6), (0, 1, 6, 7), (0, 0, 7, 0), (0, 0, 7, 1), (0, 0, 7, 2), (0, 0, 7, 3), (0, 0, 7, 4), (0, 0, 7, 5), (0, 0, 7, 6), (0, 0, 7, 7)] */
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+100] // *= alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+101] // *= alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+102] // *= alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+103] // *= alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+104] // *= alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+105] // *= alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+106] // *= alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+107] // *= alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+108] // *= alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+109] // *= alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+110] // *= alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+111] // *= alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+112] // *= alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+113] // *= alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+114] // *= alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+115] // *= alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+116] // *= alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+117] // *= alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+118] // *= alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+119] // *= alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

FP32_TO_BF16 vgprValuC+100     // convert C to bf16
buffer_store_short v100, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+101     // convert C to bf16
buffer_store_short v101, v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+102     // convert C to bf16
buffer_store_short v102, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+103     // convert C to bf16
buffer_store_short v103, v137, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+104     // convert C to bf16
buffer_store_short v104, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+105     // convert C to bf16
buffer_store_short v105, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+106     // convert C to bf16
buffer_store_short v106, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+107     // convert C to bf16
buffer_store_short v107, v141, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+108     // convert C to bf16
buffer_store_short v108, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+109     // convert C to bf16
buffer_store_short v109, v143, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+110     // convert C to bf16
buffer_store_short v110, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+111     // convert C to bf16
buffer_store_short v111, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+112     // convert C to bf16
buffer_store_short v112, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+113     // convert C to bf16
buffer_store_short v113, v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+114     // convert C to bf16
buffer_store_short v114, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+115     // convert C to bf16
buffer_store_short v115, v149, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+116     // convert C to bf16
buffer_store_short v116, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+117     // convert C to bf16
buffer_store_short v117, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+118     // convert C to bf16
buffer_store_short v118, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+119     // convert C to bf16
buffer_store_short v119, v153, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #6 (d1,d0,vc1,vc0) = */
/*    (0,1,7,0:vw1); (0,1,7,1:vw1); (0,1,7,2:vw1); (0,1,7,3:vw1); (0,1,7,4:vw1); (0,1,7,5:vw1); (0,1,7,6:vw1); (0,1,7,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,7,1,0) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=0
s_mov_b32 s55, 144
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,1) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=0
s_mov_b32 s55, 148
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v135, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v135, vcc, s[sgprStrideoffD+0], v135  // add 4* loop offset
v_add_co_u32 v135, vcc, s[sgprEdgeCheck+1], v135   // 
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,2) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=0
s_mov_b32 s55, 152
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+1], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,3) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=0
s_mov_b32 s55, 156
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v137, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v137, vcc, s[sgprStrideoffD+2], v137  // add 4* loop offset
v_add_co_u32 v137, vcc, s[sgprEdgeCheck+1], v137   // 
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,4) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
s_mov_b32 s55, 144
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,5) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
s_mov_b32 s55, 148
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v139, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v139, vcc, s[sgprStrideoffD+0], v139  // add 4* loop offset
v_add_co_u32 v139, vcc, s[sgprEdgeCheck+1], v139   // 
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,6) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
s_mov_b32 s55, 152
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+1], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,7) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
s_mov_b32 s55, 156
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v141, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v141, vcc, s[sgprStrideoffD+2], v141  // add 4* loop offset
v_add_co_u32 v141, vcc, s[sgprEdgeCheck+1], v141   // 
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 1, 7, 0), (0, 1, 7, 1), (0, 1, 7, 2), (0, 1, 7, 3), (0, 1, 7, 4), (0, 1, 7, 5), (0, 1, 7, 6), (0, 1, 7, 7)] */
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+120] // *= alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+121] // *= alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+122] // *= alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+123] // *= alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+124] // *= alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+125] // *= alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+126] // *= alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+127] // *= alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

FP32_TO_BF16 vgprValuC+120     // convert C to bf16
buffer_store_short v120, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+121     // convert C to bf16
buffer_store_short v121, v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+122     // convert C to bf16
buffer_store_short v122, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+123     // convert C to bf16
buffer_store_short v123, v137, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+124     // convert C to bf16
buffer_store_short v124, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+125     // convert C to bf16
buffer_store_short v125, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+126     // convert C to bf16
buffer_store_short v126, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+127     // convert C to bf16
buffer_store_short v127, v141, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_branch label_GW_End_109                          // jump to end
GW_Beta_110:
s_and_b32 s54, 255, s[sgprSizeI]                   // s54 = s[sgprSizeI] % 256
s_add_u32 s56, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s56                // wg0 >= nwg0-1 ?
s_cselect_b32 s54, s54, 0                          // set rMT0
s_cmpk_gt_u32 s54, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B1_E1_108                        // jump if edges required
s_and_b32 s54, 255, s[sgprSizeJ]                   // s54 = s[sgprSizeJ] % 256
s_add_u32 s56, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s56                // wg1 >= nwg1-1
s_cselect_b32 s54, s54, 0                          // set rMT1
s_cmpk_gt_u32 s54, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B1_E1_108                        // jump if edges required
GW_B1_E0_105:

/* edge=0, allocate 2 sgpr. perBatch=2 perElement=0 elementsPerBatch=58 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSharedMask=1 optSrdIncForRow=1 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,0,4:vw1); (0,0,0,5:vw1); (0,0,0,6:vw1); (0,0,0,7:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1); (0,1,0,4:vw1); (0,1,0,5:vw1); (0,1,0,6:vw1); (0,1,0,7:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,0,1,4:vw1); (0,0,1,5:vw1); (0,0,1,6:vw1); (0,0,1,7:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,1,1,2:vw1); (0,1,1,3:vw1); (0,1,1,4:vw1); (0,1,1,5:vw1); (0,1,1,6:vw1); (0,1,1,7:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,2,4:vw1); (0,0,2,5:vw1); (0,0,2,6:vw1); (0,0,2,7:vw1); (0,1,2,0:vw1); (0,1,2,1:vw1); (0,1,2,2:vw1); (0,1,2,3:vw1); (0,1,2,4:vw1); (0,1,2,5:vw1); (0,1,2,6:vw1); (0,1,2,7:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,0,3,4:vw1); (0,0,3,5:vw1); (0,0,3,6:vw1); (0,0,3,7:vw1); (0,1,3,0:vw1); (0,1,3,1:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
s_mov_b32 s66,0x60
_v_add_lshl_u32 v134, v130, v128, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
buffer_load_short_d16 v136, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
buffer_load_short_d16 v137, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
buffer_load_short_d16 v138, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,0,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v139, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,0,5) */
buffer_load_short_d16 v140, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,0,6) */
buffer_load_short_d16 v141, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,0,7) */
buffer_load_short_d16 v142, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v134, vcc, v134, s66                  // load not edge offset
buffer_load_short_d16 v143, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
buffer_load_short_d16 v144, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
buffer_load_short_d16 v145, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
buffer_load_short_d16 v146, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v147, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,5) */
buffer_load_short_d16 v148, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,6) */
buffer_load_short_d16 v149, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,7) */
buffer_load_short_d16 v150, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
_v_add_co_u32 v134, vcc, v134, s66                  // load not edge offset
buffer_load_short_d16 v151, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
buffer_load_short_d16 v152, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
buffer_load_short_d16 v153, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
buffer_load_short_d16 v154, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,0,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v155, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,0,5) */
buffer_load_short_d16 v156, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,0,6) */
buffer_load_short_d16 v157, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,0,7) */
buffer_load_short_d16 v158, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
_v_add_co_u32 v134, vcc, v134, s66                  // load not edge offset
buffer_load_short_d16 v159, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,1,1) */
buffer_load_short_d16 v160, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,1,2) */
buffer_load_short_d16 v161, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,1,3) */
buffer_load_short_d16 v162, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,1,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v163, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,1,5) */
buffer_load_short_d16 v164, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,1,6) */
buffer_load_short_d16 v165, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,1,1,7) */
buffer_load_short_d16 v166, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
_v_add_lshl_u32 v134, v130, v128, 0x1              // 
s_mul_i32 s54, s[sgprStrideC1J], 32                // scale StrideC *= numRows(16) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v167, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
buffer_load_short_d16 v168, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
buffer_load_short_d16 v169, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
buffer_load_short_d16 v170, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,0,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v171, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,0,5) */
buffer_load_short_d16 v172, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,0,6) */
buffer_load_short_d16 v173, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,0,7) */
buffer_load_short_d16 v174, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
_v_add_co_u32 v134, vcc, v134, s66                  // load not edge offset
buffer_load_short_d16 v175, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,1,1) */
buffer_load_short_d16 v176, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,1,2) */
buffer_load_short_d16 v177, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,1,3) */
buffer_load_short_d16 v178, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,1,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v179, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,1,5) */
buffer_load_short_d16 v180, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,1,6) */
buffer_load_short_d16 v181, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,2,1,7) */
buffer_load_short_d16 v182, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
_v_add_co_u32 v134, vcc, v134, s66                  // load not edge offset
buffer_load_short_d16 v183, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
buffer_load_short_d16 v184, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
buffer_load_short_d16 v185, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
buffer_load_short_d16 v186, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,0,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v187, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,0,5) */
buffer_load_short_d16 v191, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,0,6) */
buffer_load_short_d16 v192, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,0,7) */
buffer_load_short_d16 v193, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
_v_add_co_u32 v134, vcc, v134, s66                  // load not edge offset
buffer_load_short_d16 v194, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,1) */
buffer_load_short_d16 v195, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v128, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128

/* rC *= alpha batchEements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 0, 4), (0, 0, 0, 5), (0, 0, 0, 6), (0, 0, 0, 7), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (0, 1, 0, 4), (0, 1, 0, 5), (0, 1, 0, 6), (0, 1, 0, 7), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 0, 1, 4), (0, 0, 1, 5), (0, 0, 1, 6), (0, 0, 1, 7), (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 1, 3), (0, 1, 1, 4), (0, 1, 1, 5), (0, 1, 1, 6), (0, 1, 1, 7), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 2, 4), (0, 0, 2, 5), (0, 0, 2, 6), (0, 0, 2, 7), (0, 1, 2, 0), (0, 1, 2, 1), (0, 1, 2, 2), (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 5), (0, 1, 2, 6), (0, 1, 2, 7), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 0, 3, 4), (0, 0, 3, 5), (0, 0, 3, 6), (0, 0, 3, 7), (0, 1, 3, 0), (0, 1, 3, 1)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // *= alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // *= alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // *= alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // *= alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // *= alpha
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // *= alpha
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // *= alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // *= alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // *= alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // *= alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // *= alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // *= alpha
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // *= alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // *= alpha
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // *= alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // *= alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // *= alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // *= alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // *= alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // *= alpha
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // *= alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // *= alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // *= alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // *= alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // *= alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // *= alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // *= alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // *= alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // *= alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // *= alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16


s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 0 + 0 - 1
v_lshlrev_b32 v209, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+0], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+0     // convert C to bf16
buffer_store_short v0, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 1 + 1 - 1
v_lshlrev_b32 v209, 16, v136                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+1], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+1     // convert C to bf16
buffer_store_short v1, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 2 + 2 - 1
v_lshlrev_b32 v209, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+2], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+2     // convert C to bf16
buffer_store_short v2, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 3 + 3 - 1
v_lshlrev_b32 v209, 16, v138                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+3], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+3     // convert C to bf16
buffer_store_short v3, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 4 + 4 - 1
v_lshlrev_b32 v209, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+4], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+4     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v4, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 5 + 5 - 1
v_lshlrev_b32 v209, 16, v140                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+5], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+5     // convert C to bf16
buffer_store_short v5, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 6 + 6 - 1
v_lshlrev_b32 v209, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+6], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+6     // convert C to bf16
buffer_store_short v6, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 7 + 7 - 1
v_lshlrev_b32 v209, 16, v142                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+7], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+7     // convert C to bf16
buffer_store_short v7, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 8 + 8 - 1
v_lshlrev_b32 v209, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+8], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+8     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v8, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 9 + 9 - 1
v_lshlrev_b32 v209, 16, v144                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+9], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+9     // convert C to bf16
buffer_store_short v9, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 10 + 10 - 1
v_lshlrev_b32 v209, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+10], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+10     // convert C to bf16
buffer_store_short v10, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 11 + 11 - 1
v_lshlrev_b32 v209, 16, v146                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+11], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+11     // convert C to bf16
buffer_store_short v11, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 12 + 12 - 1
v_lshlrev_b32 v209, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+12], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+12     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v12, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 13 + 13 - 1
v_lshlrev_b32 v209, 16, v148                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+13], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+13     // convert C to bf16
buffer_store_short v13, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 14 + 14 - 1
v_lshlrev_b32 v209, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+14], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+14     // convert C to bf16
buffer_store_short v14, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 15 + 15 - 1
v_lshlrev_b32 v209, 16, v150                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+15], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+15     // convert C to bf16
buffer_store_short v15, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 16 + 16 - 1
v_lshlrev_b32 v209, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+16], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+16     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v16, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 17 + 17 - 1
v_lshlrev_b32 v209, 16, v152                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+17], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+17     // convert C to bf16
buffer_store_short v17, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 18 + 18 - 1
v_lshlrev_b32 v209, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+18], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+18     // convert C to bf16
buffer_store_short v18, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 19 + 19 - 1
v_lshlrev_b32 v209, 16, v154                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+19], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+19     // convert C to bf16
buffer_store_short v19, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 20 + 20 - 1
v_lshlrev_b32 v209, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+20], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+20     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v20, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 21 + 21 - 1
v_lshlrev_b32 v209, 16, v156                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+21], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+21     // convert C to bf16
buffer_store_short v21, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 22 + 22 - 1
v_lshlrev_b32 v209, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+22], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+22     // convert C to bf16
buffer_store_short v22, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 23 + 23 - 1
v_lshlrev_b32 v209, 16, v158                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+23], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+23     // convert C to bf16
buffer_store_short v23, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 24 + 24 - 1
v_lshlrev_b32 v209, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+24], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+24     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v24, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 25 + 25 - 1
v_lshlrev_b32 v209, 16, v160                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+25], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+25     // convert C to bf16
buffer_store_short v25, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 26 + 26 - 1
v_lshlrev_b32 v209, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+26], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+26     // convert C to bf16
buffer_store_short v26, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 27 + 27 - 1
v_lshlrev_b32 v209, 16, v162                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+27], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+27     // convert C to bf16
buffer_store_short v27, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 28 + 28 - 1
v_lshlrev_b32 v209, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+28], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+28     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v28, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 29 + 29 - 1
v_lshlrev_b32 v209, 16, v164                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+29], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+29     // convert C to bf16
buffer_store_short v29, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 30 + 30 - 1
v_lshlrev_b32 v209, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+30], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+30     // convert C to bf16
buffer_store_short v30, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 31 + 31 - 1
v_lshlrev_b32 v209, 16, v166                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+31], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+31     // convert C to bf16
buffer_store_short v31, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 32 + 32 - 1
v_lshlrev_b32 v209, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+32], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+32     // convert C to bf16
_v_add_lshl_u32 v134, v131, v128, 0x1              // inner add offset
s_mul_i32 s54, s[sgprStrideD1J], 32                // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_short v32, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 33 + 33 - 1
v_lshlrev_b32 v209, 16, v168                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+33], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+33     // convert C to bf16
buffer_store_short v33, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 34 + 34 - 1
v_lshlrev_b32 v209, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+34], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+34     // convert C to bf16
buffer_store_short v34, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 35 + 35 - 1
v_lshlrev_b32 v209, 16, v170                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+35], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+35     // convert C to bf16
buffer_store_short v35, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 36 + 36 - 1
v_lshlrev_b32 v209, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+36], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+36     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v36, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 37 + 37 - 1
v_lshlrev_b32 v209, 16, v172                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+37], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+37     // convert C to bf16
buffer_store_short v37, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 38 + 38 - 1
v_lshlrev_b32 v209, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+38], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+38     // convert C to bf16
buffer_store_short v38, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 39 + 39 - 1
v_lshlrev_b32 v209, 16, v174                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+39], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+39     // convert C to bf16
buffer_store_short v39, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 40 + 40 - 1
v_lshlrev_b32 v209, 16, v175                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+40], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+40     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v40, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 41 + 41 - 1
v_lshlrev_b32 v209, 16, v176                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+41], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+41     // convert C to bf16
buffer_store_short v41, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 42 + 42 - 1
v_lshlrev_b32 v209, 16, v177                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+42], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+42     // convert C to bf16
buffer_store_short v42, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 43 + 43 - 1
v_lshlrev_b32 v209, 16, v178                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+43], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+43     // convert C to bf16
buffer_store_short v43, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 44 + 44 - 1
v_lshlrev_b32 v209, 16, v179                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+44], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+44     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v44, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 45 + 45 - 1
v_lshlrev_b32 v209, 16, v180                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+45], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+45     // convert C to bf16
buffer_store_short v45, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 46 + 46 - 1
v_lshlrev_b32 v209, 16, v181                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+46], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+46     // convert C to bf16
buffer_store_short v46, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 47 + 47 - 1
v_lshlrev_b32 v209, 16, v182                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+47], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+47     // convert C to bf16
buffer_store_short v47, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 48 + 48 - 1
v_lshlrev_b32 v209, 16, v183                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+48], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+48     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v48, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 49 + 49 - 1
v_lshlrev_b32 v209, 16, v184                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+49], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+49     // convert C to bf16
buffer_store_short v49, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 50 + 50 - 1
v_lshlrev_b32 v209, 16, v185                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+50], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+50     // convert C to bf16
buffer_store_short v50, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 51 + 51 - 1
v_lshlrev_b32 v209, 16, v186                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+51], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+51     // convert C to bf16
buffer_store_short v51, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 52 + 52 - 1
v_lshlrev_b32 v209, 16, v187                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+52], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+52     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v52, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 53 + 53 - 1
v_lshlrev_b32 v209, 16, v191                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+53], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+53     // convert C to bf16
buffer_store_short v53, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 54 + 54 - 1
v_lshlrev_b32 v209, 16, v192                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+54], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+54     // convert C to bf16
buffer_store_short v54, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 55 + 55 - 1
v_lshlrev_b32 v209, 16, v193                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+55], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+55     // convert C to bf16
buffer_store_short v55, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 56 + 56 - 1
v_lshlrev_b32 v209, 16, v194                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+56], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+56     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v56, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 57 + 57 - 1
v_lshlrev_b32 v209, 16, v195                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+57], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+57     // convert C to bf16
buffer_store_short v57, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSharedMask=1 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,1,3,2:vw1); (0,1,3,3:vw1); (0,1,3,4:vw1); (0,1,3,5:vw1); (0,1,3,6:vw1); (0,1,3,7:vw1); (0,0,4,0:vw1); (0,0,4,1:vw1); (0,0,4,2:vw1); (0,0,4,3:vw1); (0,0,4,4:vw1); (0,0,4,5:vw1); (0,0,4,6:vw1); (0,0,4,7:vw1); (0,1,4,0:vw1); (0,1,4,1:vw1); (0,1,4,2:vw1); (0,1,4,3:vw1); (0,1,4,4:vw1); (0,1,4,5:vw1); (0,1,4,6:vw1); (0,1,4,7:vw1); (0,0,5,0:vw1); (0,0,5,1:vw1); (0,0,5,2:vw1); (0,0,5,3:vw1); (0,0,5,4:vw1); (0,0,5,5:vw1); (0,0,5,6:vw1); (0,0,5,7:vw1); (0,1,5,0:vw1); (0,1,5,1:vw1); (0,1,5,2:vw1); (0,1,5,3:vw1); (0,1,5,4:vw1); (0,1,5,5:vw1); (0,1,5,6:vw1); (0,1,5,7:vw1); (0,0,6,0:vw1); (0,0,6,1:vw1); (0,0,6,2:vw1); (0,0,6,3:vw1); (0,0,6,4:vw1); (0,0,6,5:vw1); (0,0,6,6:vw1); (0,0,6,7:vw1); (0,1,6,0:vw1); (0,1,6,1:vw1); (0,1,6,2:vw1); (0,1,6,3:vw1); (0,1,6,4:vw1); (0,1,6,5:vw1); (0,1,6,6:vw1); (0,1,6,7:vw1); (0,0,7,0:vw1); (0,0,7,1:vw1); (0,0,7,2:vw1); (0,0,7,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,3,1,2) */
_v_add_lshl_u32 v134, v130, v128, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128
v_add_co_u32 v134, vcc, 0x180, v134                 // add offset 
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,3) */
buffer_load_short_d16 v136, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v137, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,5) */
buffer_load_short_d16 v138, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,6) */
buffer_load_short_d16 v139, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,7) */
buffer_load_short_d16 v140, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
_v_add_lshl_u32 v134, v130, v128, 0x1              // 
s_mov_b32 s54,224
s_mul_i32 s54, s[sgprStrideC1J], s54                // scale StrideC *= numRows(16) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v141, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,1) */
buffer_load_short_d16 v142, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,2) */
buffer_load_short_d16 v143, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,3) */
buffer_load_short_d16 v144, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v145, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,5) */
buffer_load_short_d16 v146, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,6) */
buffer_load_short_d16 v147, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,7) */
buffer_load_short_d16 v148, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,0) */
_v_add_co_u32 v134, vcc, v134, s66                  // load not edge offset
buffer_load_short_d16 v149, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,1) */
buffer_load_short_d16 v150, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,2) */
buffer_load_short_d16 v151, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,3) */
buffer_load_short_d16 v152, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v153, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,5) */
buffer_load_short_d16 v154, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,6) */
buffer_load_short_d16 v155, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,7) */
buffer_load_short_d16 v156, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
_v_add_co_u32 v134, vcc, v134, s66                  // load not edge offset
buffer_load_short_d16 v157, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,0,1) */
buffer_load_short_d16 v158, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,0,2) */
buffer_load_short_d16 v159, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,0,3) */
buffer_load_short_d16 v160, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,0,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v161, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,0,5) */
buffer_load_short_d16 v162, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,0,6) */
buffer_load_short_d16 v163, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,0,7) */
buffer_load_short_d16 v164, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,0) */
_v_add_co_u32 v134, vcc, v134, s66                  // load not edge offset
buffer_load_short_d16 v165, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,1) */
buffer_load_short_d16 v166, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,2) */
buffer_load_short_d16 v167, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,3) */
buffer_load_short_d16 v168, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v169, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,5) */
buffer_load_short_d16 v170, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,6) */
buffer_load_short_d16 v171, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,7) */
buffer_load_short_d16 v172, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
_v_add_lshl_u32 v134, v130, v128, 0x1              // 
s_mul_i32 s54, s[sgprStrideC1J], 32                // scale StrideC *= numRows(16) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v173, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,1) */
buffer_load_short_d16 v174, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,2) */
buffer_load_short_d16 v175, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,3) */
buffer_load_short_d16 v176, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v177, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,5) */
buffer_load_short_d16 v178, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,6) */
buffer_load_short_d16 v179, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,7) */
buffer_load_short_d16 v180, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,0) */
_v_add_co_u32 v134, vcc, v134, s66                  // load not edge offset
buffer_load_short_d16 v181, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,1) */
buffer_load_short_d16 v182, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,2) */
buffer_load_short_d16 v183, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,3) */
buffer_load_short_d16 v184, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v185, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,5) */
buffer_load_short_d16 v186, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,6) */
buffer_load_short_d16 v187, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,7) */
buffer_load_short_d16 v191, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
_v_add_co_u32 v134, vcc, v134, s66                  // load not edge offset
buffer_load_short_d16 v192, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,0,1) */
buffer_load_short_d16 v193, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,0,2) */
buffer_load_short_d16 v194, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,0,3) */
buffer_load_short_d16 v195, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v128, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128
v_add_co_u32 v134, vcc, 0x180, v134                 // add offset 

/* rC *= alpha batchEements=[(0, 1, 3, 2), (0, 1, 3, 3), (0, 1, 3, 4), (0, 1, 3, 5), (0, 1, 3, 6), (0, 1, 3, 7), (0, 0, 4, 0), (0, 0, 4, 1), (0, 0, 4, 2), (0, 0, 4, 3), (0, 0, 4, 4), (0, 0, 4, 5), (0, 0, 4, 6), (0, 0, 4, 7), (0, 1, 4, 0), (0, 1, 4, 1), (0, 1, 4, 2), (0, 1, 4, 3), (0, 1, 4, 4), (0, 1, 4, 5), (0, 1, 4, 6), (0, 1, 4, 7), (0, 0, 5, 0), (0, 0, 5, 1), (0, 0, 5, 2), (0, 0, 5, 3), (0, 0, 5, 4), (0, 0, 5, 5), (0, 0, 5, 6), (0, 0, 5, 7), (0, 1, 5, 0), (0, 1, 5, 1), (0, 1, 5, 2), (0, 1, 5, 3), (0, 1, 5, 4), (0, 1, 5, 5), (0, 1, 5, 6), (0, 1, 5, 7), (0, 0, 6, 0), (0, 0, 6, 1), (0, 0, 6, 2), (0, 0, 6, 3), (0, 0, 6, 4), (0, 0, 6, 5), (0, 0, 6, 6), (0, 0, 6, 7), (0, 1, 6, 0), (0, 1, 6, 1), (0, 1, 6, 2), (0, 1, 6, 3), (0, 1, 6, 4), (0, 1, 6, 5), (0, 1, 6, 6), (0, 1, 6, 7), (0, 0, 7, 0), (0, 0, 7, 1), (0, 0, 7, 2), (0, 0, 7, 3)] */
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // *= alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // *= alpha
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // *= alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // *= alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // *= alpha
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // *= alpha
v_mul_f32 v[vgprValuC+64], s[sgprAlpha], v[vgprValuC+64] // *= alpha
v_mul_f32 v[vgprValuC+65], s[sgprAlpha], v[vgprValuC+65] // *= alpha
v_mul_f32 v[vgprValuC+66], s[sgprAlpha], v[vgprValuC+66] // *= alpha
v_mul_f32 v[vgprValuC+67], s[sgprAlpha], v[vgprValuC+67] // *= alpha
v_mul_f32 v[vgprValuC+68], s[sgprAlpha], v[vgprValuC+68] // *= alpha
v_mul_f32 v[vgprValuC+69], s[sgprAlpha], v[vgprValuC+69] // *= alpha
v_mul_f32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+70] // *= alpha
v_mul_f32 v[vgprValuC+71], s[sgprAlpha], v[vgprValuC+71] // *= alpha
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+72] // *= alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+73] // *= alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+74] // *= alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+75] // *= alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+76] // *= alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+77] // *= alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+78] // *= alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+79] // *= alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+80] // *= alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+81] // *= alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+82] // *= alpha
v_mul_f32 v[vgprValuC+83], s[sgprAlpha], v[vgprValuC+83] // *= alpha
v_mul_f32 v[vgprValuC+84], s[sgprAlpha], v[vgprValuC+84] // *= alpha
v_mul_f32 v[vgprValuC+85], s[sgprAlpha], v[vgprValuC+85] // *= alpha
v_mul_f32 v[vgprValuC+86], s[sgprAlpha], v[vgprValuC+86] // *= alpha
v_mul_f32 v[vgprValuC+87], s[sgprAlpha], v[vgprValuC+87] // *= alpha
v_mul_f32 v[vgprValuC+88], s[sgprAlpha], v[vgprValuC+88] // *= alpha
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+89] // *= alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+90] // *= alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+91] // *= alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+92] // *= alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+93] // *= alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+94] // *= alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+95] // *= alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+96] // *= alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+97] // *= alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+98] // *= alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+99] // *= alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+100] // *= alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+101] // *= alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+102] // *= alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+103] // *= alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+104] // *= alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+105] // *= alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+106] // *= alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+107] // *= alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+108] // *= alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+109] // *= alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+110] // *= alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+111] // *= alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+112] // *= alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+113] // *= alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+114] // *= alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+115] // *= alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16


s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 0 + 0 - 1
v_lshlrev_b32 v209, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+58], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+58     // convert C to bf16
buffer_store_short v58, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 1 + 1 - 1
v_lshlrev_b32 v209, 16, v136                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+59], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+59     // convert C to bf16
buffer_store_short v59, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 2 + 2 - 1
v_lshlrev_b32 v209, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+60], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+60     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v60, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 3 + 3 - 1
v_lshlrev_b32 v209, 16, v138                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+61], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+61     // convert C to bf16
buffer_store_short v61, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 4 + 4 - 1
v_lshlrev_b32 v209, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+62], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+62     // convert C to bf16
buffer_store_short v62, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 5 + 5 - 1
v_lshlrev_b32 v209, 16, v140                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+63], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+63     // convert C to bf16
buffer_store_short v63, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 6 + 6 - 1
v_lshlrev_b32 v209, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+64], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+64     // convert C to bf16
_v_add_lshl_u32 v134, v131, v128, 0x1              // inner add offset
s_mov_b32 s54,224
s_mul_i32 s54, s[sgprStrideD1J], s54                // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_short v64, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 7 + 7 - 1
v_lshlrev_b32 v209, 16, v142                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+65], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+65     // convert C to bf16
buffer_store_short v65, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 8 + 8 - 1
v_lshlrev_b32 v209, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+66], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+66     // convert C to bf16
buffer_store_short v66, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 9 + 9 - 1
v_lshlrev_b32 v209, 16, v144                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+67], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+67     // convert C to bf16
buffer_store_short v67, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 10 + 10 - 1
v_lshlrev_b32 v209, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+68], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+68     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v68, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 11 + 11 - 1
v_lshlrev_b32 v209, 16, v146                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+69], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+69     // convert C to bf16
buffer_store_short v69, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 12 + 12 - 1
v_lshlrev_b32 v209, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+70], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+70     // convert C to bf16
buffer_store_short v70, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 13 + 13 - 1
v_lshlrev_b32 v209, 16, v148                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+71], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+71     // convert C to bf16
buffer_store_short v71, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 14 + 14 - 1
v_lshlrev_b32 v209, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+72], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+72     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v72, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 15 + 15 - 1
v_lshlrev_b32 v209, 16, v150                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+73], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+73     // convert C to bf16
buffer_store_short v73, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 16 + 16 - 1
v_lshlrev_b32 v209, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+74], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+74     // convert C to bf16
buffer_store_short v74, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 17 + 17 - 1
v_lshlrev_b32 v209, 16, v152                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+75], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+75     // convert C to bf16
buffer_store_short v75, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 18 + 18 - 1
v_lshlrev_b32 v209, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+76], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+76     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v76, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 19 + 19 - 1
v_lshlrev_b32 v209, 16, v154                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+77], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+77     // convert C to bf16
buffer_store_short v77, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 20 + 20 - 1
v_lshlrev_b32 v209, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+78], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+78     // convert C to bf16
buffer_store_short v78, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 21 + 21 - 1
v_lshlrev_b32 v209, 16, v156                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+79], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+79     // convert C to bf16
buffer_store_short v79, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 22 + 22 - 1
v_lshlrev_b32 v209, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+80], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+80     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v80, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 23 + 23 - 1
v_lshlrev_b32 v209, 16, v158                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+81], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+81     // convert C to bf16
buffer_store_short v81, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 24 + 24 - 1
v_lshlrev_b32 v209, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+82], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+82     // convert C to bf16
buffer_store_short v82, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 25 + 25 - 1
v_lshlrev_b32 v209, 16, v160                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+83], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+83     // convert C to bf16
buffer_store_short v83, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 26 + 26 - 1
v_lshlrev_b32 v209, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+84], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+84     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v84, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 27 + 27 - 1
v_lshlrev_b32 v209, 16, v162                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+85], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+85     // convert C to bf16
buffer_store_short v85, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 28 + 28 - 1
v_lshlrev_b32 v209, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+86], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+86     // convert C to bf16
buffer_store_short v86, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 29 + 29 - 1
v_lshlrev_b32 v209, 16, v164                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+87], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+87     // convert C to bf16
buffer_store_short v87, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 30 + 30 - 1
v_lshlrev_b32 v209, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+88], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+88     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v88, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 31 + 31 - 1
v_lshlrev_b32 v209, 16, v166                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+89], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+89     // convert C to bf16
buffer_store_short v89, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 32 + 32 - 1
v_lshlrev_b32 v209, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+90], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+90     // convert C to bf16
buffer_store_short v90, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 33 + 33 - 1
v_lshlrev_b32 v209, 16, v168                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+91], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+91     // convert C to bf16
buffer_store_short v91, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 34 + 34 - 1
v_lshlrev_b32 v209, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+92], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+92     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v92, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 35 + 35 - 1
v_lshlrev_b32 v209, 16, v170                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+93], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+93     // convert C to bf16
buffer_store_short v93, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 36 + 36 - 1
v_lshlrev_b32 v209, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+94], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+94     // convert C to bf16
buffer_store_short v94, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 37 + 37 - 1
v_lshlrev_b32 v209, 16, v172                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+95], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+95     // convert C to bf16
buffer_store_short v95, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 38 + 38 - 1
v_lshlrev_b32 v209, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+96], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+96     // convert C to bf16
_v_add_lshl_u32 v134, v131, v128, 0x1              // inner add offset
s_mul_i32 s54, s[sgprStrideD1J], 32                // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_short v96, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 39 + 39 - 1
v_lshlrev_b32 v209, 16, v174                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+97], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+97     // convert C to bf16
buffer_store_short v97, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 40 + 40 - 1
v_lshlrev_b32 v209, 16, v175                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+98], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+98     // convert C to bf16
buffer_store_short v98, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 41 + 41 - 1
v_lshlrev_b32 v209, 16, v176                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+99], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+99     // convert C to bf16
buffer_store_short v99, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 42 + 42 - 1
v_lshlrev_b32 v209, 16, v177                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+100], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+100     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v100, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 43 + 43 - 1
v_lshlrev_b32 v209, 16, v178                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+101], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+101     // convert C to bf16
buffer_store_short v101, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 44 + 44 - 1
v_lshlrev_b32 v209, 16, v179                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+102], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+102     // convert C to bf16
buffer_store_short v102, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 45 + 45 - 1
v_lshlrev_b32 v209, 16, v180                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+103], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+103     // convert C to bf16
buffer_store_short v103, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 46 + 46 - 1
v_lshlrev_b32 v209, 16, v181                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+104], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+104     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v104, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 47 + 47 - 1
v_lshlrev_b32 v209, 16, v182                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+105], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+105     // convert C to bf16
buffer_store_short v105, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 48 + 48 - 1
v_lshlrev_b32 v209, 16, v183                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+106], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+106     // convert C to bf16
buffer_store_short v106, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 49 + 49 - 1
v_lshlrev_b32 v209, 16, v184                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+107], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+107     // convert C to bf16
buffer_store_short v107, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 50 + 50 - 1
v_lshlrev_b32 v209, 16, v185                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+108], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+108     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v108, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 51 + 51 - 1
v_lshlrev_b32 v209, 16, v186                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+109], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+109     // convert C to bf16
buffer_store_short v109, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 52 + 52 - 1
v_lshlrev_b32 v209, 16, v187                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+110], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+110     // convert C to bf16
buffer_store_short v110, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 53 + 53 - 1
v_lshlrev_b32 v209, 16, v191                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+111], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+111     // convert C to bf16
buffer_store_short v111, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 54 + 54 - 1
v_lshlrev_b32 v209, 16, v192                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+112], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+112     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v112, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 55 + 55 - 1
v_lshlrev_b32 v209, 16, v193                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+113], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+113     // convert C to bf16
buffer_store_short v113, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 56 + 56 - 1
v_lshlrev_b32 v209, 16, v194                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+114], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+114     // convert C to bf16
buffer_store_short v114, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(57)                                // wait C (interleaved) 57 = 58 - 57 + 57 - 1
v_lshlrev_b32 v209, 16, v195                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+115], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+115     // convert C to bf16
buffer_store_short v115, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSharedMask=1 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,0,7,4:vw1); (0,0,7,5:vw1); (0,0,7,6:vw1); (0,0,7,7:vw1); (0,1,7,0:vw1); (0,1,7,1:vw1); (0,1,7,2:vw1); (0,1,7,3:vw1); (0,1,7,4:vw1); (0,1,7,5:vw1); (0,1,7,6:vw1); (0,1,7,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,7,0,4) */
_v_add_lshl_u32 v134, v130, v128, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128
v_add_co_u32 v134, vcc, 0x100, v134                 // add offset 
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,0,5) */
buffer_load_short_d16 v136, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,0,6) */
buffer_load_short_d16 v137, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,0,7) */
buffer_load_short_d16 v138, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,0) */
_v_add_co_u32 v134, vcc, v134, s66                  // load not edge offset
buffer_load_short_d16 v139, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,1) */
buffer_load_short_d16 v140, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,2) */
buffer_load_short_d16 v141, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,3) */
buffer_load_short_d16 v142, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v143, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,5) */
buffer_load_short_d16 v144, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,6) */
buffer_load_short_d16 v145, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,7) */
buffer_load_short_d16 v146, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v128, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128
v_add_co_u32 v134, vcc, 0x100, v134                 // add offset 

/* rC *= alpha batchEements=[(0, 0, 7, 4), (0, 0, 7, 5), (0, 0, 7, 6), (0, 0, 7, 7), (0, 1, 7, 0), (0, 1, 7, 1), (0, 1, 7, 2), (0, 1, 7, 3), (0, 1, 7, 4), (0, 1, 7, 5), (0, 1, 7, 6), (0, 1, 7, 7)] */
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+116] // *= alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+117] // *= alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+118] // *= alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+119] // *= alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+120] // *= alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+121] // *= alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+122] // *= alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+123] // *= alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+124] // *= alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+125] // *= alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+126] // *= alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+127] // *= alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16


s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 12 - 0 + 0 - 1
v_lshlrev_b32 v209, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+116], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+116     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v116, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 12 - 1 + 1 - 1
v_lshlrev_b32 v209, 16, v136                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+117], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+117     // convert C to bf16
buffer_store_short v117, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 12 - 2 + 2 - 1
v_lshlrev_b32 v209, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+118], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+118     // convert C to bf16
buffer_store_short v118, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 12 - 3 + 3 - 1
v_lshlrev_b32 v209, 16, v138                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+119], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+119     // convert C to bf16
buffer_store_short v119, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 12 - 4 + 4 - 1
v_lshlrev_b32 v209, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+120], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+120     // convert C to bf16
v_add_co_u32 v134, vcc, s66, v134                 // add offset1
buffer_store_short v120, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 12 - 5 + 5 - 1
v_lshlrev_b32 v209, 16, v140                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+121], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+121     // convert C to bf16
buffer_store_short v121, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 12 - 6 + 6 - 1
v_lshlrev_b32 v209, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+122], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+122     // convert C to bf16
buffer_store_short v122, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 12 - 7 + 7 - 1
v_lshlrev_b32 v209, 16, v142                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+123], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+123     // convert C to bf16
buffer_store_short v123, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 12 - 8 + 8 - 1
v_lshlrev_b32 v209, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+124], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+124     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v124, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 12 - 9 + 9 - 1
v_lshlrev_b32 v209, 16, v144                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+125], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+125     // convert C to bf16
buffer_store_short v125, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 12 - 10 + 10 - 1
v_lshlrev_b32 v209, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+126], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+126     // convert C to bf16
buffer_store_short v126, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 12 - 11 + 11 - 1
v_lshlrev_b32 v209, 16, v146                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+127], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+127     // convert C to bf16
buffer_store_short v127, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
s_branch label_GW_End_109                          // jump to end
GW_B1_E1_108:

/* edge=1, allocate 46 sgpr. perBatch=6 perElement=2 elementsPerBatch=20 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,0,4:vw1); (0,0,0,5:vw1); (0,0,0,6:vw1); (0,0,0,7:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1); (0,1,0,4:vw1); (0,1,0,5:vw1); (0,1,0,6:vw1); (0,1,0,7:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v136, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffC+0], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDC clip if OOB. offset
buffer_load_short_d16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+0], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v138, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffC+1], v138  // add 4* loop offset
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDC clip if OOB. offset
buffer_load_short_d16 v139, v138, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v138, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+1], v138  // add 4* loop offset
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v140, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffC+2], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v141, v140, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v140, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,4) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,5) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffC+0], v144  // add 4* loop offset
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v145, v144, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+0], v144  // add 4* loop offset
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,6) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v146, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffC+1], v146  // add 4* loop offset
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDC clip if OOB. offset
buffer_load_short_d16 v147, v146, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+1], v146  // add 4* loop offset
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,7) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v148, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffC+2], v148  // add 4* loop offset
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDC clip if OOB. offset
buffer_load_short_d16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v150, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDC clip if OOB. offset
buffer_load_short_d16 v151, v150, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v152, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffC+0], v152  // add 4* loop offset
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDC clip if OOB. offset
buffer_load_short_d16 v153, v152, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+0], v152  // add 4* loop offset
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v154, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffC+1], v154  // add 4* loop offset
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDC clip if OOB. offset
buffer_load_short_d16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+1], v154  // add 4* loop offset
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v156, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffC+2], v156  // add 4* loop offset
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDC clip if OOB. offset
buffer_load_short_d16 v157, v156, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v156, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffD+2], v156  // add 4* loop offset
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,4) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v158, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDC clip if OOB. offset
buffer_load_short_d16 v159, v158, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v158, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,5) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v160, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffC+0], v160  // add 4* loop offset
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDC clip if OOB. offset
buffer_load_short_d16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffD+0], v160  // add 4* loop offset
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,6) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v162, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffC+1], v162  // add 4* loop offset
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDC clip if OOB. offset
buffer_load_short_d16 v163, v162, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v162, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffD+1], v162  // add 4* loop offset
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,7) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v164, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffC+2], v164  // add 4* loop offset
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDC clip if OOB. offset
buffer_load_short_d16 v165, v164, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffD+2], v164  // add 4* loop offset
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
s_mov_b32 s54,128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v166, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDC clip if OOB. offset
buffer_load_short_d16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v168, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffC+0], v168  // add 4* loop offset
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDC clip if OOB. offset
buffer_load_short_d16 v169, v168, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v168, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffD+0], v168  // add 4* loop offset
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v170, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffC+1], v170  // add 4* loop offset
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDC clip if OOB. offset
buffer_load_short_d16 v171, v170, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v170, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffD+1], v170  // add 4* loop offset
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v172, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffC+2], v172  // add 4* loop offset
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDC clip if OOB. offset
buffer_load_short_d16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffD+2], v172  // add 4* loop offset
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 0, 4), (0, 0, 0, 5), (0, 0, 0, 6), (0, 0, 0, 7), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (0, 1, 0, 4), (0, 1, 0, 5), (0, 1, 0, 6), (0, 1, 0, 7), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // *= alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // *= alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

v_lshlrev_b32 v209, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+0], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+0     // convert C to bf16
buffer_store_short v0, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+1], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+1     // convert C to bf16
buffer_store_short v1, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+2], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+2     // convert C to bf16
buffer_store_short v2, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+3], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+3     // convert C to bf16
buffer_store_short v3, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+4], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+4     // convert C to bf16
buffer_store_short v4, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+5], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+5     // convert C to bf16
buffer_store_short v5, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+6], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+6     // convert C to bf16
buffer_store_short v6, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+7], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+7     // convert C to bf16
buffer_store_short v7, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+8], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+8     // convert C to bf16
buffer_store_short v8, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+9], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+9     // convert C to bf16
buffer_store_short v9, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+10], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+10     // convert C to bf16
buffer_store_short v10, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+11], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+11     // convert C to bf16
buffer_store_short v11, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+12], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+12     // convert C to bf16
buffer_store_short v12, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+13], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+13     // convert C to bf16
buffer_store_short v13, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+14], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+14     // convert C to bf16
buffer_store_short v14, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+15], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+15     // convert C to bf16
buffer_store_short v15, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+16], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+16     // convert C to bf16
buffer_store_short v16, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+17], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+17     // convert C to bf16
buffer_store_short v17, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+18], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+18     // convert C to bf16
buffer_store_short v18, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+19], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+19     // convert C to bf16
buffer_store_short v19, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,1,4:vw1); (0,0,1,5:vw1); (0,0,1,6:vw1); (0,0,1,7:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,1,1,2:vw1); (0,1,1,3:vw1); (0,1,1,4:vw1); (0,1,1,5:vw1); (0,1,1,6:vw1); (0,1,1,7:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,2,4:vw1); (0,0,2,5:vw1); (0,0,2,6:vw1); (0,0,2,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,1,0,4) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,5) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v136, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffC+0], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDC clip if OOB. offset
buffer_load_short_d16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+0], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,6) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v138, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffC+1], v138  // add 4* loop offset
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDC clip if OOB. offset
buffer_load_short_d16 v139, v138, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+1], v138  // add 4* loop offset
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,7) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v140, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffC+2], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v141, v140, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,1) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=1
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffC+0], v144  // add 4* loop offset
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v145, v144, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+0], v144  // add 4* loop offset
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,2) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=2
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v146, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffC+1], v146  // add 4* loop offset
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDC clip if OOB. offset
buffer_load_short_d16 v147, v146, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+1], v146  // add 4* loop offset
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,3) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=3
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v148, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffC+2], v148  // add 4* loop offset
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDC clip if OOB. offset
buffer_load_short_d16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,4) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v150, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDC clip if OOB. offset
buffer_load_short_d16 v151, v150, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,5) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v152, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffC+0], v152  // add 4* loop offset
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDC clip if OOB. offset
buffer_load_short_d16 v153, v152, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+0], v152  // add 4* loop offset
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,6) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v154, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffC+1], v154  // add 4* loop offset
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDC clip if OOB. offset
buffer_load_short_d16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+1], v154  // add 4* loop offset
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,7) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v156, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffC+2], v156  // add 4* loop offset
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDC clip if OOB. offset
buffer_load_short_d16 v157, v156, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v156, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffD+2], v156  // add 4* loop offset
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v158, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
s_mul_i32 s[sgprEdgeCheck], s[sgprStrideC1J], 32   // add offset stride
v_add_co_u32 v158, vcc, s[sgprEdgeCheck], v158     // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDC clip if OOB. offset
buffer_load_short_d16 v159, v158, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v158, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
s_mul_i32 s[sgprEdgeCheck+1], s[sgprStrideD1J], 32 // add offset stride
v_add_co_u32 v158, vcc, s[sgprEdgeCheck+1], v158   // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v160, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffC+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck], v160     // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDC clip if OOB. offset
buffer_load_short_d16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffD+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck+1], v160   // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v162, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffC+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck], v162     // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDC clip if OOB. offset
buffer_load_short_d16 v163, v162, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v162, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffD+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck+1], v162   // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v164, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffC+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck], v164     // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDC clip if OOB. offset
buffer_load_short_d16 v165, v164, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffD+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck+1], v164   // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,4) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v166, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v166, vcc, s[sgprEdgeCheck], v166     // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDC clip if OOB. offset
buffer_load_short_d16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v166, vcc, s[sgprEdgeCheck+1], v166   // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,5) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v168, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffC+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck], v168     // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDC clip if OOB. offset
buffer_load_short_d16 v169, v168, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v168, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffD+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck+1], v168   // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,6) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v170, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffC+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck], v170     // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDC clip if OOB. offset
buffer_load_short_d16 v171, v170, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v170, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffD+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck+1], v170   // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,7) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v172, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffC+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck], v172     // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDC clip if OOB. offset
buffer_load_short_d16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffD+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck+1], v172   // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 0, 1, 4), (0, 0, 1, 5), (0, 0, 1, 6), (0, 0, 1, 7), (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 1, 3), (0, 1, 1, 4), (0, 1, 1, 5), (0, 1, 1, 6), (0, 1, 1, 7), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 2, 4), (0, 0, 2, 5), (0, 0, 2, 6), (0, 0, 2, 7)] */
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // *= alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // *= alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // *= alpha
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // *= alpha
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // *= alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // *= alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // *= alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // *= alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // *= alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

v_lshlrev_b32 v209, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+20], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+20     // convert C to bf16
buffer_store_short v20, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+21], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+21     // convert C to bf16
buffer_store_short v21, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+22], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+22     // convert C to bf16
buffer_store_short v22, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+23], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+23     // convert C to bf16
buffer_store_short v23, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+24], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+24     // convert C to bf16
buffer_store_short v24, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+25], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+25     // convert C to bf16
buffer_store_short v25, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+26], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+26     // convert C to bf16
buffer_store_short v26, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+27], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+27     // convert C to bf16
buffer_store_short v27, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+28], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+28     // convert C to bf16
buffer_store_short v28, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+29], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+29     // convert C to bf16
buffer_store_short v29, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+30], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+30     // convert C to bf16
buffer_store_short v30, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+31], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+31     // convert C to bf16
buffer_store_short v31, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+32], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+32     // convert C to bf16
buffer_store_short v32, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+33], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+33     // convert C to bf16
buffer_store_short v33, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+34], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+34     // convert C to bf16
buffer_store_short v34, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+35], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+35     // convert C to bf16
buffer_store_short v35, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+36], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+36     // convert C to bf16
buffer_store_short v36, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+37], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+37     // convert C to bf16
buffer_store_short v37, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+38], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+38     // convert C to bf16
buffer_store_short v38, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+39], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+39     // convert C to bf16
buffer_store_short v39, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,1,2,0:vw1); (0,1,2,1:vw1); (0,1,2,2:vw1); (0,1,2,3:vw1); (0,1,2,4:vw1); (0,1,2,5:vw1); (0,1,2,6:vw1); (0,1,2,7:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,0,3,4:vw1); (0,0,3,5:vw1); (0,0,3,6:vw1); (0,0,3,7:vw1); (0,1,3,0:vw1); (0,1,3,1:vw1); (0,1,3,2:vw1); (0,1,3,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck], v134     // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,1) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v136, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffC+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck], v136     // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDC clip if OOB. offset
buffer_load_short_d16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,2) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v138, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffC+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck], v138     // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDC clip if OOB. offset
buffer_load_short_d16 v139, v138, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,3) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v140, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffC+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck], v140     // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v141, v140, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,4) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck], v142     // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,5) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffC+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck], v144     // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v145, v144, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,6) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v146, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffC+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck], v146     // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDC clip if OOB. offset
buffer_load_short_d16 v147, v146, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,7) */
s_mov_b32 s54, 80
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v148, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffC+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck], v148     // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDC clip if OOB. offset
buffer_load_short_d16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v150, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck], v150     // 
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDC clip if OOB. offset
buffer_load_short_d16 v151, v150, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck+1], v150   // 
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v152, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffC+0], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck], v152     // 
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDC clip if OOB. offset
buffer_load_short_d16 v153, v152, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+0], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck+1], v152   // 
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v154, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffC+1], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck], v154     // 
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDC clip if OOB. offset
buffer_load_short_d16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+1], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck+1], v154   // 
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
s_mov_b32 s54, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v156, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffC+2], v156  // add 4* loop offset
v_add_co_u32 v156, vcc, s[sgprEdgeCheck], v156     // 
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDC clip if OOB. offset
buffer_load_short_d16 v157, v156, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v156, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffD+2], v156  // add 4* loop offset
v_add_co_u32 v156, vcc, s[sgprEdgeCheck+1], v156   // 
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,4) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v158, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v158, vcc, s[sgprEdgeCheck], v158     // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDC clip if OOB. offset
buffer_load_short_d16 v159, v158, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v158, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v158, vcc, s[sgprEdgeCheck+1], v158   // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,5) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v160, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffC+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck], v160     // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDC clip if OOB. offset
buffer_load_short_d16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffD+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck+1], v160   // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,6) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v162, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffC+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck], v162     // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDC clip if OOB. offset
buffer_load_short_d16 v163, v162, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v162, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffD+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck+1], v162   // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,7) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v164, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffC+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck], v164     // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDC clip if OOB. offset
buffer_load_short_d16 v165, v164, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffD+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck+1], v164   // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v166, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v166, vcc, s[sgprEdgeCheck], v166     // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDC clip if OOB. offset
buffer_load_short_d16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v166, vcc, s[sgprEdgeCheck+1], v166   // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,1) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=1
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v168, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffC+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck], v168     // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDC clip if OOB. offset
buffer_load_short_d16 v169, v168, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v168, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffD+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck+1], v168   // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,2) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=2
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v170, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffC+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck], v170     // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDC clip if OOB. offset
buffer_load_short_d16 v171, v170, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v170, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffD+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck+1], v170   // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,3) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=3
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v172, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffC+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck], v172     // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDC clip if OOB. offset
buffer_load_short_d16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffD+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck+1], v172   // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 1, 2, 0), (0, 1, 2, 1), (0, 1, 2, 2), (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 5), (0, 1, 2, 6), (0, 1, 2, 7), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 0, 3, 4), (0, 0, 3, 5), (0, 0, 3, 6), (0, 0, 3, 7), (0, 1, 3, 0), (0, 1, 3, 1), (0, 1, 3, 2), (0, 1, 3, 3)] */
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // *= alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // *= alpha
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // *= alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // *= alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // *= alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // *= alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // *= alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // *= alpha
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // *= alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // *= alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // *= alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // *= alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // *= alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // *= alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // *= alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // *= alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // *= alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // *= alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // *= alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

v_lshlrev_b32 v209, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+40], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+40     // convert C to bf16
buffer_store_short v40, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+41], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+41     // convert C to bf16
buffer_store_short v41, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+42], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+42     // convert C to bf16
buffer_store_short v42, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+43], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+43     // convert C to bf16
buffer_store_short v43, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+44], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+44     // convert C to bf16
buffer_store_short v44, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+45], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+45     // convert C to bf16
buffer_store_short v45, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+46], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+46     // convert C to bf16
buffer_store_short v46, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+47], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+47     // convert C to bf16
buffer_store_short v47, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+48], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+48     // convert C to bf16
buffer_store_short v48, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+49], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+49     // convert C to bf16
buffer_store_short v49, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+50], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+50     // convert C to bf16
buffer_store_short v50, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+51], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+51     // convert C to bf16
buffer_store_short v51, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+52], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+52     // convert C to bf16
buffer_store_short v52, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+53], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+53     // convert C to bf16
buffer_store_short v53, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+54], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+54     // convert C to bf16
buffer_store_short v54, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+55], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+55     // convert C to bf16
buffer_store_short v55, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+56], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+56     // convert C to bf16
buffer_store_short v56, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+57], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+57     // convert C to bf16
buffer_store_short v57, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+58], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+58     // convert C to bf16
buffer_store_short v58, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+59], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+59     // convert C to bf16
buffer_store_short v59, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (0,1,3,4:vw1); (0,1,3,5:vw1); (0,1,3,6:vw1); (0,1,3,7:vw1); (0,0,4,0:vw1); (0,0,4,1:vw1); (0,0,4,2:vw1); (0,0,4,3:vw1); (0,0,4,4:vw1); (0,0,4,5:vw1); (0,0,4,6:vw1); (0,0,4,7:vw1); (0,1,4,0:vw1); (0,1,4,1:vw1); (0,1,4,2:vw1); (0,1,4,3:vw1); (0,1,4,4:vw1); (0,1,4,5:vw1); (0,1,4,6:vw1); (0,1,4,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,3,1,4) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck], v134     // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,5) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v136, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffC+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck], v136     // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDC clip if OOB. offset
buffer_load_short_d16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,6) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v138, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffC+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck], v138     // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDC clip if OOB. offset
buffer_load_short_d16 v139, v138, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,7) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v140, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffC+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck], v140     // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v141, v140, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
s_mov_b32 s55, 128
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
s_mul_i32 s[sgprEdgeCheck], s[sgprStrideC1J], 0x100   // add offset stride
v_add_co_u32 v142, vcc, s[sgprEdgeCheck], v142     // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
s_mul_i32 s[sgprEdgeCheck+1], s[sgprStrideD1J], 0x100 // add offset stride
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,1) */
s_mov_b32 s55, 132
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v144, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffC+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck], v144     // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v145, v144, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,2) */
s_mov_b32 s55, 136
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v146, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffC+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck], v146     // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDC clip if OOB. offset
buffer_load_short_d16 v147, v146, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,3) */
s_mov_b32 s55, 140
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v148, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffC+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck], v148     // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDC clip if OOB. offset
buffer_load_short_d16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,4) */
s_mov_b32 s55, 128
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v150, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck], v150     // 
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDC clip if OOB. offset
buffer_load_short_d16 v151, v150, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck+1], v150   // 
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,5) */
s_mov_b32 s55, 132
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v152, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffC+0], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck], v152     // 
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDC clip if OOB. offset
buffer_load_short_d16 v153, v152, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+0], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck+1], v152   // 
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,6) */
s_mov_b32 s55, 136
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v154, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffC+1], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck], v154     // 
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDC clip if OOB. offset
buffer_load_short_d16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+1], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck+1], v154   // 
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,7) */
s_mov_b32 s55, 140
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v156, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffC+2], v156  // add 4* loop offset
v_add_co_u32 v156, vcc, s[sgprEdgeCheck], v156     // 
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDC clip if OOB. offset
buffer_load_short_d16 v157, v156, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v156, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffD+2], v156  // add 4* loop offset
v_add_co_u32 v156, vcc, s[sgprEdgeCheck+1], v156   // 
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,0) */
s_mov_b32 s55, 128
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v158, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v158, vcc, s[sgprEdgeCheck], v158     // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDC clip if OOB. offset
buffer_load_short_d16 v159, v158, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v158, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v158, vcc, s[sgprEdgeCheck+1], v158   // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,1) */
s_mov_b32 s55, 132
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v160, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffC+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck], v160     // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDC clip if OOB. offset
buffer_load_short_d16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffD+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck+1], v160   // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,2) */
s_mov_b32 s55, 136
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v162, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffC+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck], v162     // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDC clip if OOB. offset
buffer_load_short_d16 v163, v162, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v162, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffD+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck+1], v162   // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,3) */
s_mov_b32 s55, 140
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v164, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffC+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck], v164     // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDC clip if OOB. offset
buffer_load_short_d16 v165, v164, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffD+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck+1], v164   // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,4) */
s_mov_b32 s54, 80
s_mov_b32 s55, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v166, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v166, vcc, s[sgprEdgeCheck], v166     // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDC clip if OOB. offset
buffer_load_short_d16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v166, vcc, s[sgprEdgeCheck+1], v166   // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,5) */
s_mov_b32 s54, 80
s_mov_b32 s55, 132
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v168, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffC+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck], v168     // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDC clip if OOB. offset
buffer_load_short_d16 v169, v168, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v168, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffD+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck+1], v168   // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,6) */
s_mov_b32 s54, 80
s_mov_b32 s55, 136
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v170, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffC+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck], v170     // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDC clip if OOB. offset
buffer_load_short_d16 v171, v170, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v170, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffD+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck+1], v170   // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,7) */
s_mov_b32 s54, 80
s_mov_b32 s55, 140
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v172, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffC+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck], v172     // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDC clip if OOB. offset
buffer_load_short_d16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffD+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck+1], v172   // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 1, 3, 4), (0, 1, 3, 5), (0, 1, 3, 6), (0, 1, 3, 7), (0, 0, 4, 0), (0, 0, 4, 1), (0, 0, 4, 2), (0, 0, 4, 3), (0, 0, 4, 4), (0, 0, 4, 5), (0, 0, 4, 6), (0, 0, 4, 7), (0, 1, 4, 0), (0, 1, 4, 1), (0, 1, 4, 2), (0, 1, 4, 3), (0, 1, 4, 4), (0, 1, 4, 5), (0, 1, 4, 6), (0, 1, 4, 7)] */
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // *= alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // *= alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // *= alpha
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // *= alpha
v_mul_f32 v[vgprValuC+64], s[sgprAlpha], v[vgprValuC+64] // *= alpha
v_mul_f32 v[vgprValuC+65], s[sgprAlpha], v[vgprValuC+65] // *= alpha
v_mul_f32 v[vgprValuC+66], s[sgprAlpha], v[vgprValuC+66] // *= alpha
v_mul_f32 v[vgprValuC+67], s[sgprAlpha], v[vgprValuC+67] // *= alpha
v_mul_f32 v[vgprValuC+68], s[sgprAlpha], v[vgprValuC+68] // *= alpha
v_mul_f32 v[vgprValuC+69], s[sgprAlpha], v[vgprValuC+69] // *= alpha
v_mul_f32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+70] // *= alpha
v_mul_f32 v[vgprValuC+71], s[sgprAlpha], v[vgprValuC+71] // *= alpha
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+72] // *= alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+73] // *= alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+74] // *= alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+75] // *= alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+76] // *= alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+77] // *= alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+78] // *= alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+79] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

v_lshlrev_b32 v209, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+60], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+60     // convert C to bf16
buffer_store_short v60, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+61], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+61     // convert C to bf16
buffer_store_short v61, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+62], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+62     // convert C to bf16
buffer_store_short v62, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+63], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+63     // convert C to bf16
buffer_store_short v63, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+64], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+64     // convert C to bf16
buffer_store_short v64, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+65], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+65     // convert C to bf16
buffer_store_short v65, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+66], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+66     // convert C to bf16
buffer_store_short v66, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+67], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+67     // convert C to bf16
buffer_store_short v67, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+68], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+68     // convert C to bf16
buffer_store_short v68, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+69], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+69     // convert C to bf16
buffer_store_short v69, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+70], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+70     // convert C to bf16
buffer_store_short v70, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+71], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+71     // convert C to bf16
buffer_store_short v71, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+72], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+72     // convert C to bf16
buffer_store_short v72, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+73], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+73     // convert C to bf16
buffer_store_short v73, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+74], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+74     // convert C to bf16
buffer_store_short v74, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+75], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+75     // convert C to bf16
buffer_store_short v75, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+76], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+76     // convert C to bf16
buffer_store_short v76, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+77], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+77     // convert C to bf16
buffer_store_short v77, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+78], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+78     // convert C to bf16
buffer_store_short v78, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+79], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+79     // convert C to bf16
buffer_store_short v79, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #4 (d1,d0,vc1,vc0) = */
/*    (0,0,5,0:vw1); (0,0,5,1:vw1); (0,0,5,2:vw1); (0,0,5,3:vw1); (0,0,5,4:vw1); (0,0,5,5:vw1); (0,0,5,6:vw1); (0,0,5,7:vw1); (0,1,5,0:vw1); (0,1,5,1:vw1); (0,1,5,2:vw1); (0,1,5,3:vw1); (0,1,5,4:vw1); (0,1,5,5:vw1); (0,1,5,6:vw1); (0,1,5,7:vw1); (0,0,6,0:vw1); (0,0,6,1:vw1); (0,0,6,2:vw1); (0,0,6,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
s_mov_b32 s54, 128
s_mov_b32 s55, 128
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck], v134     // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,1) */
s_mov_b32 s54, 128
s_mov_b32 s55, 132
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v136, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffC+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck], v136     // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDC clip if OOB. offset
buffer_load_short_d16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,2) */
s_mov_b32 s54, 128
s_mov_b32 s55, 136
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v138, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffC+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck], v138     // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDC clip if OOB. offset
buffer_load_short_d16 v139, v138, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,3) */
s_mov_b32 s54, 128
s_mov_b32 s55, 140
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v140, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffC+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck], v140     // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v141, v140, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,4) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 128
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck], v142     // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,5) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 132
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffC+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck], v144     // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v145, v144, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,6) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 136
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v146, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffC+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck], v146     // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDC clip if OOB. offset
buffer_load_short_d16 v147, v146, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,7) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 140
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v148, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffC+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck], v148     // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDC clip if OOB. offset
buffer_load_short_d16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,0) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=0
s_mov_b32 s55, 128
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v150, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck], v150     // 
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDC clip if OOB. offset
buffer_load_short_d16 v151, v150, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck+1], v150   // 
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,1) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=0
s_mov_b32 s55, 132
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v152, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffC+0], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck], v152     // 
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDC clip if OOB. offset
buffer_load_short_d16 v153, v152, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+0], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck+1], v152   // 
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,2) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=0
s_mov_b32 s55, 136
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v154, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffC+1], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck], v154     // 
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDC clip if OOB. offset
buffer_load_short_d16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+1], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck+1], v154   // 
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,3) */
s_mov_b32 s54, 192                                  // coordOffset0 d0=1 vc0=0
s_mov_b32 s55, 140
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v156, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffC+2], v156  // add 4* loop offset
v_add_co_u32 v156, vcc, s[sgprEdgeCheck], v156     // 
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDC clip if OOB. offset
buffer_load_short_d16 v157, v156, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v156, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffD+2], v156  // add 4* loop offset
v_add_co_u32 v156, vcc, s[sgprEdgeCheck+1], v156   // 
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,4) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
s_mov_b32 s55, 128
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v158, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v158, vcc, s[sgprEdgeCheck], v158     // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDC clip if OOB. offset
buffer_load_short_d16 v159, v158, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v158, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v158, vcc, s[sgprEdgeCheck+1], v158   // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,5) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
s_mov_b32 s55, 132
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v160, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffC+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck], v160     // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDC clip if OOB. offset
buffer_load_short_d16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffD+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck+1], v160   // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,6) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
s_mov_b32 s55, 136
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v162, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffC+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck], v162     // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDC clip if OOB. offset
buffer_load_short_d16 v163, v162, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v162, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffD+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck+1], v162   // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,7) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
s_mov_b32 s55, 140
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v164, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffC+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck], v164     // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDC clip if OOB. offset
buffer_load_short_d16 v165, v164, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffD+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck+1], v164   // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
s_mov_b32 s55, 144
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v166, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
s_mov_b32 s54, 288
s_mul_i32 s[sgprEdgeCheck], s[sgprStrideC1J], s54   // add offset stride
v_add_co_u32 v166, vcc, s[sgprEdgeCheck], v166     // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDC clip if OOB. offset
buffer_load_short_d16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
s_mul_i32 s[sgprEdgeCheck+1], s[sgprStrideD1J], s54 // add offset stride
v_add_co_u32 v166, vcc, s[sgprEdgeCheck+1], v166   // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,1) */
s_mov_b32 s55, 148
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v168, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffC+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck], v168     // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDC clip if OOB. offset
buffer_load_short_d16 v169, v168, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v168, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffD+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck+1], v168   // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,2) */
s_mov_b32 s55, 152
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v170, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffC+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck], v170     // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDC clip if OOB. offset
buffer_load_short_d16 v171, v170, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v170, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffD+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck+1], v170   // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,3) */
s_mov_b32 s55, 156
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v172, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffC+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck], v172     // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDC clip if OOB. offset
buffer_load_short_d16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffD+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck+1], v172   // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 0, 5, 0), (0, 0, 5, 1), (0, 0, 5, 2), (0, 0, 5, 3), (0, 0, 5, 4), (0, 0, 5, 5), (0, 0, 5, 6), (0, 0, 5, 7), (0, 1, 5, 0), (0, 1, 5, 1), (0, 1, 5, 2), (0, 1, 5, 3), (0, 1, 5, 4), (0, 1, 5, 5), (0, 1, 5, 6), (0, 1, 5, 7), (0, 0, 6, 0), (0, 0, 6, 1), (0, 0, 6, 2), (0, 0, 6, 3)] */
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+80] // *= alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+81] // *= alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+82] // *= alpha
v_mul_f32 v[vgprValuC+83], s[sgprAlpha], v[vgprValuC+83] // *= alpha
v_mul_f32 v[vgprValuC+84], s[sgprAlpha], v[vgprValuC+84] // *= alpha
v_mul_f32 v[vgprValuC+85], s[sgprAlpha], v[vgprValuC+85] // *= alpha
v_mul_f32 v[vgprValuC+86], s[sgprAlpha], v[vgprValuC+86] // *= alpha
v_mul_f32 v[vgprValuC+87], s[sgprAlpha], v[vgprValuC+87] // *= alpha
v_mul_f32 v[vgprValuC+88], s[sgprAlpha], v[vgprValuC+88] // *= alpha
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+89] // *= alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+90] // *= alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+91] // *= alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+92] // *= alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+93] // *= alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+94] // *= alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+95] // *= alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+96] // *= alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+97] // *= alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+98] // *= alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+99] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

v_lshlrev_b32 v209, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+80], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+80     // convert C to bf16
buffer_store_short v80, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+81], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+81     // convert C to bf16
buffer_store_short v81, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+82], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+82     // convert C to bf16
buffer_store_short v82, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+83], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+83     // convert C to bf16
buffer_store_short v83, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+84], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+84     // convert C to bf16
buffer_store_short v84, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+85], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+85     // convert C to bf16
buffer_store_short v85, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+86], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+86     // convert C to bf16
buffer_store_short v86, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+87], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+87     // convert C to bf16
buffer_store_short v87, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+88], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+88     // convert C to bf16
buffer_store_short v88, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+89], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+89     // convert C to bf16
buffer_store_short v89, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+90], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+90     // convert C to bf16
buffer_store_short v90, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+91], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+91     // convert C to bf16
buffer_store_short v91, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+92], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+92     // convert C to bf16
buffer_store_short v92, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+93], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+93     // convert C to bf16
buffer_store_short v93, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+94], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+94     // convert C to bf16
buffer_store_short v94, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+95], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+95     // convert C to bf16
buffer_store_short v95, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+96], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+96     // convert C to bf16
buffer_store_short v96, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+97], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+97     // convert C to bf16
buffer_store_short v97, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+98], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+98     // convert C to bf16
buffer_store_short v98, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+99], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+99     // convert C to bf16
buffer_store_short v99, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #5 (d1,d0,vc1,vc0) = */
/*    (0,0,6,4:vw1); (0,0,6,5:vw1); (0,0,6,6:vw1); (0,0,6,7:vw1); (0,1,6,0:vw1); (0,1,6,1:vw1); (0,1,6,2:vw1); (0,1,6,3:vw1); (0,1,6,4:vw1); (0,1,6,5:vw1); (0,1,6,6:vw1); (0,1,6,7:vw1); (0,0,7,0:vw1); (0,0,7,1:vw1); (0,0,7,2:vw1); (0,0,7,3:vw1); (0,0,7,4:vw1); (0,0,7,5:vw1); (0,0,7,6:vw1); (0,0,7,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,6,0,4) */
s_mov_b32 s55, 144
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck], v134     // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,5) */
s_mov_b32 s55, 148
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v136, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffC+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck], v136     // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDC clip if OOB. offset
buffer_load_short_d16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,6) */
s_mov_b32 s55, 152
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v138, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffC+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck], v138     // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDC clip if OOB. offset
buffer_load_short_d16 v139, v138, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,7) */
s_mov_b32 s55, 156
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v140, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffC+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck], v140     // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v141, v140, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,0) */
s_mov_b32 s55, 144
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck], v142     // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,1) */
s_mov_b32 s55, 148
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffC+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck], v144     // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v145, v144, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,2) */
s_mov_b32 s55, 152
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v146, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffC+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck], v146     // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDC clip if OOB. offset
buffer_load_short_d16 v147, v146, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,3) */
s_mov_b32 s55, 156
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v148, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffC+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck], v148     // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDC clip if OOB. offset
buffer_load_short_d16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,4) */
s_mov_b32 s54, 80
s_mov_b32 s55, 144
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v150, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck], v150     // 
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDC clip if OOB. offset
buffer_load_short_d16 v151, v150, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck+1], v150   // 
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,5) */
s_mov_b32 s54, 80
s_mov_b32 s55, 148
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v152, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffC+0], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck], v152     // 
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDC clip if OOB. offset
buffer_load_short_d16 v153, v152, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+0], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck+1], v152   // 
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,6) */
s_mov_b32 s54, 80
s_mov_b32 s55, 152
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v154, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffC+1], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck], v154     // 
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDC clip if OOB. offset
buffer_load_short_d16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+1], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck+1], v154   // 
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,7) */
s_mov_b32 s54, 80
s_mov_b32 s55, 156
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v156, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffC+2], v156  // add 4* loop offset
v_add_co_u32 v156, vcc, s[sgprEdgeCheck], v156     // 
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDC clip if OOB. offset
buffer_load_short_d16 v157, v156, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v156, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffD+2], v156  // add 4* loop offset
v_add_co_u32 v156, vcc, s[sgprEdgeCheck+1], v156   // 
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
s_mov_b32 s54, 128
s_mov_b32 s55, 144
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v158, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v158, vcc, s[sgprEdgeCheck], v158     // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDC clip if OOB. offset
buffer_load_short_d16 v159, v158, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v158, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v158, vcc, s[sgprEdgeCheck+1], v158   // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,1) */
s_mov_b32 s54, 128
s_mov_b32 s55, 148
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v160, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffC+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck], v160     // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDC clip if OOB. offset
buffer_load_short_d16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffD+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck+1], v160   // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,2) */
s_mov_b32 s54, 128
s_mov_b32 s55, 152
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v162, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffC+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck], v162     // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDC clip if OOB. offset
buffer_load_short_d16 v163, v162, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v162, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffD+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck+1], v162   // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,3) */
s_mov_b32 s54, 128
s_mov_b32 s55, 156
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v164, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffC+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck], v164     // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDC clip if OOB. offset
buffer_load_short_d16 v165, v164, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffD+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck+1], v164   // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,4) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 144
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v166, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v166, vcc, s[sgprEdgeCheck], v166     // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDC clip if OOB. offset
buffer_load_short_d16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v166, vcc, s[sgprEdgeCheck+1], v166   // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,5) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 148
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v168, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffC+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck], v168     // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDC clip if OOB. offset
buffer_load_short_d16 v169, v168, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v168, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffD+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck+1], v168   // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,6) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 152
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v170, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffC+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck], v170     // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDC clip if OOB. offset
buffer_load_short_d16 v171, v170, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v170, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffD+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck+1], v170   // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,7) */
s_mov_b32 s54, 144                                  // coordOffset0 d0=0 vc0=4
s_mov_b32 s55, 156
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v172, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffC+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck], v172     // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDC clip if OOB. offset
buffer_load_short_d16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffD+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck+1], v172   // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 0, 6, 4), (0, 0, 6, 5), (0, 0, 6, 6), (0, 0, 6, 7), (0, 1, 6, 0), (0, 1, 6, 1), (0, 1, 6, 2), (0, 1, 6, 3), (0, 1, 6, 4), (0, 1, 6, 5), (0, 1, 6, 6), (0, 1, 6, 7), (0, 0, 7, 0), (0, 0, 7, 1), (0, 0, 7, 2), (0, 0, 7, 3), (0, 0, 7, 4), (0, 0, 7, 5), (0, 0, 7, 6), (0, 0, 7, 7)] */
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+100] // *= alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+101] // *= alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+102] // *= alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+103] // *= alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+104] // *= alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+105] // *= alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+106] // *= alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+107] // *= alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+108] // *= alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+109] // *= alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+110] // *= alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+111] // *= alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+112] // *= alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+113] // *= alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+114] // *= alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+115] // *= alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+116] // *= alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+117] // *= alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+118] // *= alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+119] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

v_lshlrev_b32 v209, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+100], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+100     // convert C to bf16
buffer_store_short v100, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+101], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+101     // convert C to bf16
buffer_store_short v101, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+102], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+102     // convert C to bf16
buffer_store_short v102, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+103], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+103     // convert C to bf16
buffer_store_short v103, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+104], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+104     // convert C to bf16
buffer_store_short v104, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+105], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+105     // convert C to bf16
buffer_store_short v105, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+106], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+106     // convert C to bf16
buffer_store_short v106, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+107], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+107     // convert C to bf16
buffer_store_short v107, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+108], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+108     // convert C to bf16
buffer_store_short v108, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+109], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+109     // convert C to bf16
buffer_store_short v109, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+110], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+110     // convert C to bf16
buffer_store_short v110, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+111], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+111     // convert C to bf16
buffer_store_short v111, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+112], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+112     // convert C to bf16
buffer_store_short v112, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+113], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+113     // convert C to bf16
buffer_store_short v113, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+114], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+114     // convert C to bf16
buffer_store_short v114, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+115], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+115     // convert C to bf16
buffer_store_short v115, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+116], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+116     // convert C to bf16
buffer_store_short v116, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+117], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+117     // convert C to bf16
buffer_store_short v117, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+118], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+118     // convert C to bf16
buffer_store_short v118, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+119], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+119     // convert C to bf16
buffer_store_short v119, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #6 (d1,d0,vc1,vc0) = */
/*    (0,1,7,0:vw1); (0,1,7,1:vw1); (0,1,7,2:vw1); (0,1,7,3:vw1); (0,1,7,4:vw1); (0,1,7,5:vw1); (0,1,7,6:vw1); (0,1,7,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,7,1,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=1 vc0=0
s_mov_b32 s55, 144
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck], v134     // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,1) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=1 vc0=0
s_mov_b32 s55, 148
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v136, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffC+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck], v136     // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDC clip if OOB. offset
buffer_load_short_d16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,2) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=1 vc0=0
s_mov_b32 s55, 152
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v138, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffC+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck], v138     // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDC clip if OOB. offset
buffer_load_short_d16 v139, v138, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,3) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=1 vc0=0
s_mov_b32 s55, 156
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v140, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffC+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck], v140     // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v141, v140, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,4) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
s_mov_b32 s55, 144
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck], v142     // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,5) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
s_mov_b32 s55, 148
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffC+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck], v144     // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v145, v144, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,6) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
s_mov_b32 s55, 152
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v146, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffC+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck], v146     // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDC clip if OOB. offset
buffer_load_short_d16 v147, v146, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,7) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=1 vc0=4
s_mov_b32 s55, 156
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, s55                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v148, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffC+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck], v148     // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDC clip if OOB. offset
buffer_load_short_d16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 1, 7, 0), (0, 1, 7, 1), (0, 1, 7, 2), (0, 1, 7, 3), (0, 1, 7, 4), (0, 1, 7, 5), (0, 1, 7, 6), (0, 1, 7, 7)] */
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+120] // *= alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+121] // *= alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+122] // *= alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+123] // *= alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+124] // *= alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+125] // *= alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+126] // *= alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+127] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
v_mov_b32 v206, 0x7fff0000                         // fp32 Nan
v_mov_b32 v207, 0x7fff                             // rounding bias for bfloat16

v_lshlrev_b32 v209, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+120], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+120     // convert C to bf16
buffer_store_short v120, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+121], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+121     // convert C to bf16
buffer_store_short v121, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+122], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+122     // convert C to bf16
buffer_store_short v122, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+123], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+123     // convert C to bf16
buffer_store_short v123, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+124], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+124     // convert C to bf16
buffer_store_short v124, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+125], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+125     // convert C to bf16
buffer_store_short v125, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+126], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+126     // convert C to bf16
buffer_store_short v126, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v209, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+127], v209, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+127     // convert C to bf16
buffer_store_short v127, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_branch label_GW_End_109                          // jump to end
label_GW_End_109:

label_0111:  /// KernelEnd
/* The 0x60 workspace starts with problem_count * ceil(N/256) claim words;
 * num_done_groups logical completion counters follow them.  Wait for every
 * final-output store from this 512-thread CTA before lane zero contributes
 * exactly one completion for its claimed (problem, physical-N-tile) task. */
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
s_lshr_b32 s80, s80, 8                            // physical tiles per logical group
s_cmp_eq_u32 s80, 0
s_cbranch_scc1 .L_nn_moe_nchunk_signal_restore_exec
s_cmp_eq_u32 s83, 0
s_cbranch_scc1 .L_nn_moe_nchunk_signal_restore_exec

s_add_u32 s82, s82, 255
s_lshr_b32 s82, s82, 8                            // total physical N tiles
s_mul_i32 s89, s81, s82                           // claim words before counters
s_lshl_b32 s89, s89, 2
s_add_u32 s90, s90, s89                           // logical counter array base
s_addc_u32 s91, s91, 0

s_mov_b32 s84, 0                                  // logical group id
s_mov_b32 s85, s100                               // preserved physical N tile
.L_nn_moe_nchunk_group_div:
s_cmp_lt_u32 s85, s80
s_cbranch_scc1 .L_nn_moe_nchunk_group_div_done
s_sub_u32 s85, s85, s80
s_add_u32 s84, s84, 1
s_branch .L_nn_moe_nchunk_group_div
.L_nn_moe_nchunk_group_div_done:
s_cmp_ge_u32 s84, s83
s_cbranch_scc1 .L_nn_moe_nchunk_signal_restore_exec

/* The last logical group may contain fewer than chunk_size_n/256 tiles. */
s_mul_i32 s87, s84, s80                           // first physical tile in group
s_sub_u32 s88, s82, s87                           // physical tiles remaining
s_cmp_lt_u32 s88, s80
s_cselect_b32 s87, s88, s80                       // actual physical tiles in group
s_mul_i32 s88, s81, s87                           // expected task completions

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

/* Publish only after all BF16 stores are globally ordered before the final
 * counter increment observed above. */
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
s_endpgm                                           // Kernel End
