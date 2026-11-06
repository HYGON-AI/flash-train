/******************************************/
/* Flux MoE AG MT256 asm family template */
/* variant: contract_rank_tile_x64_v36_dynamic_direct_output_nchunk_signal */
/* tile: MT256x256x64 */
/* status: source-built v36 dynamic scheduler with direct final output,
 * CTA-cached final-row bases, and logical N-group completion signaling */
/* default Flux route is untouched */
/******************************************/

.ifndef FLUX_TNLOADER_KN_DIRECT_PRODUCER8
.set FLUX_TNLOADER_KN_DIRECT_PRODUCER8, 0
.endif
.ifndef FLUX_TNLOADER_KN_DIRECT_LDS
.set FLUX_TNLOADER_KN_DIRECT_LDS, 0
.endif
.ifndef FLUX_TNLOADER_KN_RAW_USE_TN
.set FLUX_TNLOADER_KN_RAW_USE_TN, 0
.endif
.ifndef FLUX_TNLOADER_KN_RAW_ALT
.set FLUX_TNLOADER_KN_RAW_ALT, 0
.endif
.ifndef FLUX_TNLOADER_KN_PRODUCER4
.set FLUX_TNLOADER_KN_PRODUCER4, 0
.endif
.ifndef FLUX_TNLOADER_KN_DIRECT_LDS_AILKMAP
.set FLUX_TNLOADER_KN_DIRECT_LDS_AILKMAP, 0
.endif
.ifndef FLUX_TNLOADER_KN_DIRECT_LDS_AILKPROD
.set FLUX_TNLOADER_KN_DIRECT_LDS_AILKPROD, 0
.endif
.ifndef FLUX_TNLOADER_KN_DIRECT_LDS_AILKSTAGE
.set FLUX_TNLOADER_KN_DIRECT_LDS_AILKSTAGE, 0
.endif
.ifndef FLUX_TNLOADER_KN_DIRECT_LDS_AILKPROD_RAW_SWAP
.set FLUX_TNLOADER_KN_DIRECT_LDS_AILKPROD_RAW_SWAP, 0
.endif
.ifndef FLUX_TNLOADER_KN_DIRECT_LDS_ROWMAJOR
.set FLUX_TNLOADER_KN_DIRECT_LDS_ROWMAJOR, 0
.endif
.ifndef FLUX_TNLOADER_KN_DIRECT_LDS_STRUCT
.set FLUX_TNLOADER_KN_DIRECT_LDS_STRUCT, 0
.endif
.ifndef FLUX_TNLOADER_KN_DIRECT_LDS_STRUCT_SPLIT
.set FLUX_TNLOADER_KN_DIRECT_LDS_STRUCT_SPLIT, 0
.endif
.ifndef FLUX_TNLOADER_KN_DIRECT_LDS_VECSTORE
.set FLUX_TNLOADER_KN_DIRECT_LDS_VECSTORE, 0
.endif
.ifndef FLUX_TNLOADER_KN_DIRECT_LDS_ROW_M32
.set FLUX_TNLOADER_KN_DIRECT_LDS_ROW_M32, 0
.endif


/******************************************/
/* Function Prefix                        */
/******************************************/



/******************************************/
/* Begin Kernel                           */
/******************************************/

// Component.Signature.SignatureCOV3
.amdgcn_target "amdgcn-amd-amdhsa--gfx936"
.text
	.protected flux_ag_grouped_gemm_contract_rank_tile_x64_native_nk_a2a_direct_output_nchunk_signal
	.globl flux_ag_grouped_gemm_contract_rank_tile_x64_native_nk_a2a_direct_output_nchunk_signal
	.p2align 8
	.type flux_ag_grouped_gemm_contract_rank_tile_x64_native_nk_a2a_direct_output_nchunk_signal,@function
	.section .rodata,#alloc
	.p2align 6
	.amdhsa_kernel flux_ag_grouped_gemm_contract_rank_tile_x64_native_nk_a2a_direct_output_nchunk_signal
	  .amdhsa_user_sgpr_kernarg_segment_ptr 1
	  .amdhsa_next_free_vgpr 255 // vgprs
	  .amdhsa_next_free_sgpr 101 // sgprs; s100 preserves physical N-tile id
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
  - .name: flux_ag_grouped_gemm_contract_rank_tile_x64_native_nk_a2a_direct_output_nchunk_signal
    .symbol: 'flux_ag_grouped_gemm_contract_rank_tile_x64_native_nk_a2a_direct_output_nchunk_signal.kd'
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
    .max_flat_workgroup_size:    768
    .private_segment_fixed_size: 0
    .sgpr_count:                 101
    .sgpr_spill_count:           0
    .vgpr_count:                 255
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
flux_ag_grouped_gemm_contract_rank_tile_x64_native_nk_a2a_direct_output_nchunk_signal:

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
.set vgprValuB_X0_I0, 192
.set vgprValuB_X1_I0, 208
.set vgprGlobalReadOffsetA, 224
.set vgprGlobalReadOffsetB, 232
.set vgprLocalWriteAddrA, 240
.set vgprLocalWriteAddrB, 241
.set vgprLocalReadAddrA, 242
.set vgprLocalReadAddrA_ori, 243
.set vgprLocalReadAddrASwap, 244
.set vgprLocalReadAddrASwap_ori, 245
.set vgprSerial,246
.set vgprSerial_forLdsWrapOptA, 247
.set vgprG2LA, 128
.set vgprG2LB, 160
.set vgprLocalReadAddrB, 248
.set vgprLocalReadAddrASwap1, 249
.set vgprGlobalReadOffsetB1, 250
.set vgprLocalReadAddrA_forLdsWrapTailLoop, 247
/* Num VGPR=218 */
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
.set sgprLocalWriteAddrA_ori, 80
.set sgprLoopforPf, 81
.set sgprShadowLimitB_forTailLoop, 82
.set sgprShadowLimitA_forTailLoop, 88
.set sgprSrdA_forTailLoop, 84
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
.set constStrideAL, 1
.set sgprStrideA0I, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1
.set sgprStrideoffC, sgprSrdA
.set sgprStrideoffD, sgprAddressD
.set sgprEdgeCheck, sgprAddressB

.set MT0, 256
.set MT1, 256
.set DepthU, 64
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
.macro GLOBAL_OFFSET_A vgprAddr:req vgprOffsetL:req vgprOffset0I:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideA0I], v[\vgprOffset0I] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/* Raw-KN direct-LDS read helper. */
.macro KN_RAW_A_READS page:req k0:req k1:req
.if FLUX_TNLOADER_KN_DIRECT_LDS
.if FLUX_TNLOADER_KN_DIRECT_LDS_ROW_M32
/* Row-major EKN consumer map: eight N32 subtiles for each K16 slab. */
.if \k0 == 0x0000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:\page+0x0000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:\page+0x0040
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:\page+0x0080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:\page+0x00c0
ds_read_m32x16_b16 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:\page+0x0100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:\page+0x0140
ds_read_m32x16_b16 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:\page+0x0180
ds_read_m32x16_b16 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA] offset:\page+0x01c0
ds_read_m32x16_b16 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA] offset:\page+0x4000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrA] offset:\page+0x4040
ds_read_m32x16_b16 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+40+3], v[vgprLocalReadAddrA] offset:\page+0x4080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+44+3], v[vgprLocalReadAddrA] offset:\page+0x40c0
ds_read_m32x16_b16 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+48+3], v[vgprLocalReadAddrA] offset:\page+0x4100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+52+3], v[vgprLocalReadAddrA] offset:\page+0x4140
ds_read_m32x16_b16 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+56+3], v[vgprLocalReadAddrA] offset:\page+0x4180
ds_read_m32x16_b16 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+60+3], v[vgprLocalReadAddrA] offset:\page+0x41c0
.else
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:\page+0x4000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:\page+0x4040
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:\page+0x4080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:\page+0x40c0
ds_read_m32x16_b16 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:\page+0x4100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:\page+0x4140
ds_read_m32x16_b16 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:\page+0x4180
ds_read_m32x16_b16 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA] offset:\page+0x41c0
ds_read_m32x16_b16 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA] offset:\page+0x2000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrA] offset:\page+0x2040
ds_read_m32x16_b16 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+40+3], v[vgprLocalReadAddrA] offset:\page+0x2080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+44+3], v[vgprLocalReadAddrA] offset:\page+0x20c0
ds_read_m32x16_b16 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+48+3], v[vgprLocalReadAddrA] offset:\page+0x2100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+52+3], v[vgprLocalReadAddrA] offset:\page+0x2140
ds_read_m32x16_b16 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+56+3], v[vgprLocalReadAddrA] offset:\page+0x2180
ds_read_m32x16_b16 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+60+3], v[vgprLocalReadAddrA] offset:\page+0x21c0
.endif
.elseif FLUX_TNLOADER_KN_DIRECT_LDS_AILKSTAGE
/* Four native Ailk K16 stages.  Each stage contributes four m32x16
 * fragments; the complete K64 tile is placed in the same 64 VGPR slots
 * consumed by the 640-MMAC body. */
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x0000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:0x0080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:0x0100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:0x0180
ds_read_m32x16_b16 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:0x4000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:0x4080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:0x4100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrASwap] offset:0x4180
ds_read_m32x16_b16 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA] offset:0x8000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrA] offset:0x8080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+40+3], v[vgprLocalReadAddrA] offset:0x8100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+44+3], v[vgprLocalReadAddrASwap] offset:0x8180
ds_read_m32x16_b16 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+48+3], v[vgprLocalReadAddrA] offset:0xc000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+52+3], v[vgprLocalReadAddrA] offset:0xc080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+56+3], v[vgprLocalReadAddrA] offset:0xc100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+60+3], v[vgprLocalReadAddrASwap] offset:0xc180
.else
.if FLUX_TNLOADER_KN_DIRECT_LDS_AILKMAP
.if \k0 == 0x0000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:\page+0x0000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:\page+0x0080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:\page+0x0100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:\page+0x0180
ds_read_m32x16_b16 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:\page+0x2000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:\page+0x2080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:\page+0x2100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrASwap] offset:\page+0x2180
ds_read_m32x16_b16 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA] offset:\page+0x4000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrA] offset:\page+0x4080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+40+3], v[vgprLocalReadAddrA] offset:\page+0x4100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+44+3], v[vgprLocalReadAddrASwap] offset:\page+0x4180
ds_read_m32x16_b16 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+48+3], v[vgprLocalReadAddrA] offset:\page+0x6000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+52+3], v[vgprLocalReadAddrA] offset:\page+0x6080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+56+3], v[vgprLocalReadAddrA] offset:\page+0x6100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+60+3], v[vgprLocalReadAddrASwap] offset:\page+0x6180
.else
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:\page+0x0000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:\page+0x0080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:\page+0x0100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:\page+0x0180
ds_read_m32x16_b16 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:\page+0x2000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:\page+0x2080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:\page+0x2100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrASwap] offset:\page+0x2180
ds_read_m32x16_b16 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA] offset:\page+0x4000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrA] offset:\page+0x4080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+40+3], v[vgprLocalReadAddrA] offset:\page+0x4100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+44+3], v[vgprLocalReadAddrASwap] offset:\page+0x4180
ds_read_m32x16_b16 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+48+3], v[vgprLocalReadAddrA] offset:\page+0x6000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+52+3], v[vgprLocalReadAddrA] offset:\page+0x6080
ds_read_m32x16_b16 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+56+3], v[vgprLocalReadAddrA] offset:\page+0x6100
ds_read_m32x16_b16 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+60+3], v[vgprLocalReadAddrASwap] offset:\page+0x6180
.endif
.else
.if FLUX_TNLOADER_KN_DIRECT_LDS_AILKPROD_RAW_SWAP
/* Candidate-only raw immediate map with the Ailk swap base on k1. */
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x0000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrASwap] offset:\page+\k1+0x0000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x1000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrASwap] offset:\page+\k1+0x1000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x2000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrASwap] offset:\page+\k1+0x2000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x3000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrASwap] offset:\page+\k1+0x3000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x4000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrASwap] offset:\page+\k1+0x4000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+40+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x5000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+44+3], v[vgprLocalReadAddrASwap] offset:\page+\k1+0x5000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+48+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x6000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+52+3], v[vgprLocalReadAddrASwap] offset:\page+\k1+0x6000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+56+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x7000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+60+3], v[vgprLocalReadAddrASwap] offset:\page+\k1+0x7000
.elseif FLUX_TNLOADER_KN_RAW_ALT
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x0000
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x0000
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x1000
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x1000
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x2000
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x2000
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x3000
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x3000
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x4000
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x4000
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+40+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x5000
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+44+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x5000
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+48+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x6000
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+52+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x6000
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+56+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x7000
ds_read_m32x16_b16_alt v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+60+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x7000
.else
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x0000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x0000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x1000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x1000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x2000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x2000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x3000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x3000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+32+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x4000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+36+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x4000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+40+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x5000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+44+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x5000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+48+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x6000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+52+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x6000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+56+3], v[vgprLocalReadAddrA] offset:\page+\k0+0x7000
ds_read_m32x16_b16 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+60+3], v[vgprLocalReadAddrA] offset:\page+\k1+0x7000
.endif
.endif
s_waitcnt lgkmcnt(0)
.endif
.endif
.endm

.macro GLOBAL_STRUCT_OFFSET_B vgprAddr:req vgprOffset:req vgprIndex:req // 
v_cmp_lt_u32 vcc, v[\vgprIndex], s81               // row < tile_rows
_v_add_u32 v[\vgprAddr+0], s63, v[\vgprIndex]      // sorted_start + row
v_lshlrev_b32 v[\vgprAddr+0], 0x2, v[\vgprAddr+0]  // int32 byte offset
global_load_dword v[\vgprAddr+0], v[\vgprAddr+0], s[68:69] glc
s_waitcnt vmcnt(0)
v_cndmask_b32 v[\vgprAddr+0], 0, v[\vgprAddr+0], vcc // token or zero for padding
v_mul_lo_u32 v[\vgprAddr+0], s[sgprStrideB1J], v[\vgprAddr+0]
_v_add_u32 v[\vgprAddr+0], v[\vgprAddr+0], v[\vgprOffset]
_v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]
v_lshlrev_b32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]
.endm
.macro GLOBAL_STRUCT_INDEX_B vgprAddr:req vgprIndex:req // 
v_mov_b32 v[\vgprAddr+0], 0
.endm

/* Global Offset B */
.macro GLOBAL_ROWMAP_D_ADDR_REUSE vgprAddr:req, vgprCoord0:req, sgprMask:req, vgprRow:req // reuse cached direct-output row map
	v_cmp_ge_i32 vcc, v[\vgprRow], 0                  // cached row base is valid
	s_and_b64 s[\sgprMask:\sgprMask+1], s[\sgprMask:\sgprMask+1], vcc
	_v_add_lshl_u32 v[\vgprAddr], v[\vgprRow], v[\vgprCoord0], 0x1
.endm
.macro BUFFER_STORE_SHORT_D_CLIPPED vgprVal:req, vgprAddr:req, sgprMask:req // direct-output clipped OOB D store
	buffer_store_short v[\vgprVal], v[\vgprAddr], s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0
.endm

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


.macro TN_MAC_32x4x2_X0_I0
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
s_setprio 1 // Raise priority while processing macs
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
s_setprio 0 // Reset priority after macs 
.endm

.macro TN_MAC_32x4x2_X0_I1
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
s_setprio 1 // Raise priority while processing macs
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
s_setprio 0 // Reset priority after macs 
.endm
.macro TN_MAC_32x4x2_X1_I0
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
s_setprio 1 // Raise priority while processing macs
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
s_setprio 0 // Reset priority after macs 
.endm
.macro TN_MAC_32x4x2_X1_I1
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
s_setprio 1 // Raise priority while processing macs
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
s_setprio 0 // Reset priority after macs 
.endm


.macro KN_MAC_32x4x2_X0_I0
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
s_setprio 1 // Raise priority while processing macs
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
s_setprio 0 // Reset priority after macs 
.endm

.macro KN_MAC_32x4x2_X0_I1
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
s_setprio 1 // Raise priority while processing macs
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
s_setprio 0 // Reset priority after macs 
.endm

.macro KN_MAC_32x4x2_X1_I0
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
s_setprio 1 // Raise priority while processing macs
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
s_setprio 0 // Reset priority after macs 
.endm

.macro KN_MAC_32x4x2_X1_I1
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
s_setprio 1 // Raise priority while processing macs
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+0: vgprValuC+0+1: vgprValuC+0+2: vgprValuC+0+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+4: vgprValuC+4+1: vgprValuC+4+2: vgprValuC+4+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+8: vgprValuC+8+1: vgprValuC+8+2: vgprValuC+8+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+12: vgprValuC+12+1: vgprValuC+12+2: vgprValuC+12+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+16: vgprValuC+16+1: vgprValuC+16+2: vgprValuC+16+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+20: vgprValuC+20+1: vgprValuC+20+2: vgprValuC+20+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+24: vgprValuC+24+1: vgprValuC+24+2: vgprValuC+24+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+28: vgprValuC+28+1: vgprValuC+28+2: vgprValuC+28+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+32: vgprValuC+32+1: vgprValuC+32+2: vgprValuC+32+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+36: vgprValuC+36+1: vgprValuC+36+2: vgprValuC+36+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+40: vgprValuC+40+1: vgprValuC+40+2: vgprValuC+40+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+44: vgprValuC+44+1: vgprValuC+44+2: vgprValuC+44+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+48: vgprValuC+48+1: vgprValuC+48+2: vgprValuC+48+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+52: vgprValuC+52+1: vgprValuC+52+2: vgprValuC+52+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+56: vgprValuC+56+1: vgprValuC+56+2: vgprValuC+56+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+60: vgprValuC+60+1: vgprValuC+60+2: vgprValuC+60+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+64: vgprValuC+64+1: vgprValuC+64+2: vgprValuC+64+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+68: vgprValuC+68+1: vgprValuC+68+2: vgprValuC+68+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+72: vgprValuC+72+1: vgprValuC+72+2: vgprValuC+72+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+76: vgprValuC+76+1: vgprValuC+76+2: vgprValuC+76+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+80: vgprValuC+80+1: vgprValuC+80+2: vgprValuC+80+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+84: vgprValuC+84+1: vgprValuC+84+2: vgprValuC+84+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+88: vgprValuC+88+1: vgprValuC+88+2: vgprValuC+88+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+92: vgprValuC+92+1: vgprValuC+92+2: vgprValuC+92+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+96: vgprValuC+96+1: vgprValuC+96+2: vgprValuC+96+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+100: vgprValuC+100+1: vgprValuC+100+2: vgprValuC+100+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+104: vgprValuC+104+1: vgprValuC+104+2: vgprValuC+104+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+108: vgprValuC+108+1: vgprValuC+108+2: vgprValuC+108+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+112: vgprValuC+112+1: vgprValuC+112+2: vgprValuC+112+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+116: vgprValuC+116+1: vgprValuC+116+2: vgprValuC+116+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+120: vgprValuC+120+1: vgprValuC+120+2: vgprValuC+120+3]
v_mmac_f32_16x16x16_bf16 v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+124: vgprValuC+124+1: vgprValuC+124+2: vgprValuC+124+3]
s_setprio 0 // Reset priority after macs 
.endm

/* Dispatch to the original TN MFMA schedule or the raw-KN register mapping. */
.macro MAC_32x4x2_X0_I0
.if FLUX_TNLOADER_KN_DIRECT_LDS
.if FLUX_TNLOADER_KN_RAW_USE_TN
TN_MAC_32x4x2_X0_I0
.else
KN_MAC_32x4x2_X0_I0
.endif
.else
TN_MAC_32x4x2_X0_I0
.endif
.endm

.macro MAC_32x4x2_X0_I1
.if FLUX_TNLOADER_KN_DIRECT_LDS
.if FLUX_TNLOADER_KN_RAW_USE_TN
TN_MAC_32x4x2_X0_I1
.else
KN_MAC_32x4x2_X0_I1
.endif
.else
TN_MAC_32x4x2_X0_I1
.endif
.endm

.macro MAC_32x4x2_X1_I0
.if FLUX_TNLOADER_KN_DIRECT_LDS
.if FLUX_TNLOADER_KN_RAW_USE_TN
TN_MAC_32x4x2_X1_I0
.else
KN_MAC_32x4x2_X1_I0
.endif
.else
TN_MAC_32x4x2_X1_I0
.endif
.endm

.macro MAC_32x4x2_X1_I1
.if FLUX_TNLOADER_KN_DIRECT_LDS
.if FLUX_TNLOADER_KN_RAW_USE_TN
TN_MAC_32x4x2_X1_I1
.else
KN_MAC_32x4x2_X1_I1
.endif
.else
TN_MAC_32x4x2_X1_I1
.endif
.endm


.macro FP32_TO_BF16 vgprValue:req
	v_cmp_u_f32 s[80:81], v[\vgprValue], v[\vgprValue] // check Nan
	v_bfe_u32 v249, v[\vgprValue], 16, 1               // Non-Nan case: store lsb of bf16
	v_add3_u32 v249, v[\vgprValue], v249, v251           // Non-Nan case: add lsb and the increment for rounding
	v_cndmask_b32 v[\vgprValue], v249, v250, s[80:81]   // 
	v_lshrrev_b32 v[\vgprValue], 16, v[\vgprValue]   // convert C to bf16
.endm


/******************************************/
/* Allocate Resources                     */
/******************************************/

s_mov_b32 m0, 0x10000                              // LDS clamp at 65536 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

/* Load 112-byte AG grouped-GEMM contract args.
 *   0x00 input_buffer        -> imported B / packed rows
 *   0x08 weight              -> imported A / [expert][n][k]
 *   0x10 sorted_gather_index -> rank-tile sorted_expert_offsets
 *   0x18 output              -> imported D
 *   0x20 barrier_ptr
 *   0x28 problem_info        -> rank-tile tile_info SoA
 *   0x30 problem_count       -> rank tile count
 *   0x34 n_dim
 *   0x38 k_dim
 *   0x3c topk, positive routing width (row mapping is host-provided)
 *   0x40 rank
 *   0x44 wait_signal_value
 *   0x48 epilogue_mode, must be 1 for direct final output
 *   0x4c logical chunk_size_n, a multiple of 256
 *   0x50 logical done-group count
 *   0x54 launch mode, 1 selects logical N-group-major ordering
 *   0x58 sorted_token_ids final-row map
 *   0x60 task/control workspace: claims followed by done counters
 *   0x68 logical done flags
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
.ifdef FLUX_TNLOADER_KN_DIRECT
/* The TN-loader KN sibling consumes physical [E,K_local,N]. */
s_mov_b32 s[sgprStrideA0I], s[sgprStrideD1J]
s_mov_b32 s[sgprStrideAK], s[sgprStrideD1J]
.else
s_mov_b32 s[sgprStrideA0I], s[sgprSizeL]
s_mov_b32 s[sgprStrideAK], s[sgprSizeL]
.endif
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

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // Alpha == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if alpha != 0
s_mov_b32 s[sgprSizesSum+0], 0x0                   // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:

v_lshrrev_b32 v[vgprValuA_X0_I0], 6, v[vgprSerial]
v_readfirstlane_b32 s[sgprWaveiD], v[vgprValuA_X0_I0]
s_mov_b32 s[sgprMask], 0x10000

/******************************************/
/* Local Read Addresses                   */
/******************************************/


/* local read addresses: tile assignments a/b */

.if FLUX_TNLOADER_KN_DIRECT_LDS
/* Raw row-major [K64,N32] page: m32x16 uses a 16x32 K-slice and the
 * per-lane address is the linear row-major element offset. */
.if FLUX_TNLOADER_KN_DIRECT_LDS_AILKMAP
/* Ailk/Tensile lane-fragment mapping for ds_read_m32x16_b16.  This is an
 * isolated consumer-side experiment; the direct producer remains unchanged
 * so any correctness change is attributable to the read map. */
v_and_b32 v0, 63, v[vgprSerial]
v_lshrrev_b32 v4, 6, v[vgprSerial]
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
v_mov_b32 v7, 0x200
v_cndmask_b32 v7, 0, v7, s[62:63]
v_sub_i32 v6, v6, v7
v_add_lshl_u32 v[vgprLocalReadAddrASwap], v6, v8, 0x1
.else
v_and_b32 v0, 63, v[vgprSerial]
v_lshrrev_b32 v6, 2, v0
v_lshlrev_b32 v6, 6, v6
v_and_b32 v7, 3, v0
v_lshlrev_b32 v7, 4, v7
v_add_co_u32 v6, vcc, v6, v7
v_mov_b32 v[vgprLocalReadAddrA], v6
v_mov_b32 v[vgprLocalReadAddrASwap], v6
.endif
v_mov_b32 v[vgprLocalReadAddrASwap1], v6
v_mov_b32 v[vgprLocalReadAddrA_forLdsWrapTailLoop], v6
.else
v_and_b32 v0, 63, v[vgprSerial]                    // v0 = v[vgprSerial] % 64
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // v4 = v[vgprSerial] / 64
v_and_b32 v5, 15, v0                               // v5 = v0 % 15
v_lshrrev_b32 v3, 4, v0                            // v3 = v0 / 16


/* local read addresses: final offsets a */

v_and_b32 v6, 15, v0
v_lshrrev_b32 v7, 4,v0

v_and_b32 v8, 3, v6
v_lshrrev_b32 v9, 2, v6

v_lshlrev_b32 v11, 4, v8
v_lshlrev_b32 v8, 9, v8
v_lshlrev_b32 v9, 6, v9

v_lshlrev_b32 v7, 3, v7

v_add_co_u32 v7, vcc, v7, v8
v_add_co_u32 v7, vcc, v7, v9
v_add_co_u32 v7, vcc, v7, v11

v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v7
v_sub_co_u32  v[vgprLocalReadAddrA_forLdsWrapTailLoop], vcc, v7, v11
v_lshlrev_b32 v[vgprLocalReadAddrA_forLdsWrapTailLoop], 1, v[vgprLocalReadAddrA_forLdsWrapTailLoop]

v_cmp_eq_u32 s[64:65], v0, 47
v_cmp_eq_u32 s[66:67], v0, 63
s_or_b64 s[64:65], s[64:65], s[66:67]

v_mov_b32 v8, 0x200
v_cndmask_b32 v8, 0, v8, s[64:65]

v_sub_i32 v9, v7, v8

v_lshlrev_b32 v[vgprLocalReadAddrASwap], 1, v9


v_cmp_eq_u32 s[64:65], v0, 45
v_cmp_eq_u32 s[66:67], v0, 61
s_or_b64 s[68:69], s[64:65], s[66:67]

v_and_b32 v6, 15, v0
v_cmp_eq_u32 s[64:65], v6, 15
v_cmp_eq_u32 s[66:67], v6, 14
s_or_b64 s[64:65], s[64:65], s[66:67]

s_or_b64 s[64:65], s[68:69], s[64:65]

v_mov_b32 v8, 0x200

v_cndmask_b32 v8, 0, v8, s[64:65]

v_sub_i32 v7, v7, v8

v_lshlrev_b32 v[vgprLocalReadAddrASwap1], 1, v7
.endif

/* local read addresses: final offsets b */


/* local read addresses: declare addresses a */

/* N/A */


/* local read addresses: declare addresses b */

v_mov_b32 v[vgprLocalReadAddrA_ori], v[vgprLocalReadAddrA]
v_mov_b32 v[vgprLocalReadAddrASwap_ori], v[vgprLocalReadAddrASwap]

/******************************************/
/* Begin setupNewTile, isPap=False         */
/******************************************/



/* global read addresses: work-group */

/* Flux AGGroupGemm rank-tile scheduler/prologue.
 * ABI overlay:
 *   tensor2dSizeC -> int32 sorted_gather_index
 *   dataD         -> packed/AG output base
 *   dataC         -> int32 tile_info SoA, fields [expert,sorted_start,seg_start,seg_end,count]
 *   dataA         -> all-expert TN weight, [expert][n_dim][k_dim]
 *   dataB         -> full input rows, [token][k_dim]
 *   sizeI         -> global n_dim before this block; replaced by local tile_n_count
 *   sizeK         -> rank tile count before this block; replaced by 1
 *   sizeL         -> k_dim
 *   strideD1J     -> global n_dim, also used for D row stride and weight expert stride
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
s_cmp_eq_u32 s77, 1
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
 * problem id to all 12 waves in this 768-thread CTA.
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

.L_v12_claim_scan:
/* Read the candidate's owner-rank interval from tile_info SoA fields 2/3. */
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
s_mov_b32 s83, s80
.L_v12_ready_owner_loop:
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
s_mov_b32 s93, s63                                // direct-output sorted_start shadow
v_readfirstlane_b32 s80, v3                       // segment_start
v_readfirstlane_b32 s82, v4                       // segment_end
v_readfirstlane_b32 s81, v5                       // tile_m_count
s_mov_b32 s97, s81                                // preserve rows for LDS row-map staging

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

/* packed output base = problem_id * 256.  B loads use sorted_start above to
 * gather token rows from full input through sorted_gather_index.
 */
s_waitcnt lgkmcnt(0)
s_lshl_b32 s65, s[sgprWorkGroup0], 8

/* Wait all owner ranks in [segment_start, segment_end], same contract as the
 * hand-written rank-tile kernel.  The intermediate ABI uses fixed signal 1.
 */
s_cmp_lt_i32 s80, 0
s_cbranch_scc1 .L_moe_after_wait
s_cmp_gt_i32 s80, s82
s_cbranch_scc1 .L_moe_after_wait
s_mov_b32 s83, s80
.L_moe_wait_owner_loop:
s_cmp_gt_i32 s83, s82
s_cbranch_scc1 .L_moe_after_wait
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
s_mov_b32 s94, s[sgprAddressC+0]                   // shadow rowmap ptr low before epilogue SGPR aliasing
s_mov_b32 s95, s[sgprAddressC+1]                   // shadow rowmap ptr high before epilogue SGPR aliasing

/* Rebase the task-local weight pointer. */
.ifdef FLUX_TNLOADER_KN_DIRECT
/* Physical [expert][k_dim][n_dim]: weight[expert, 0, tile_n_base]. */
s_mul_i32 s66, s62, s[sgprSizeL]
s_mul_hi_u32 s67, s66, s[sgprStrideD1J]
s_mul_i32 s66, s66, s[sgprStrideD1J]
s_add_u32 s66, s66, s64
s_addc_u32 s67, s67, 0
.else
/* Physical [expert][n_dim][k_dim]: weight[expert, tile_n_base, 0]. */
s_mul_i32 s66, s62, s[sgprStrideD1J]
s_add_u32 s66, s66, s64
s_mul_hi_u32 s67, s66, s[sgprSizeL]
s_mul_i32 s66, s66, s[sgprSizeL]
.endif
s_lshl_b64 s[66:67], s[66:67], 1
s_add_u32 s[sgprAddressA+0], s[sgprAddressA+0], s66
s_addc_u32 s[sgprAddressA+1], s[sgprAddressA+1], s67

/* B base stays at full_input; per-lane B offsets include token*k_dim. */

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

/* Tighten A/B tensor limits after rebasing pointers to the problem tile. */
.ifdef FLUX_TNLOADER_KN_DIRECT
/* KN A spans the remainder of this expert matrix from tile_n_base onward. */
s_mul_hi_u32 s[sgprTensor2dSizeA+1], s[sgprSizeL], s[sgprStrideD1J]
s_mul_i32 s[sgprTensor2dSizeA+0], s[sgprSizeL], s[sgprStrideD1J]
s_sub_u32 s[sgprTensor2dSizeA+0], s[sgprTensor2dSizeA+0], s64
s_subb_u32 s[sgprTensor2dSizeA+1], s[sgprTensor2dSizeA+1], 0
.else
s_mul_hi_u32 s[sgprTensor2dSizeA+1], s84, s[sgprSizeL]
s_mul_i32 s[sgprTensor2dSizeA+0], s84, s[sgprSizeL]
.endif
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

.ifdef FLUX_TNLOADER_KN_DIRECT
/* Four producer waves load physical KN weight.  A producer lane owns one
 * linear chunk of the raw [K,N] tile.  In the direct-LDS candidate, wave p
 * owns K rows [16p,16p+15] and lane q owns columns [8(lane%4),+7].  This
 * makes the implicit 16-byte-per-lane MUBUF LDS destination row-major.  The
 * software-scatter path keeps the historical lane map below.  Eight vectors
 * spaced by 32 columns cover the complete 256-column tile.
 */
.if FLUX_TNLOADER_KN_DIRECT_LDS
v_and_b32 v8, 63, v[vgprSerial]          // lane within producer wave
v_lshrrev_b32 v0, 6, v[vgprSerial]
v_and_b32 v0, 3, v0                      // producer wave in the four-wave set
v_lshrrev_b32 v8, 2, v8                  // K row within this 16-row wave slice
v_lshlrev_b32 v5, 4, v0
v_add_co_u32 v8, vcc, v8, v5             // K = 16*wave + lane/4
v_and_b32 v5, 3, v[vgprSerial]
v_lshlrev_b32 v17, 3, v5                 // N base within the 32-column block
.else
v_and_b32 v8, 63, v[vgprSerial]          // K coordinate in this DepthU=64 block
v_lshrrev_b32 v0, 6, v[vgprSerial]
v_and_b32 v0, 3, v0                      // producer wave in the four-wave set
v_lshlrev_b32 v4, 3, v0                  // local N base: 0, 8, 16, or 24
.endif
.else
/* LVCA = 4 */
/* v0 = (local)groA-tile = serial/LVCA (note (wgA*MTA) will be added to SRD) */
/* v1 = groA-unroll = serial%LVCA */
v_and_b32 v6, 255, v[vgprSerial]
v_lshrrev_b32 v0, 6, v6                  // v0 = v[vgprSerial] / 64
v_and_b32 v1, 63, v6                     // v1 = v[vgprSerial] % 64
/* gro-unroll *= glvw */

v_and_b32 v2, 7, v1
v_lshrrev_b32 v3, 3, v1

v_lshlrev_b32 v1, 3, v2

v_lshlrev_b32 v3, 2, v3
v_add_co_u32 v0, vcc, v0, v3
.endif

/* global read addresses: tile offset assignment b */

/* LVCB = 4 */
/* v2 = (local)groB-tile = serial/LVCB (note (wgB*MTB) will be added to SRD) */
/* v3 = groB-unroll = serial%LVCB */

v_and_b32 v4, 63, v[vgprSerial]
v_lshrrev_b32 v5, 6, v[vgprSerial]

v_lshrrev_b32 v3, 4, v4                  // v2 = v[vgprSerial] / 16
v_and_b32 v2, 15, v4                     // v3 = v[vgprSerial] % 16
/* gro-unroll *= glvw */
v_lshlrev_b32 v3, 0x3, v3                          // v3 = v3 * 8

v_lshlrev_b32 v5, 5, v5
v_add_co_u32 v2, vcc, v2, v5

/* global read addresses: unroll assignment a */

/* v1 */


/* global read addresses: unroll assignment b */

/* v3 */


/* global read addresses: other free assignments */

/* s[sgprWorkGroup2] */


/* global read addresses: tile offsets a */

.ifdef FLUX_TNLOADER_KN_DIRECT
.if FLUX_TNLOADER_KN_DIRECT_LDS
v_mov_b32 v4, v17                         // restore row-major N base
.else
v_lshlrev_b32 v4, 3, v0                  // restore N base after B setup
.endif
_v_add_co_u32 v5, vcc, 32, v4
_v_add_co_u32 v11, vcc, 32, v5
_v_add_co_u32 v12, vcc, 32, v11
_v_add_co_u32 v13, vcc, 32, v12
_v_add_co_u32 v14, vcc, 32, v13
_v_add_co_u32 v15, vcc, 32, v14
_v_add_co_u32 v16, vcc, 32, v15
.else
v_mov_b32 v4, v0                                   // groA0I_0
_v_add_co_u32 v5, vcc, 32, v4                     // groA0I_1 += LSPA
_v_add_co_u32 v11, vcc, 32, v5                     // groA0I_1 += LSPA
_v_add_co_u32 v12, vcc, 32, v11                     // groA0I_1 += LSPA
_v_add_co_u32 v13, vcc, 32, v12                     // groA0I_1 += LSPA
_v_add_co_u32 v14, vcc, 32, v13                     // groA0I_1 += LSPA
_v_add_co_u32 v15, vcc, 32, v14                     // groA0I_1 += LSPA
_v_add_co_u32 v16, vcc, 32, v15                     // groA0I_1 += LSPA
.endif

/* global read addresses: tile offsets b */

v_mov_b32 v6, v2                                  // groB1J_0
_v_add_co_u32 v7, vcc, 16, v6                     // groB1J_1 += LSPB

/* global read addresses: unroll offsets a */

.ifndef FLUX_TNLOADER_KN_DIRECT
v_mov_b32 v8, v1                                   // groAL_0
.endif


/* global read addresses: unroll offsets b */

v_mov_b32 v9, v3                                  // groBL_0
_v_add_co_u32 v18, vcc, 32, v9                    // groBL_1 + LSCB


/* global read addresses: final offsets a */

.ifdef FLUX_TNLOADER_KN_DIRECT
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  4,  8, 10
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+1,  5,  8, 10
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+2, 11,  8, 10
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+3, 12,  8, 10
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+4, 13,  8, 10
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+5, 14,  8, 10
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+6, 15,  8, 10
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+7, 16,  8, 10
.if FLUX_TNLOADER_KN_DIRECT_LDS_AILKSTAGE
/* Native Ailk lane contract for four K16 stages.  Each stage is a pair of
 * loads at K offsets s*16 and s*16+8; the eight addresses below are consumed
 * by the producer in stage order. */
v_and_b32 v180, 63, v[vgprSerial]
v_lshrrev_b32 v181, 5, v180
v_lshrrev_b32 v182, 6, v[vgprSerial]
v_sub_u32 v182, v182, 8
v_lshlrev_b32 v183, 1, v182
v_add_u32 v184, v183, v181
v_and_b32 v185, 3, v184
v_and_b32 v186, 0xfffffffc, v184
v_lshlrev_b32 v187, 1, v185
v_or_b32 v186, v186, v187
v_and_b32 v185, 2, v184
v_lshrrev_b32 v185, 1, v185
v_or_b32 v186, v186, v185
v_and_b32 v181, 31, v180
v_lshlrev_b32 v181, 3, v181
.macro KN_AILK_STAGE_OFFSET dst:req kadd:req
v_add_u32 v180, v186, \kadd
v_mul_lo_u32 v180, s[sgprStrideA0I], v180
v_add_u32 v180, v180, v181
v_add_u32 v180, 8, v180
v_lshlrev_b32 v[\dst], 1, v180
.endm
KN_AILK_STAGE_OFFSET vgprGlobalReadOffsetA+0, 0x0000
KN_AILK_STAGE_OFFSET vgprGlobalReadOffsetA+1, 0x0008
KN_AILK_STAGE_OFFSET vgprGlobalReadOffsetA+2, 0x0010
KN_AILK_STAGE_OFFSET vgprGlobalReadOffsetA+3, 0x0018
KN_AILK_STAGE_OFFSET vgprGlobalReadOffsetA+4, 0x0020
KN_AILK_STAGE_OFFSET vgprGlobalReadOffsetA+5, 0x0028
KN_AILK_STAGE_OFFSET vgprGlobalReadOffsetA+6, 0x0030
KN_AILK_STAGE_OFFSET vgprGlobalReadOffsetA+7, 0x0038
.elseif FLUX_TNLOADER_KN_DIRECT_LDS_AILKPROD
/* Keep the 640-MMAC body's eight N-strided vectors.  Only the K row's
 * low two bits are permuted using the Ailk map; vector j advances N by 32.
 * This matches the raw-KN loader's original lane contract. */
v_and_b32 v180, 63, v[vgprSerial]
v_lshrrev_b32 v181, 2, v180
v_lshrrev_b32 v182, 6, v[vgprSerial]
v_sub_u32 v182, v182, 8
v_lshlrev_b32 v183, 4, v182
v_add_u32 v184, v183, v181
v_and_b32 v185, 3, v184
v_and_b32 v186, 0xfffffffc, v184
v_lshlrev_b32 v187, 1, v185
v_or_b32 v186, v186, v187
v_and_b32 v185, 2, v184
v_lshrrev_b32 v185, 1, v185
v_or_b32 v186, v186, v185
v_and_b32 v181, 3, v180
v_lshlrev_b32 v181, 3, v181
.macro KN_AILK_GLOBAL_OFFSET dst:req kadd:req
v_mul_lo_u32 v180, s[sgprStrideA0I], v186
v_add_u32 v180, v180, v181
v_add_u32 v180, \kadd, v180
v_add_u32 v180, 8, v180
v_lshlrev_b32 v[\dst], 1, v180
.endm
KN_AILK_GLOBAL_OFFSET vgprGlobalReadOffsetA+0, 0x0000
KN_AILK_GLOBAL_OFFSET vgprGlobalReadOffsetA+1, 0x0020
KN_AILK_GLOBAL_OFFSET vgprGlobalReadOffsetA+2, 0x0040
KN_AILK_GLOBAL_OFFSET vgprGlobalReadOffsetA+3, 0x0060
KN_AILK_GLOBAL_OFFSET vgprGlobalReadOffsetA+4, 0x0080
KN_AILK_GLOBAL_OFFSET vgprGlobalReadOffsetA+5, 0x00a0
KN_AILK_GLOBAL_OFFSET vgprGlobalReadOffsetA+6, 0x00c0
KN_AILK_GLOBAL_OFFSET vgprGlobalReadOffsetA+7, 0x00e0
.endif
.else
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  8,  4, 10 // gROA_0_0_0_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+1,  8,  5, 10 // gROA_0_0_1_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+2,  8,  11, 10 // gROA_0_0_0_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+3,  8,  12, 10 // gROA_0_0_1_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+4,  8,  13, 10 // gROA_0_0_0_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+5,  8,  14, 10 // gROA_0_0_1_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+6,  8,  15, 10 // gROA_0_0_0_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+7,  8,  16, 10 // gROA_0_0_1_0
.endif

GLOBAL_OFFSET_B_TailLoop vgprGlobalReadOffsetB1+0,  9,  6, 10 // gROB_0_0_0_0
GLOBAL_OFFSET_B_TailLoop vgprGlobalReadOffsetB1+1,  18, 6, 10 // gROB_0_0_1_0
GLOBAL_OFFSET_B_TailLoop vgprGlobalReadOffsetB1+2,  9,  7, 10 // gROB_0_0_0_0
GLOBAL_OFFSET_B_TailLoop vgprGlobalReadOffsetB1+3,  18, 7, 10 // gROB_0_0_1_0


/* global read addresses: final offsets b */
// s_mov_b32 s[sgprStructBit], 12
// s_lshl_b32 s[sgprStrideSwizzle], 0x1, s[sgprStructBit]

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
GLOBAL_STRUCT_INDEX_B vgprGlobalReadOffsetB+0, 6
GLOBAL_STRUCT_OFFSET_B vgprGlobalReadOffsetB+1, 9, 6
GLOBAL_STRUCT_INDEX_B vgprGlobalReadOffsetB+2, 6
GLOBAL_STRUCT_OFFSET_B vgprGlobalReadOffsetB+3, 18, 6
GLOBAL_STRUCT_INDEX_B vgprGlobalReadOffsetB+4, 7
GLOBAL_STRUCT_OFFSET_B vgprGlobalReadOffsetB+5, 9, 7
GLOBAL_STRUCT_INDEX_B vgprGlobalReadOffsetB+6, 7
GLOBAL_STRUCT_OFFSET_B vgprGlobalReadOffsetB+7, 18, 7


//GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  9,  6, 10 // gROB_0_0_0_0
//GLOBAL_OFFSET_B vgprGlobalReadOffsetB+1,  18,  6, 10 // gROB_0_0_1_0
//GLOBAL_OFFSET_B vgprGlobalReadOffsetB+2,  9,  7, 10 // gROB_0_0_0_0
//GLOBAL_OFFSET_B vgprGlobalReadOffsetB+3,  18,  7, 10 // gROB_0_0_1_0

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

s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 6  // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 64
s_mov_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeB       // incrB (unrollIdx)
s_mul_i32 s[sgprGlobalReadIncsB+0], s[sgprGlobalReadIncsB+0], s[sgprLoopCounterL]

s_sub_u32 s[sgprShadowLimitB_forTailLoop+0], s[sgprShadowLimitB_forTailLoop+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB_forTailLoop+1], s[sgprShadowLimitB_forTailLoop+1], 0 // limit -= inc)


/* global read addresses: addresses a */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s65, s[sgprWorkGroup0], 256           // WorkGroup[01] * MT
s_mul_i32 s64, s[sgprWorkGroup0], 256              // WorkGroup[01] * MT
s_mul_hi_u32 s65, s64, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s64, s64, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
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

s_mov_b64 s[sgprSrdA_forTailLoop+0:sgprSrdA_forTailLoop+1], s[sgprSrdA+0:sgprSrdA+1]
s_mov_b64 s[sgprSrdA_forTailLoop+2:sgprSrdA_forTailLoop+3], s[sgprSrdA+2:sgprSrdA+3]
s_mov_b64 s[sgprShadowLimitA_forTailLoop+0:sgprShadowLimitA_forTailLoop+1], s[sgprShadowLimitA+0:sgprShadowLimitA+1]

.ifdef FLUX_TNLOADER_KN_DIRECT
s_mul_i32 s[sgprGlobalReadIncsA+0], DepthU*BpeA, s[sgprStrideA0I] // advance 64 K rows
.else
s_mov_b32 s[sgprGlobalReadIncsA+0], DepthU*BpeA       // incrA (unrollIdx)
.endif
s_mul_i32 s[sgprGlobalReadIncsA+0], s[sgprGlobalReadIncsA+0], s[sgprLoopCounterL]

s_add_u32 s[sgprSrdA_forTailLoop+0], s[sgprSrdA_forTailLoop+0], s[sgprGlobalReadIncsA+0] // 
s_addc_u32 s[sgprSrdA_forTailLoop+1], s[sgprSrdA_forTailLoop+1], 0         // 
s_sub_u32 s[sgprShadowLimitA_forTailLoop+0], s[sgprShadowLimitA_forTailLoop+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA_forTailLoop+1], s[sgprShadowLimitA_forTailLoop+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA_forTailLoop+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA_forTailLoop+2], s[sgprShadowLimitA_forTailLoop+0]    // Move shadow to real if we are within 2^32


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
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

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
s_mov_b32 s[sgprSrdB+2], BufferLimit              // direct-input offsets span full input


/* global read addresses: increments a */

.ifdef FLUX_TNLOADER_KN_DIRECT
s_mul_i32 s[sgprGlobalReadIncsA+0], DepthU*BpeA, s[sgprStrideA0I] // advance 64 K rows
.else
s_mov_b32 s[sgprGlobalReadIncsA+0], DepthU*BpeA    // incrA (unrollIdx)
.endif


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
.ifdef FLUX_TNLOADER_KN_DIRECT
/* Reproduce the TN compute body's wave-rotated LDS layout from contiguous
 * physical KN vectors.  The 2*p rotation is encoded by the 0x400-spaced
 * half-word stores in the producer block below.
 */
v_and_b32 v[vgprLocalWriteAddrA], 63, v[vgprSerial]
v_lshlrev_b32 v[vgprLocalWriteAddrA], 1, v[vgprLocalWriteAddrA]
v_lshrrev_b32 v4, 6, v[vgprSerial]
v_and_b32 v4, 3, v4
v_lshlrev_b32 v4, 8, v4
_v_add_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA]
.else
v_and_b32 v4, 255, v[vgprSerial]
v_and_b32 v1, 7, v4
v_lshrrev_b32 v0, 3, v4
v_lshlrev_b32 v1, 3, v1

v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x40, v0     // lwAL**(MTA + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrA], v1, v[vgprLocalWriteAddrA], 0x1 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpe
v_readfirstlane_b32 s[sgprLocalWriteAddrA], v[vgprLocalWriteAddrA] // Copy lds write address VGPR to SGPR

s_and_b32 s90, 3, s[sgprWaveiD]                    // 
s_lshl_b32 s90, s90, 17                            // 
s_or_b32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA], s90 // 

s_mov_b32 s[sgprLocalWriteAddrA_ori], s[sgprLocalWriteAddrA]  // 
.endif

/* local write addresses: first offset b */

/* declare loop num iterations */


s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 6 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 64
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter

/* local read addresses: init pointers a */


/* localReadInitPointers */

/* local read addresses: init pointers b */


/* localReadInitPointers */


/* prefetch: global -> local */

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 ShadowInitStart_9                   // skip to ShadowInitStart iter b/c numIter==0

s_cmp_le_i32 s[sgprWaveiD], 7                      // 
s_cbranch_scc1 MmacWave                            // 
s_mov_b32 s[sgprLoopforPf], 0                      // 
Prefetch_Begin:                                    // 

.ifdef FLUX_TNLOADER_KN_DIRECT
.if FLUX_TNLOADER_KN_DIRECT_LDS
/* Candidate-only hardware direct-to-LDS producer.  Waves 0..7 enter the
 * compute path above; waves 8..11 are the four producer waves, with wave p
 * owning K rows [16p,16p+15] and its 0x400-byte LDS slice. */
s_cmp_ge_u32 s[sgprWaveiD], 12
s_cbranch_scc1 .L_tnloader_kn_directlds_skip_producer
.if FLUX_TNLOADER_KN_DIRECT_LDS_VECSTORE
/* Coalesced EKN loads followed by eight contiguous row-major LDS stores.
 * This keeps the external input unpacked and replaces the historical BF16
 * half-word transpose scatter with vector stores. */
buffer_load_dwordx4 v[128:131], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[132:135], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[136:139], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[140:143], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[144:147], v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[148:151], v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[152:155], v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[156:159], v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
s_waitcnt vmcnt(0)
v_and_b32 v167, 0x8000, v[vgprLocalWriteAddrA]
v_and_b32 v180, 63, v[vgprSerial]
v_lshrrev_b32 v181, 2, v180
v_lshrrev_b32 v182, 6, v[vgprSerial]
v_and_b32 v182, 3, v182
v_lshlrev_b32 v182, 4, v182
v_add_u32 v181, v181, v182
v_lshlrev_b32 v181, 9, v181
v_and_b32 v180, 3, v[vgprSerial]
v_lshlrev_b32 v180, 4, v180
v_add_u32 v167, v167, v181
v_add_u32 v167, v167, v180
ds_write_b128 v167, v[128:131] offset:0
ds_write_b128 v167, v[132:135] offset:64
ds_write_b128 v167, v[136:139] offset:128
ds_write_b128 v167, v[140:143] offset:192
ds_write_b128 v167, v[144:147] offset:256
ds_write_b128 v167, v[148:151] offset:320
ds_write_b128 v167, v[152:155] offset:384
ds_write_b128 v167, v[156:159] offset:448
s_waitcnt lgkmcnt(0)
.elseif FLUX_TNLOADER_KN_DIRECT_LDS_STRUCT_SPLIT
/* Split struct-buffer candidate.  Each producer wave owns one K16 page,
 * keeping voffset below 8KB; m0 selects the page in the 32KB A tile. */
v_and_b32 v167, 0x8000, v[vgprLocalWriteAddrA]
v_lshrrev_b32 v180, 6, v[vgprSerial]
v_and_b32 v180, 3, v180
v_lshlrev_b32 v180, 13, v180
v_add_u32 v167, v167, v180
v_readfirstlane_b32 s[sgprLocalWriteAddrA], v167
s_mov_b32 m0, s[sgprLocalWriteAddrA]
v_and_b32 v180, 63, v[vgprSerial]
v_lshrrev_b32 v181, 2, v180
v_lshrrev_b32 v182, 6, v[vgprSerial]
v_and_b32 v182, 3, v182
v_lshlrev_b32 v180, 4, v182
v_add_u32 v180, v180, v181
v_mul_lo_u32 v184, s[sgprStrideA0I], v180
v_lshlrev_b32 v182, 8, v181
v_sub_u32 v184, v184, v182
v_lshlrev_b32 v184, 1, v184
v_add_u32 v184, 16, v184
v_lshlrev_b32 v183, 9, v181
v_and_b32 v181, 3, v[vgprSerial]
v_lshlrev_b32 v181, 1, v181
v_add_u32 v183, v183, v181
v_mov_b32 v185, 0
v_add_u32 v185, v183, v185
buffer_load_dwordx4 v[184:185], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
v_mov_b32 v187, 64
v_add_u32 v187, v183, v187
v_mov_b32 v186, v184
buffer_load_dwordx4 v[186:187], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
v_mov_b32 v189, 128
v_add_u32 v189, v183, v189
v_mov_b32 v188, v184
buffer_load_dwordx4 v[188:189], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
v_mov_b32 v191, 192
v_add_u32 v191, v183, v191
v_mov_b32 v190, v184
buffer_load_dwordx4 v[190:191], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
v_mov_b32 v193, 256
v_add_u32 v193, v183, v193
v_mov_b32 v192, v184
buffer_load_dwordx4 v[192:193], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
v_mov_b32 v195, 320
v_add_u32 v195, v183, v195
v_mov_b32 v194, v184
buffer_load_dwordx4 v[194:195], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
v_mov_b32 v197, 384
v_add_u32 v197, v183, v197
v_mov_b32 v196, v184
buffer_load_dwordx4 v[196:197], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
v_mov_b32 v199, 448
v_add_u32 v199, v183, v199
v_mov_b32 v198, v184
buffer_load_dwordx4 v[198:199], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
.elseif FLUX_TNLOADER_KN_DIRECT_LDS_STRUCT
/* Struct-buffer direct candidate.  vindex carries the difference between
 * the full EKN row stride and the compact 256-column LDS row stride, while
 * voffset is the compact LDS tile offset.  This is the only direct load form
 * that can represent the two layouts independently on gfx936. */
v_and_b32 v167, 0x8000, v[vgprLocalWriteAddrA]
v_readfirstlane_b32 s[sgprLocalWriteAddrA], v167
s_mov_b32 m0, s[sgprLocalWriteAddrA]
v_and_b32 v180, 63, v[vgprSerial]
v_lshrrev_b32 v181, 2, v180
v_lshrrev_b32 v182, 6, v[vgprSerial]
v_and_b32 v182, 3, v182
v_lshlrev_b32 v182, 4, v182
v_add_u32 v180, v181, v182
v_lshlrev_b32 v183, 9, v180
v_and_b32 v181, 3, v[vgprSerial]
v_lshlrev_b32 v181, 4, v181
v_add_u32 v183, v183, v181
v_add_u32 v185, v183, 0
v_sub_u32 v184, v[vgprGlobalReadOffsetA+0], v185
buffer_load_dwordx4 v[184:185], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
v_mov_b32 v187, 64
v_add_u32 v187, v183, v187
v_sub_u32 v186, v[vgprGlobalReadOffsetA+1], v187
buffer_load_dwordx4 v[186:187], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
v_mov_b32 v189, 128
v_add_u32 v189, v183, v189
v_sub_u32 v188, v[vgprGlobalReadOffsetA+2], v189
buffer_load_dwordx4 v[188:189], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
v_mov_b32 v191, 192
v_add_u32 v191, v183, v191
v_sub_u32 v190, v[vgprGlobalReadOffsetA+3], v191
buffer_load_dwordx4 v[190:191], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
v_mov_b32 v193, 256
v_add_u32 v193, v183, v193
v_sub_u32 v192, v[vgprGlobalReadOffsetA+4], v193
buffer_load_dwordx4 v[192:193], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
v_mov_b32 v195, 320
v_add_u32 v195, v183, v195
v_sub_u32 v194, v[vgprGlobalReadOffsetA+5], v195
buffer_load_dwordx4 v[194:195], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
v_mov_b32 v197, 384
v_add_u32 v197, v183, v197
v_sub_u32 v196, v[vgprGlobalReadOffsetA+6], v197
buffer_load_dwordx4 v[196:197], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
v_mov_b32 v199, 448
v_add_u32 v199, v183, v199
v_sub_u32 v198, v[vgprGlobalReadOffsetA+7], v199
buffer_load_dwordx4 v[198:199], s[sgprSrdA:sgprSrdA+3], 0, idxen offen offset:0, lds
.elseif FLUX_TNLOADER_KN_DIRECT_LDS_ROWMAJOR
/* Direct global->LDS row-major candidate.  buffer_load_lds uses the VGPR
 * offset for the global address and the LDS-relative write position, so the
 * EKN byte offset is used directly and m0 remains at the page base. */
v_and_b32 v4, 0x8000, v[vgprLocalWriteAddrA]
v_readfirstlane_b32 s[sgprLocalWriteAddrA], v4
s_mov_b32 m0, s[sgprLocalWriteAddrA]
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
.elseif FLUX_TNLOADER_KN_DIRECT_LDS_AILKSTAGE
/* Native Ailk four-stage producer.  Every K16 stage uses the same four
 * virtual producer waves as the source-backed Ailk contract. */
v_and_b32 v4, 0x8000, v[vgprLocalWriteAddrA]
v_readfirstlane_b32 s[sgprLocalWriteAddrA], v4
s_sub_u32 s64, s[sgprWaveiD], 8
s_lshl_b32 s65, s64, 10
s_and_b32 s66, s64, 1
s_lshl_b32 s66, s66, 18
s_or_b32 s65, s65, s66
s_or_b32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA], s65
s_mov_b64 exec, -1
s_mov_b32 m0, s[sgprLocalWriteAddrA]
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, s[sgprLocalWriteAddrA], 0x4000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, s[sgprLocalWriteAddrA], 0x8000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, s[sgprLocalWriteAddrA], 0xc000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_mov_b64 exec, 0
.elseif FLUX_TNLOADER_KN_DIRECT_LDS_AILKPROD
/* Native Ailk producer contract.  Each p wave writes four K16 slices;
 * within a slice the two direct loads are the permuted K row and +8 row.
 * The m0 sequence 0,0x1000,...,0x7000 matches the four 0x2000 slices. */
v_and_b32 v4, 0x8000, v[vgprLocalWriteAddrA]
v_readfirstlane_b32 s[sgprLocalWriteAddrA], v4
s_sub_u32 s64, s[sgprWaveiD], 8
s_lshl_b32 s65, s64, 10
s_and_b32 s66, s64, 1
s_lshl_b32 s66, s66, 18
s_or_b32 s65, s65, s66
s_or_b32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA], s65
s_mov_b32 m0, s[sgprLocalWriteAddrA]
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
.else
v_and_b32 v4, 0x8000, v[vgprLocalWriteAddrA]
v_lshrrev_b32 v5, 6, v[vgprSerial]
v_and_b32 v5, 3, v5
v_lshlrev_b32 v5, 10, v5
v_or_b32 v4, v4, v5
v_readfirstlane_b32 s[sgprLocalWriteAddrA], v4
s_mov_b32 m0, s[sgprLocalWriteAddrA]
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
s_add_u32 m0, m0, 0x1000
s_nop 0
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds
.endif
.L_tnloader_kn_directlds_skip_producer:
.else
.if FLUX_TNLOADER_KN_DIRECT_PRODUCER8
/* Candidate-only eight-wave producer.  Waves 8..11 retire the first four
 * N-strided vectors and waves 12..15 retire the second four vectors.  The
 * paired waves share the same raw-KN lane map but write disjoint halves of
 * the established TN LDS tile, so no transpose or extra staging is added. */
v_sub_u32 v168, v[vgprLocalWriteAddrA], v165
v_sub_u32 v169, v[vgprLocalWriteAddrA], v166
v_sub_u32 v170, v[vgprLocalWriteAddrA], v167
s_cmp_lt_i32 s[sgprWaveiD], 12
s_cbranch_scc1 KN_PRODUCER8_LOW

buffer_load_dwordx4 v[128:131], v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[132:135], v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[136:139], v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[140:143], v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
s_waitcnt vmcnt(0)
ds_write_b16 v[vgprLocalWriteAddrA], v128 offset:0x4000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v128 offset:0x4420
ds_write_b16 v[vgprLocalWriteAddrA], v129 offset:0x4840
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v129 offset:0x4c60
ds_write_b16 v[vgprLocalWriteAddrA], v130 offset:0x4080
ds_write_b16_d16_hi v168, v130 offset:0x44a0
ds_write_b16 v169, v131 offset:0x48c0
ds_write_b16_d16_hi v170, v131 offset:0x4ce0
ds_write_b16 v[vgprLocalWriteAddrA], v132 offset:0x5000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v132 offset:0x5420
ds_write_b16 v[vgprLocalWriteAddrA], v133 offset:0x5840
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v133 offset:0x5c60
ds_write_b16 v[vgprLocalWriteAddrA], v134 offset:0x5080
ds_write_b16_d16_hi v168, v134 offset:0x54a0
ds_write_b16 v169, v135 offset:0x58c0
ds_write_b16_d16_hi v170, v135 offset:0x5ce0
ds_write_b16 v[vgprLocalWriteAddrA], v136 offset:0x6000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v136 offset:0x6420
ds_write_b16 v[vgprLocalWriteAddrA], v137 offset:0x6840
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v137 offset:0x6c60
ds_write_b16 v[vgprLocalWriteAddrA], v138 offset:0x6080
ds_write_b16_d16_hi v168, v138 offset:0x64a0
ds_write_b16 v169, v139 offset:0x68c0
ds_write_b16_d16_hi v170, v139 offset:0x6ce0
ds_write_b16 v[vgprLocalWriteAddrA], v140 offset:0x7000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v140 offset:0x7420
ds_write_b16 v[vgprLocalWriteAddrA], v141 offset:0x7840
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v141 offset:0x7c60
ds_write_b16 v[vgprLocalWriteAddrA], v142 offset:0x7080
ds_write_b16_d16_hi v168, v142 offset:0x74a0
ds_write_b16 v169, v143 offset:0x78c0
ds_write_b16_d16_hi v170, v143 offset:0x7ce0
s_branch KN_PRODUCER8_DONE

KN_PRODUCER8_LOW:
buffer_load_dwordx4 v[128:131], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[132:135], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[136:139], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[140:143], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
s_waitcnt vmcnt(0)
ds_write_b16 v[vgprLocalWriteAddrA], v128 offset:0x0000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v128 offset:0x0420
ds_write_b16 v[vgprLocalWriteAddrA], v129 offset:0x0840
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v129 offset:0x0c60
ds_write_b16 v[vgprLocalWriteAddrA], v130 offset:0x0080
ds_write_b16_d16_hi v168, v130 offset:0x04a0
ds_write_b16 v169, v131 offset:0x08c0
ds_write_b16_d16_hi v170, v131 offset:0x0ce0
ds_write_b16 v[vgprLocalWriteAddrA], v132 offset:0x1000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v132 offset:0x1420
ds_write_b16 v[vgprLocalWriteAddrA], v133 offset:0x1840
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v133 offset:0x1c60
ds_write_b16 v[vgprLocalWriteAddrA], v134 offset:0x1080
ds_write_b16_d16_hi v168, v134 offset:0x14a0
ds_write_b16 v169, v135 offset:0x18c0
ds_write_b16_d16_hi v170, v135 offset:0x1ce0
ds_write_b16 v[vgprLocalWriteAddrA], v136 offset:0x2000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v136 offset:0x2420
ds_write_b16 v[vgprLocalWriteAddrA], v137 offset:0x2840
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v137 offset:0x2c60
ds_write_b16 v[vgprLocalWriteAddrA], v138 offset:0x2080
ds_write_b16_d16_hi v168, v138 offset:0x24a0
ds_write_b16 v169, v139 offset:0x28c0
ds_write_b16_d16_hi v170, v139 offset:0x2ce0
ds_write_b16 v[vgprLocalWriteAddrA], v140 offset:0x3000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v140 offset:0x3420
ds_write_b16 v[vgprLocalWriteAddrA], v141 offset:0x3840
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v141 offset:0x3c60
ds_write_b16 v[vgprLocalWriteAddrA], v142 offset:0x3080
ds_write_b16_d16_hi v168, v142 offset:0x34a0
ds_write_b16 v169, v143 offset:0x38c0
ds_write_b16_d16_hi v170, v143 offset:0x3ce0

KN_PRODUCER8_DONE:
s_waitcnt lgkmcnt(0)
.endif
.if !FLUX_TNLOADER_KN_DIRECT_PRODUCER8
.ifdef FLUX_TNLOADER_KN_DIRECT_PIPE
/* Safe producer pipeline: each next global vector is launched before the
 * previous vector's LDS scatter, but every vector is consumed only after a
 * full vmcnt(0).  This avoids relying on vmcnt ordering across registers. */
.macro KN_DIRECT_STORE_GROUP src0:req, src1:req, src2:req, src3:req, off0:req, off1:req, off2:req, off3:req, off4:req, off5:req, off6:req, off7:req
ds_write_b16 v[vgprLocalWriteAddrA], v[\src0] offset:\off0
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v[\src0] offset:\off1
ds_write_b16 v[vgprLocalWriteAddrA], v[\src1] offset:\off2
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v[\src1] offset:\off3
ds_write_b16 v[vgprLocalWriteAddrA], v[\src2] offset:\off4
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v[\src2] offset:\off5
ds_write_b16 v[vgprLocalWriteAddrA], v[\src3] offset:\off6
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v[\src3] offset:\off7
.endm

buffer_load_dwordx4 v[128:131], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
s_waitcnt vmcnt(0)
buffer_load_dwordx4 v[132:135], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
KN_DIRECT_STORE_GROUP 128, 129, 130, 131, 0x0000, 0x0400, 0x0800, 0x0c00, 0x0080, 0x0480, 0x0880, 0x0c80
s_waitcnt vmcnt(0)
buffer_load_dwordx4 v[136:139], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
KN_DIRECT_STORE_GROUP 132, 133, 134, 135, 0x1000, 0x1400, 0x1800, 0x1c00, 0x1080, 0x1480, 0x1880, 0x1c80
s_waitcnt vmcnt(0)
buffer_load_dwordx4 v[140:143], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
KN_DIRECT_STORE_GROUP 136, 137, 138, 139, 0x2000, 0x2400, 0x2800, 0x2c00, 0x2080, 0x2480, 0x2880, 0x2c80
s_waitcnt vmcnt(0)
buffer_load_dwordx4 v[144:147], v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
KN_DIRECT_STORE_GROUP 140, 141, 142, 143, 0x3000, 0x3400, 0x3800, 0x3c00, 0x3080, 0x3480, 0x3880, 0x3c80
s_waitcnt vmcnt(0)
buffer_load_dwordx4 v[148:151], v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
KN_DIRECT_STORE_GROUP 144, 145, 146, 147, 0x4000, 0x4400, 0x4800, 0x4c00, 0x4080, 0x4480, 0x4880, 0x4c80
s_waitcnt vmcnt(0)
buffer_load_dwordx4 v[152:155], v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
KN_DIRECT_STORE_GROUP 148, 149, 150, 151, 0x5000, 0x5400, 0x5800, 0x5c00, 0x5080, 0x5480, 0x5880, 0x5c80
s_waitcnt vmcnt(0)
buffer_load_dwordx4 v[156:159], v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
KN_DIRECT_STORE_GROUP 152, 153, 154, 155, 0x6000, 0x6400, 0x6800, 0x6c00, 0x6080, 0x6480, 0x6880, 0x6c80
s_waitcnt vmcnt(0)
KN_DIRECT_STORE_GROUP 156, 157, 158, 159, 0x7000, 0x7400, 0x7800, 0x7c00, 0x7080, 0x7480, 0x7880, 0x7c80
s_waitcnt lgkmcnt(0)
.else
/* KN global vectors are contiguous in N.  Materialize them in VGPRs and
 * scatter their low/high BF16 halves into the TN compute body's established
 * LDS row permutation.  v128..v159 are free aliases in producer waves.
 */
.if FLUX_TNLOADER_KN_PRODUCER4
s_cmp_ge_u32 s[sgprWaveiD], 12
s_cbranch_scc1 .L_tnloader_kn_producer4_skip_a
.endif
buffer_load_dwordx4 v[128:131], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[132:135], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[136:139], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[140:143], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[144:147], v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[148:151], v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[152:155], v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
buffer_load_dwordx4 v[156:159], v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0
.ifdef FLUX_TNLOADER_KN_DIRECT_STAGGER
s_waitcnt vmcnt(7)
.else
s_waitcnt vmcnt(0)
.endif

ds_write_b16 v[vgprLocalWriteAddrA], v128 offset:0x0000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v128 offset:0x0400
ds_write_b16 v[vgprLocalWriteAddrA], v129 offset:0x0800
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v129 offset:0x0c00
ds_write_b16 v[vgprLocalWriteAddrA], v130 offset:0x0080
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v130 offset:0x0480
ds_write_b16 v[vgprLocalWriteAddrA], v131 offset:0x0880
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v131 offset:0x0c80

.ifdef FLUX_TNLOADER_KN_DIRECT_STAGGER
s_waitcnt vmcnt(6)
.endif
ds_write_b16 v[vgprLocalWriteAddrA], v132 offset:0x1000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v132 offset:0x1400
ds_write_b16 v[vgprLocalWriteAddrA], v133 offset:0x1800
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v133 offset:0x1c00
ds_write_b16 v[vgprLocalWriteAddrA], v134 offset:0x1080
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v134 offset:0x1480
ds_write_b16 v[vgprLocalWriteAddrA], v135 offset:0x1880
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v135 offset:0x1c80

.ifdef FLUX_TNLOADER_KN_DIRECT_STAGGER
s_waitcnt vmcnt(5)
.endif
ds_write_b16 v[vgprLocalWriteAddrA], v136 offset:0x2000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v136 offset:0x2400
ds_write_b16 v[vgprLocalWriteAddrA], v137 offset:0x2800
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v137 offset:0x2c00
ds_write_b16 v[vgprLocalWriteAddrA], v138 offset:0x2080
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v138 offset:0x2480
ds_write_b16 v[vgprLocalWriteAddrA], v139 offset:0x2880
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v139 offset:0x2c80

.ifdef FLUX_TNLOADER_KN_DIRECT_STAGGER
s_waitcnt vmcnt(4)
.endif
ds_write_b16 v[vgprLocalWriteAddrA], v140 offset:0x3000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v140 offset:0x3400
ds_write_b16 v[vgprLocalWriteAddrA], v141 offset:0x3800
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v141 offset:0x3c00
ds_write_b16 v[vgprLocalWriteAddrA], v142 offset:0x3080
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v142 offset:0x3480
ds_write_b16 v[vgprLocalWriteAddrA], v143 offset:0x3880
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v143 offset:0x3c80

.ifdef FLUX_TNLOADER_KN_DIRECT_STAGGER
s_waitcnt vmcnt(3)
.endif
ds_write_b16 v[vgprLocalWriteAddrA], v144 offset:0x4000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v144 offset:0x4400
ds_write_b16 v[vgprLocalWriteAddrA], v145 offset:0x4800
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v145 offset:0x4c00
ds_write_b16 v[vgprLocalWriteAddrA], v146 offset:0x4080
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v146 offset:0x4480
ds_write_b16 v[vgprLocalWriteAddrA], v147 offset:0x4880
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v147 offset:0x4c80

.ifdef FLUX_TNLOADER_KN_DIRECT_STAGGER
s_waitcnt vmcnt(2)
.endif
ds_write_b16 v[vgprLocalWriteAddrA], v148 offset:0x5000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v148 offset:0x5400
ds_write_b16 v[vgprLocalWriteAddrA], v149 offset:0x5800
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v149 offset:0x5c00
ds_write_b16 v[vgprLocalWriteAddrA], v150 offset:0x5080
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v150 offset:0x5480
ds_write_b16 v[vgprLocalWriteAddrA], v151 offset:0x5880
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v151 offset:0x5c80

.ifdef FLUX_TNLOADER_KN_DIRECT_STAGGER
s_waitcnt vmcnt(1)
.endif
ds_write_b16 v[vgprLocalWriteAddrA], v152 offset:0x6000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v152 offset:0x6400
ds_write_b16 v[vgprLocalWriteAddrA], v153 offset:0x6800
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v153 offset:0x6c00
ds_write_b16 v[vgprLocalWriteAddrA], v154 offset:0x6080
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v154 offset:0x6480
ds_write_b16 v[vgprLocalWriteAddrA], v155 offset:0x6880
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v155 offset:0x6c80

ds_write_b16 v[vgprLocalWriteAddrA], v156 offset:0x7000
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v156 offset:0x7400
ds_write_b16 v[vgprLocalWriteAddrA], v157 offset:0x7800
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v157 offset:0x7c00
ds_write_b16 v[vgprLocalWriteAddrA], v158 offset:0x7080
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v158 offset:0x7480
ds_write_b16 v[vgprLocalWriteAddrA], v159 offset:0x7880
ds_write_b16_d16_hi v[vgprLocalWriteAddrA], v159 offset:0x7c80
s_waitcnt lgkmcnt(0)
.if FLUX_TNLOADER_KN_PRODUCER4
.L_tnloader_kn_producer4_skip_a:
.endif
.endif
.endif
.endif
.else
s_mov_b32 m0, s[sgprLocalWriteAddrA]            // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_0_0
s_add_u32 m0, m0, 0x1000                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 0x1000                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 0x1000                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 0x1000                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 0x1000                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 0x1000                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 0x1000                            // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_1_0
.endif

s_barrier

s_add_u32 s[sgprLoopforPf], s[sgprLoopforPf], 1    // 

s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // 
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // 
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

.ifdef FLUX_TNLOADER_KN_DIRECT
_v_add_u32 v[vgprLocalWriteAddrA], 0x8000, v[vgprLocalWriteAddrA]
v_and_b32 v[vgprLocalWriteAddrA], 0xffff, v[vgprLocalWriteAddrA]
.else
s_add_u32 s[sgprLocalWriteAddrA], 0x8000, s[sgprLocalWriteAddrA] // 
s_and_b32 s90, s[sgprLocalWriteAddrA], 0x1ffff     // 
s_cmp_ge_u32 s90, s[sgprMask]                   // 
s_cmov_b32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA_ori] // 
.endif
s_cmp_lt_i32 s[sgprLoopforPf], 1                   // 
s_cbranch_scc1 SkipBarrier                         // 
s_waitcnt vmcnt(0)                                 // 
s_barrier                                          // 
SkipBarrier:                                       // 
s_cmp_lt_i32 s[sgprLoopforPf], s[sgprLoopCounterL] // 
s_cbranch_scc1 Prefetch_Begin                      // 
s_waitcnt vmcnt(0)                                 // 
s_barrier                                          // 
s_endpgm    

                                       
MmacWave:


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

buffer_load_dwordx4 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+3], v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+7], v[vgprGlobalReadOffsetB+2:vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+11], v[vgprGlobalReadOffsetB+4:vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+15], v[vgprGlobalReadOffsetB+6:vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0,idxen  offen offset:0

s_barrier
s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=08wait for global read

s_barrier //



s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)

s_cmp_le_i32  s[sgprLoopCounterL], 1
s_cmov_b32 s[sgprSrdB+2], 0
s_cmov_b32 s[sgprGlobalReadIncsB+0], 0


/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/

openLoopL_11:

s_cmp_le_u32 s[sgprLoopCounterL], 0x0              // LoopCounterL < EndCounter
s_cbranch_scc1 LoopEndL_2                          // do not enter LoopL

s_and_b32 s90, 7, s[sgprWaveiD]
s_cmp_ge_u32 s90, 4
s_cbranch_scc1 LoopBeginL_2

LoopBeginL_1:


/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/

label_0012: // LoopCopy1 


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

.if FLUX_TNLOADER_KN_DIRECT_LDS
KN_RAW_A_READS 0, 0x0000, 0x0400
.else
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrASwap] offset:0x200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:0x1000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrASwap] offset:0x1200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:0x2000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrASwap] offset:0x2200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:0x3000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrASwap] offset:0x3200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:0x4000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrASwap] offset:0x4200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:0x5000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrASwap] offset:0x5200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:0x6000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrASwap] offset:0x6200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:0x7000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrASwap] offset:0x7200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0
.endif

s_waitcnt lgkmcnt(0)
MAC_32x4x2_X0_I0
s_barrier

 .if FLUX_TNLOADER_KN_DIRECT_LDS
KN_RAW_A_READS 0, 0x0800, 0x0c00
buffer_load_dwordx4 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+3], v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+7], v[vgprGlobalReadOffsetB+2:vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+11], v[vgprGlobalReadOffsetB+4:vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+15], v[vgprGlobalReadOffsetB+6:vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0,idxen  offen offset:0
 .else
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:0x40 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrASwap1] offset:0x240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:0x1040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrASwap1] offset:0x1240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:0x2040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrASwap1] offset:0x2240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:0x3040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrASwap1] offset:0x3240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
buffer_load_dwordx4 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+3], v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+7], v[vgprGlobalReadOffsetB+2:vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+11], v[vgprGlobalReadOffsetB+4:vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+15], v[vgprGlobalReadOffsetB+6:vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0,idxen  offen offset:0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:0x4040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrASwap1] offset:0x4240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:0x5040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrASwap1] offset:0x5240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:0x6040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrASwap1] offset:0x6240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:0x7040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrASwap1] offset:0x7240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0
 .endif

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_cmp_le_i32  s[sgprLoopCounterL], 2
s_cmov_b32 s[sgprSrdB+2], 0
s_cmov_b32 s[sgprGlobalReadIncsB+0], 0
s_waitcnt lgkmcnt(0)
MAC_32x4x2_X0_I1
s_barrier

s_waitcnt vmcnt(0)


/******************************************/
/* Unrolled Loop - End 1/2                */
/******************************************/


/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==1
s_cbranch_scc1 LoopEndL_oddexit_3                  // exit LoopL


/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/

label_0013: // LoopCopy2 


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

.if FLUX_TNLOADER_KN_DIRECT_LDS
KN_RAW_A_READS 0x8000, 0x0000, 0x0400
.else
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:0x8000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrASwap] offset:0x8200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:0x9000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrASwap] offset:0x9200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:0xa000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrASwap] offset:0xa200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:0xb000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrASwap] offset:0xb200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:0xc000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrASwap] offset:0xc200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:0xd000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrASwap] offset:0xd200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:0xe000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrASwap] offset:0xe200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:0xf000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrASwap] offset:0xf200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0
.endif

s_waitcnt lgkmcnt(0)
MAC_32x4x2_X1_I0
s_barrier

.if FLUX_TNLOADER_KN_DIRECT_LDS
KN_RAW_A_READS 0x8000, 0x0800, 0x0c00
buffer_load_dwordx4 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+3], v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+7], v[vgprGlobalReadOffsetB+2:vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+11], v[vgprGlobalReadOffsetB+4:vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+15], v[vgprGlobalReadOffsetB+6:vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0,idxen  offen offset:0
 .else
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:0x8040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrASwap1] offset:0x8240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:0x9040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrASwap1] offset:0x9240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:0xa040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrASwap1] offset:0xa240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:0xb040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrASwap1] offset:0xb240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
buffer_load_dwordx4 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+3], v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+7], v[vgprGlobalReadOffsetB+2:vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+11], v[vgprGlobalReadOffsetB+4:vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+15], v[vgprGlobalReadOffsetB+6:vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0,idxen  offen offset:0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:0xc040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrASwap1] offset:0xc240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:0xd040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrASwap1] offset:0xd240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:0xe040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrASwap1] offset:0xe240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:0xf040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrASwap1] offset:0xf240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0
.endif

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_cmp_le_i32  s[sgprLoopCounterL], 2
s_cmov_b32 s[sgprSrdB+2], 0
s_cmov_b32 s[sgprGlobalReadIncsB+0], 0
s_waitcnt lgkmcnt(0)
MAC_32x4x2_X1_I1
s_barrier

s_waitcnt vmcnt(0)

/******************************************/
/* Unrolled Loop - End 2/2 (final)        */
/******************************************/


/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==1
s_cbranch_scc0 LoopBeginL_1                        // restart LoopL
s_branch LoopEndL_2                                // exit unroll loopL (and skip oddexit)
/////////////////////////////////////////////////////
//////////////////////////////////////////////////////
LoopBeginL_2:

/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/

label_00121: // LoopCopy1 


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrASwap] offset:0x200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:0x1000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrASwap] offset:0x1200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:0x2000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrASwap] offset:0x2200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:0x3000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrASwap] offset:0x3200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:0x4000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrASwap] offset:0x4200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:0x5000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrASwap] offset:0x5200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:0x6000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrASwap] offset:0x6200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:0x7000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrASwap] offset:0x7200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0

s_waitcnt lgkmcnt(0)
s_barrier
MAC_32x4x2_X0_I0

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:0x40 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrASwap1] offset:0x240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:0x1040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrASwap1] offset:0x1240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:0x2040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrASwap1] offset:0x2240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:0x3040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrASwap1] offset:0x3240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
buffer_load_dwordx4 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+3], v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+7], v[vgprGlobalReadOffsetB+2:vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+11], v[vgprGlobalReadOffsetB+4:vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+15], v[vgprGlobalReadOffsetB+6:vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0,idxen  offen offset:0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:0x4040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrASwap1] offset:0x4240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:0x5040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrASwap1] offset:0x5240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:0x6040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrASwap1] offset:0x6240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:0x7040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrASwap1] offset:0x7240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_cmp_le_i32  s[sgprLoopCounterL], 2
s_cmov_b32 s[sgprSrdB+2], 0
s_cmov_b32 s[sgprGlobalReadIncsB+0], 0
s_waitcnt lgkmcnt(0)
s_barrier
MAC_32x4x2_X0_I1

s_waitcnt vmcnt(0)


/******************************************/
/* Unrolled Loop - End 1/2                */
/******************************************/


/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==1
s_cbranch_scc1 LoopEndL_oddexit_3                  // exit LoopL


/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/

label_00131: // LoopCopy2 


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:0x8000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrASwap] offset:0x8200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:0x9000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrASwap] offset:0x9200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:0xa000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrASwap] offset:0xa200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:0xb000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrASwap] offset:0xb200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:0xc000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrASwap] offset:0xc200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:0xd000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrASwap] offset:0xd200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:0xe000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrASwap] offset:0xe200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:0xf000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrASwap] offset:0xf200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0

s_waitcnt lgkmcnt(0)
s_barrier

MAC_32x4x2_X1_I0

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:0x8040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrASwap1] offset:0x8240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:0x9040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrASwap1] offset:0x9240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:0xa040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrASwap1] offset:0xa240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:0xb040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrASwap1] offset:0xb240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
buffer_load_dwordx4 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+3], v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+7], v[vgprGlobalReadOffsetB+2:vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+11], v[vgprGlobalReadOffsetB+4:vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+15], v[vgprGlobalReadOffsetB+6:vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0,idxen  offen offset:0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:0xc040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrASwap1] offset:0xc240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:0xd040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrASwap1] offset:0xd240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:0xe040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrASwap1] offset:0xe240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:0xf040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrASwap1] offset:0xf240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_cmp_le_i32  s[sgprLoopCounterL], 2
s_cmov_b32 s[sgprSrdB+2], 0
s_cmov_b32 s[sgprGlobalReadIncsB+0], 0
s_waitcnt lgkmcnt(0)
s_barrier

MAC_32x4x2_X1_I1

s_waitcnt vmcnt(0)

/******************************************/
/* Unrolled Loop - End 2/2 (final)        */
/******************************************/


/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==1
s_cbranch_scc0 LoopBeginL_2                        // restart LoopL


LoopEndL_oddexit_3: // unroll loop odditer exit

LoopEndL_2:


/* Before NLL: Check VGPR.checkin for INT8 LW */


/******************************************/
/* Opt. NoLoadLoop Without PAP - Begin         */
/******************************************/

s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 OptNLL_End_14                       // Branch if Beta is not zero


s_cmp_eq_u32 s[sgprAlpha], 1.0                     // alpha == 1.0?
s_cbranch_scc0 OptNLL_End_14                       // branch if alpha != 1
s_branch OptNLL_End_14                             // direct-output: use rowmapped edge write path

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

s_and_b32 s63, 63, s[sgprSizesSum+0]               // s63 = s[sgprSizesSum+0] % 32
s_cmp_eq_u32 s63, 0x0                              // numIterL == 0
s_cbranch_scc0 OptNLL_End_14                       // skip if tail loop required


/* Stores for OptNLL */
Summation_End_OptNLL_15:
/* endSummation: add vgpr [128...198) to pool */
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
v_and_b32 v134, 0, v129                            // v4&1
s_mov_b32 s54, 128                                 //  
v_mul_lo_u32 v134, s54, v134                       //  
v_lshlrev_b32 v128, 0, v132                        // ((v[vgprSerial]&63)&15)*2
_v_add_co_u32 v128, vcc, v128, v134                // row
v_lshrrev_b32 v134, 0, v129                        // v4>>1
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
                                           // 
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
v_mov_b32 v250, 0x7fff0000                         // fp32 Nan
v_mov_b32 v251, 0x7fff                             // rounding bias for bfloat16

FP32_TO_BF16 vgprValuC+0       // convert C to bf16
buffer_store_short v0, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+1       // convert C to bf16
buffer_store_short v1, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+2       // convert C to bf16
buffer_store_short v2, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+3       // convert C to bf16
buffer_store_short v3, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+4       // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v4, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+5       // convert C to bf16
buffer_store_short v5, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+6      // convert C to bf16
buffer_store_short v6, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+7       // convert C to bf16
buffer_store_short v7, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+8       // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v8, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+9       // convert C to bf16
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
s_mul_i32 s54, s[sgprStrideD1J], 32                // scale StrideD *= numRows(16) * bpe
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v96, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+97     // convert C to bf16
buffer_store_short v97, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+98     // convert C to bf16
buffer_store_short v98, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+99     // convert C to bf16
buffer_store_short v99, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+100   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v100, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+101   // convert C to bf16
buffer_store_short v101, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+102   // convert C to bf16
buffer_store_short v102, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+103   // convert C to bf16
buffer_store_short v103, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+104   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v104, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+105   // convert C to bf16
buffer_store_short v105, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+106   // convert C to bf16
buffer_store_short v106, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+107   // convert C to bf16
buffer_store_short v107, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+108   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v108, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+109   // convert C to bf16
buffer_store_short v109, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+110   // convert C to bf16
buffer_store_short v110, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+111   // convert C to bf16
buffer_store_short v111, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+112   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v112, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+113   // convert C to bf16
buffer_store_short v113, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+114   // convert C to bf16
buffer_store_short v114, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+115   // convert C to bf16
buffer_store_short v115, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+116   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v116, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+117   // convert C to bf16
buffer_store_short v117, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+118   // convert C to bf16
buffer_store_short v118, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+119   // convert C to bf16
buffer_store_short v119, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+120   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v120, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+121   // convert C to bf16
buffer_store_short v121, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+122   // convert C to bf16
buffer_store_short v122, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+123   // convert C to bf16
buffer_store_short v123, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+124   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v124, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+125   // convert C to bf16
buffer_store_short v125, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+126   // convert C to bf16
buffer_store_short v126, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+127   // convert C to bf16
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
s_cmp_le_i32 s[sgprWaveiD], 7
s_cbranch_scc0 label_0039

s_barrier

s_cmp_eq_u32 s[sgprShadowLimitB_forTailLoop+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB_forTailLoop+0]    // Move shadow to real if we are within 2^32

s_mov_b64 s[sgprSrdA+0:sgprSrdA+1], s[sgprSrdA_forTailLoop+0:sgprSrdA_forTailLoop+1]
s_mov_b64 s[sgprSrdA+2:sgprSrdA+3], s[sgprSrdA_forTailLoop+2:sgprSrdA_forTailLoop+3]

/* local write reset offsets a */



/* local write reset offsets b */



//numIterL = (((sizeL % LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 63, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 64
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0              // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 SkipTailLoopL_7                     // skip to end of tail loop b/c numIter==0

s_and_b32  s64, 7, s[sgprWaveiD]
s_lshr_b32 s64, s64, 2                             // 
s_lshl_b32 s64, s64, 8                             //

s_mul_i32  s65, s64, 64
s_mul_i32  s64, s64, s[sgprStrideA0I]

v_add_co_u32 v[vgprGlobalReadOffsetA+0], vcc, s64, v[vgprGlobalReadOffsetA+0]
v_add_co_u32 v[vgprGlobalReadOffsetA+1], vcc, s64, v[vgprGlobalReadOffsetA+1]
v_add_co_u32 v[vgprGlobalReadOffsetA+2], vcc, s64, v[vgprGlobalReadOffsetA+2]
v_add_co_u32 v[vgprGlobalReadOffsetA+3], vcc, s64, v[vgprGlobalReadOffsetA+3]

v_add_co_u32 v[vgprLocalWriteAddrA], vcc, s65, v[vgprLocalWriteAddrA]


v_and_b32 v[vgprG2LA+0], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+0], 3, v[vgprG2LA+0]      // 
v_add_co_u32 v[vgprG2LA+0], vcc, 0, v[vgprG2LA+0]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+0], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+0], vcc, 0, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v[vgprG2LA+0], -0x1, v[vgprG2LA+0], s[64:65] // 
v_and_b32 v201, v[vgprSerial], 7                   // 
v_lshlrev_b32 v201, 3, v201                        // 
v_add_co_u32 v201, vcc, 1, v201                    // 
v_cmp_lt_u32 s[64:65], v201, s[sgprLoopCounterL]   // 
v_add_co_u32 v201, vcc, 2, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v201, -0x1, v201, s[64:65]           // 
v_and_b32 v[vgprG2LA+1], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+1], 3, v[vgprG2LA+1]      // 
v_add_co_u32 v[vgprG2LA+1], vcc, 2, v[vgprG2LA+1]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+1], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+1], vcc, 4, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v[vgprG2LA+1], -0x1, v[vgprG2LA+1], s[64:65] // 
v_and_b32 v202, v[vgprSerial], 7                   // 
v_lshlrev_b32 v202, 3, v202                        // 
v_add_co_u32 v202, vcc, 3, v202                    // 
v_cmp_lt_u32 s[64:65], v202, s[sgprLoopCounterL]   // 
v_add_co_u32 v202, vcc, 6, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v202, -0x1, v202, s[64:65]           // 
v_and_b32 v[vgprG2LA+2], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+2], 3, v[vgprG2LA+2]      // 
v_add_co_u32 v[vgprG2LA+2], vcc, 4, v[vgprG2LA+2]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+2], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+2], vcc, 8, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v[vgprG2LA+2], -0x1, v[vgprG2LA+2], s[64:65] // 
v_and_b32 v203, v[vgprSerial], 7                   // 
v_lshlrev_b32 v203, 3, v203                        // 
v_add_co_u32 v203, vcc, 5, v203                    // 
v_cmp_lt_u32 s[64:65], v203, s[sgprLoopCounterL]   // 
v_add_co_u32 v203, vcc, 10, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v203, -0x1, v203, s[64:65]           // 
v_and_b32 v[vgprG2LA+3], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+3], 3, v[vgprG2LA+3]      // 
v_add_co_u32 v[vgprG2LA+3], vcc, 6, v[vgprG2LA+3]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+3], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+3], vcc, 12, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v[vgprG2LA+3], -0x1, v[vgprG2LA+3], s[64:65] // 
v_and_b32 v204, v[vgprSerial], 7                   // 
v_lshlrev_b32 v204, 3, v204                        // 
v_add_co_u32 v204, vcc, 7, v204                    // 
v_cmp_lt_u32 s[64:65], v204, s[sgprLoopCounterL]   // 
v_add_co_u32 v204, vcc, 14, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v204, -0x1, v204, s[64:65]           // 

v_and_b32 v[vgprG2LA+4], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+4], 3, v[vgprG2LA+4]      // 
v_add_co_u32 v[vgprG2LA+4], vcc, 0, v[vgprG2LA+4]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+4], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+4], vcc, 0, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v[vgprG2LA+4], -0x1, v[vgprG2LA+4], s[64:65] // 
v_and_b32 v205, v[vgprSerial], 7                   // 
v_lshlrev_b32 v205, 3, v205                        // 
v_add_co_u32 v205, vcc, 1, v205                    // 
v_cmp_lt_u32 s[64:65], v205, s[sgprLoopCounterL]   // 
v_add_co_u32 v205, vcc, 2, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v205, -0x1, v205, s[64:65]           // 
v_and_b32 v[vgprG2LA+5], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+5], 3, v[vgprG2LA+5]      // 
v_add_co_u32 v[vgprG2LA+5], vcc, 2, v[vgprG2LA+5]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+5], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+5], vcc, 4, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v[vgprG2LA+5], -0x1, v[vgprG2LA+5], s[64:65] // 
v_and_b32 v206, v[vgprSerial], 7                   // 
v_lshlrev_b32 v206, 3, v206                        // 
v_add_co_u32 v206, vcc, 3, v206                    // 
v_cmp_lt_u32 s[64:65], v206, s[sgprLoopCounterL]   // 
v_add_co_u32 v206, vcc, 6, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v206, -0x1, v206, s[64:65]           // 
v_and_b32 v[vgprG2LA+6], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+6], 3, v[vgprG2LA+6]      // 
v_add_co_u32 v[vgprG2LA+6], vcc, 4, v[vgprG2LA+6]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+6], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+6], vcc, 8, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v[vgprG2LA+6], -0x1, v[vgprG2LA+6], s[64:65] // 
v_and_b32 v207, v[vgprSerial], 7                   // 
v_lshlrev_b32 v207, 3, v207                        // 
v_add_co_u32 v207, vcc, 5, v207                    // 
v_cmp_lt_u32 s[64:65], v207, s[sgprLoopCounterL]   // 
v_add_co_u32 v207, vcc, 10, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v207, -0x1, v207, s[64:65]           // 
v_and_b32 v[vgprG2LA+7], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+7], 3, v[vgprG2LA+7]      // 
v_add_co_u32 v[vgprG2LA+7], vcc, 6, v[vgprG2LA+7]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+7], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+7], vcc, 12, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v[vgprG2LA+7], -0x1, v[vgprG2LA+7], s[64:65] // 
v_and_b32 v208, v[vgprSerial], 7                   // 
v_lshlrev_b32 v208, 3, v208                        // 
v_add_co_u32 v208, vcc, 7, v208                    // 
v_cmp_lt_u32 s[64:65], v208, s[sgprLoopCounterL]   // 
v_add_co_u32 v208, vcc, 14, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v208, -0x1, v208, s[64:65]           // 

v_and_b32 v[vgprG2LA+8], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+8], 3, v[vgprG2LA+8]      // 
v_add_co_u32 v[vgprG2LA+8], vcc, 0, v[vgprG2LA+8]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+8], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+8], vcc, 0, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v[vgprG2LA+8], -0x1, v[vgprG2LA+8], s[64:65] // 
v_and_b32 v209, v[vgprSerial], 7                   // 
v_lshlrev_b32 v209, 3, v209                        // 
v_add_co_u32 v209, vcc, 1, v209                    // 
v_cmp_lt_u32 s[64:65], v209, s[sgprLoopCounterL]   // 
v_add_co_u32 v209, vcc, 2, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v209, -0x1, v209, s[64:65]           // 
v_and_b32 v[vgprG2LA+9], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+9], 3, v[vgprG2LA+9]      // 
v_add_co_u32 v[vgprG2LA+9], vcc, 2, v[vgprG2LA+9]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+9], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+9], vcc, 4, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v[vgprG2LA+9], -0x1, v[vgprG2LA+9], s[64:65] // 
v_and_b32 v210, v[vgprSerial], 7                   // 
v_lshlrev_b32 v210, 3, v210                        // 
v_add_co_u32 v210, vcc, 3, v210                    // 
v_cmp_lt_u32 s[64:65], v210, s[sgprLoopCounterL]   // 
v_add_co_u32 v210, vcc, 6, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v210, -0x1, v210, s[64:65]           // 
v_and_b32 v[vgprG2LA+10], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+10], 3, v[vgprG2LA+10]      // 
v_add_co_u32 v[vgprG2LA+10], vcc, 4, v[vgprG2LA+10]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+10], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+10], vcc, 8, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v[vgprG2LA+10], -0x1, v[vgprG2LA+10], s[64:65] // 
v_and_b32 v211, v[vgprSerial], 7                   // 
v_lshlrev_b32 v211, 3, v211                        // 
v_add_co_u32 v211, vcc, 5, v211                    // 
v_cmp_lt_u32 s[64:65], v211, s[sgprLoopCounterL]   // 
v_add_co_u32 v211, vcc, 10, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v211, -0x1, v211, s[64:65]           // 
v_and_b32 v[vgprG2LA+11], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+11], 3, v[vgprG2LA+11]      // 
v_add_co_u32 v[vgprG2LA+11], vcc, 6, v[vgprG2LA+11]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+11], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+11], vcc, 12, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v[vgprG2LA+11], -0x1, v[vgprG2LA+11], s[64:65] // 
v_and_b32 v212, v[vgprSerial], 7                   // 
v_lshlrev_b32 v212, 3, v212                        // 
v_add_co_u32 v212, vcc, 7, v212                    // 
v_cmp_lt_u32 s[64:65], v212, s[sgprLoopCounterL]   // 
v_add_co_u32 v212, vcc, 14, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v212, -0x1, v212, s[64:65]           // 

v_and_b32 v[vgprG2LA+12], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+12], 3, v[vgprG2LA+12]      // 
v_add_co_u32 v[vgprG2LA+12], vcc, 0, v[vgprG2LA+12]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+12], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+12], vcc, 0, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v[vgprG2LA+12], -0x1, v[vgprG2LA+12], s[64:65] // 
v_and_b32 v213, v[vgprSerial], 7                   // 
v_lshlrev_b32 v213, 3, v213                        // 
v_add_co_u32 v213, vcc, 1, v213                    // 
v_cmp_lt_u32 s[64:65], v213, s[sgprLoopCounterL]   // 
v_add_co_u32 v213, vcc, 2, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v213, -0x1, v213, s[64:65]           // 
v_and_b32 v[vgprG2LA+13], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+13], 3, v[vgprG2LA+13]      // 
v_add_co_u32 v[vgprG2LA+13], vcc, 2, v[vgprG2LA+13]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+13], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+13], vcc, 4, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v[vgprG2LA+13], -0x1, v[vgprG2LA+13], s[64:65] // 
v_and_b32 v214, v[vgprSerial], 7                   // 
v_lshlrev_b32 v214, 3, v214                        // 
v_add_co_u32 v214, vcc, 3, v214                    // 
v_cmp_lt_u32 s[64:65], v214, s[sgprLoopCounterL]   // 
v_add_co_u32 v214, vcc, 6, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v214, -0x1, v214, s[64:65]           // 
v_and_b32 v[vgprG2LA+14], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+14], 3, v[vgprG2LA+14]      // 
v_add_co_u32 v[vgprG2LA+14], vcc, 4, v[vgprG2LA+14]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+14], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+14], vcc, 8, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v[vgprG2LA+14], -0x1, v[vgprG2LA+14], s[64:65] // 
v_and_b32 v215, v[vgprSerial], 7                   // 
v_lshlrev_b32 v215, 3, v215                        // 
v_add_co_u32 v215, vcc, 5, v215                    // 
v_cmp_lt_u32 s[64:65], v215, s[sgprLoopCounterL]   // 
v_add_co_u32 v215, vcc, 10, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v215, -0x1, v215, s[64:65]           // 
v_and_b32 v[vgprG2LA+15], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+15], 3, v[vgprG2LA+15]      // 
v_add_co_u32 v[vgprG2LA+15], vcc, 6, v[vgprG2LA+15]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+15], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+15], vcc, 12, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v[vgprG2LA+15], -0x1, v[vgprG2LA+15], s[64:65] // 
v_and_b32 v216, v[vgprSerial], 7                   // 
v_lshlrev_b32 v216, 3, v216                        // 
v_add_co_u32 v216, vcc, 7, v216                    // 
v_cmp_lt_u32 s[64:65], v216, s[sgprLoopCounterL]   // 
v_add_co_u32 v216, vcc, 14, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v216, -0x1, v216, s[64:65]           // 


/* global read a */

/* g2l=0, load component 0 */
buffer_load_short_d16 v[vgprG2LA+0+0], v[vgprG2LA+0+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_short_d16_hi v201, v201, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v201 // HasEccHalf: pack
/* g2l=0, load component 2 */
buffer_load_short_d16 v[vgprG2LA+0+1], v[vgprG2LA+0+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_short_d16_hi v202, v202, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v202 // HasEccHalf: pack
/* g2l=0, load component 4 */
buffer_load_short_d16 v[vgprG2LA+0+2], v[vgprG2LA+0+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_short_d16_hi v203, v203, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+0+2], v[vgprG2LA+0+2], v203 // HasEccHalf: pack
/* g2l=0, load component 6 */
buffer_load_short_d16 v[vgprG2LA+0+3], v[vgprG2LA+0+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_short_d16_hi v204, v204, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+0+3], v[vgprG2LA+0+3], v204 // HasEccHalf: pack
/* g2l=4, load component 0 */
buffer_load_short_d16 v[vgprG2LA+4+0], v[vgprG2LA+4+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_short_d16_hi v205, v205, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+4+0], v[vgprG2LA+4+0], v205 // HasEccHalf: pack
/* g2l=4, load component 2 */
buffer_load_short_d16 v[vgprG2LA+4+1], v[vgprG2LA+4+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 3 */
buffer_load_short_d16_hi v206, v206, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+4+1], v[vgprG2LA+4+1], v206 // HasEccHalf: pack
/* g2l=4, load component 4 */
buffer_load_short_d16 v[vgprG2LA+4+2], v[vgprG2LA+4+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 5 */
buffer_load_short_d16_hi v207, v207, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+4+2], v[vgprG2LA+4+2], v207 // HasEccHalf: pack
/* g2l=4, load component 6 */
buffer_load_short_d16 v[vgprG2LA+4+3], v[vgprG2LA+4+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 7 */
buffer_load_short_d16_hi v208, v208, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+4+3], v[vgprG2LA+4+3], v208 // HasEccHalf: pack

/* g2l=0, load component 0 */
buffer_load_short_d16 v[vgprG2LA+8+0], v[vgprG2LA+8+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_short_d16_hi v209, v209, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+8+0], v[vgprG2LA+8+0], v209 // HasEccHalf: pack
/* g2l=0, load component 2 */
buffer_load_short_d16 v[vgprG2LA+8+1], v[vgprG2LA+8+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_short_d16_hi v210, v210, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+8+1], v[vgprG2LA+8+1], v210 // HasEccHalf: pack
/* g2l=0, load component 4 */
buffer_load_short_d16 v[vgprG2LA+8+2], v[vgprG2LA+8+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_short_d16_hi v211, v211, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+8+2], v[vgprG2LA+8+2], v211 // HasEccHalf: pack
/* g2l=0, load component 6 */
buffer_load_short_d16 v[vgprG2LA+8+3], v[vgprG2LA+8+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_short_d16_hi v212, v212, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+8+3], v[vgprG2LA+8+3], v212 // HasEccHalf: pack
/* g2l=4, load component 0 */
buffer_load_short_d16 v[vgprG2LA+8+4], v[vgprG2LA+8+4], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_short_d16_hi v213, v213, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+8+4], v[vgprG2LA+8+4], v213 // HasEccHalf: pack
/* g2l=4, load component 2 */
buffer_load_short_d16 v[vgprG2LA+8+5], v[vgprG2LA+8+5], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 3 */
buffer_load_short_d16_hi v214, v214, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+8+5], v[vgprG2LA+8+5], v214 // HasEccHalf: pack
/* g2l=4, load component 4 */
buffer_load_short_d16 v[vgprG2LA+8+6], v[vgprG2LA+8+6], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 5 */
buffer_load_short_d16_hi v215, v215, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+8+6], v[vgprG2LA+8+6], v215 // HasEccHalf: pack
/* g2l=4, load component 6 */
buffer_load_short_d16 v[vgprG2LA+8+7], v[vgprG2LA+8+7], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 7 */
buffer_load_short_d16_hi v216, v216, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+8+7], v[vgprG2LA+8+7], v216 // HasEccHalf: pack




/* local write a */

ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4096 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 8192
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+8:vgprG2LA+8+3] offset:8192 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+12:vgprG2LA+12+3] offset:12288 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 8192

s_waitcnt lgkmcnt(0)
s_barrier //

v_mov_b32 v[vgprGlobalReadOffsetB+0], v[vgprGlobalReadOffsetB1+0]
v_mov_b32 v[vgprGlobalReadOffsetB+1], v[vgprGlobalReadOffsetB1+1]
v_mov_b32 v[vgprGlobalReadOffsetB+2], v[vgprGlobalReadOffsetB1+2]
v_mov_b32 v[vgprGlobalReadOffsetB+3], v[vgprGlobalReadOffsetB1+3]

v_and_b32 v217, 63, v[vgprSerial]                  // v217 = v[vgprSerial] % 64
v_lshrrev_b32 v218, 6, v[vgprSerial]               // v218 = v[vgprSerial] / 64

v_lshrrev_b32 v219, 4, v217                        // v219 = v217 / 16
v_and_b32 v220, 15, v217                           // v220 = v217 % 16

v_mov_b32 v[vgprG2LB+0], v219          // 
v_lshlrev_b32 v[vgprG2LB+0], 3, v[vgprG2LB+0]      // 
v_add_co_u32 v[vgprG2LB+0], vcc, 0, v[vgprG2LB+0]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+0], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+0], vcc, 0, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v[vgprG2LB+0], -0x1, v[vgprG2LB+0], s[64:65] // 
v_mov_b32 v201, v219                  // 
v_lshlrev_b32 v201, 3, v201                        // 
v_add_co_u32 v201, vcc, 1, v201                    // 
v_cmp_lt_u32 s[64:65], v201, s[sgprLoopCounterL]   // 
v_add_co_u32 v201, vcc, 2, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v201, -0x1, v201, s[64:65]           // 
v_mov_b32 v[vgprG2LB+1], v219          // 
v_lshlrev_b32 v[vgprG2LB+1], 3, v[vgprG2LB+1]      // 
v_add_co_u32 v[vgprG2LB+1], vcc, 2, v[vgprG2LB+1]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+1], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+1], vcc, 4, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v[vgprG2LB+1], -0x1, v[vgprG2LB+1], s[64:65] // 
v_mov_b32 v202, v219                   // 
v_lshlrev_b32 v202, 3, v202                        // 
v_add_co_u32 v202, vcc, 3, v202                    // 
v_cmp_lt_u32 s[64:65], v202, s[sgprLoopCounterL]   // 
v_add_co_u32 v202, vcc, 6, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v202, -0x1, v202, s[64:65]           // 
v_mov_b32 v[vgprG2LB+2], v219          // 
v_lshlrev_b32 v[vgprG2LB+2], 3, v[vgprG2LB+2]      // 
v_add_co_u32 v[vgprG2LB+2], vcc, 4, v[vgprG2LB+2]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+2], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+2], vcc, 8, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v[vgprG2LB+2], -0x1, v[vgprG2LB+2], s[64:65] // 
v_mov_b32 v203, v219                   // 
v_lshlrev_b32 v203, 3, v203                        // 
v_add_co_u32 v203, vcc, 5, v203                    // 
v_cmp_lt_u32 s[64:65], v203, s[sgprLoopCounterL]   // 
v_add_co_u32 v203, vcc, 10, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v203, -0x1, v203, s[64:65]           // 
v_mov_b32 v[vgprG2LB+3], v219          // 
v_lshlrev_b32 v[vgprG2LB+3], 3, v[vgprG2LB+3]      // 
v_add_co_u32 v[vgprG2LB+3], vcc, 6, v[vgprG2LB+3]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+3], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+3], vcc, 12, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v[vgprG2LB+3], -0x1, v[vgprG2LB+3], s[64:65] // 
v_mov_b32 v204, v219                   // 
v_lshlrev_b32 v204, 3, v204                        // 
v_add_co_u32 v204, vcc, 7, v204                    // 
v_cmp_lt_u32 s[64:65], v204, s[sgprLoopCounterL]   // 
v_add_co_u32 v204, vcc, 14, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v204, -0x1, v204, s[64:65]           // 


v_mov_b32 v[vgprG2LB+4], v219          // 
v_lshlrev_b32 v[vgprG2LB+4], 3, v[vgprG2LB+4]      // 
v_add_co_u32 v[vgprG2LB+4], vcc, 32, v[vgprG2LB+4]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+4], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+4], vcc, 0, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v[vgprG2LB+4], -0x1, v[vgprG2LB+4], s[64:65] // 
v_mov_b32 v205, v219                  // 
v_lshlrev_b32 v205, 3, v205                        // 
v_add_co_u32 v205, vcc, 33, v205                    // 
v_cmp_lt_u32 s[64:65], v205, s[sgprLoopCounterL]   // 
v_add_co_u32 v205, vcc, 2, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v205, -0x1, v205, s[64:65]           // 
v_mov_b32 v[vgprG2LB+5], v219          // 
v_lshlrev_b32 v[vgprG2LB+5], 3, v[vgprG2LB+5]      // 
v_add_co_u32 v[vgprG2LB+5], vcc, 34, v[vgprG2LB+5]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+5], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+5], vcc, 4, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v[vgprG2LB+5], -0x1, v[vgprG2LB+5], s[64:65] // 
v_mov_b32 v206, v219                   // 
v_lshlrev_b32 v206, 3, v206                        // 
v_add_co_u32 v206, vcc, 35, v206                    // 
v_cmp_lt_u32 s[64:65], v206, s[sgprLoopCounterL]   // 
v_add_co_u32 v206, vcc, 6, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v206, -0x1, v206, s[64:65]           // 
v_mov_b32 v[vgprG2LB+6], v219          // 
v_lshlrev_b32 v[vgprG2LB+6], 3, v[vgprG2LB+6]      // 
v_add_co_u32 v[vgprG2LB+6], vcc, 36, v[vgprG2LB+6]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+6], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+6], vcc, 8, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v[vgprG2LB+6], -0x1, v[vgprG2LB+6], s[64:65] // 
v_mov_b32 v207, v219                   // 
v_lshlrev_b32 v207, 3, v207                        // 
v_add_co_u32 v207, vcc, 37, v207                    // 
v_cmp_lt_u32 s[64:65], v207, s[sgprLoopCounterL]   // 
v_add_co_u32 v207, vcc, 10, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v207, -0x1, v207, s[64:65]           // 
v_mov_b32 v[vgprG2LB+7], v219          // 
v_lshlrev_b32 v[vgprG2LB+7], 3, v[vgprG2LB+7]      // 
v_add_co_u32 v[vgprG2LB+7], vcc, 38, v[vgprG2LB+7]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+7], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+7], vcc, 12, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v[vgprG2LB+7], -0x1, v[vgprG2LB+7], s[64:65] // 
v_mov_b32 v208, v219                   // 
v_lshlrev_b32 v208, 3, v208                        // 
v_add_co_u32 v208, vcc, 39, v208                    // 
v_cmp_lt_u32 s[64:65], v208, s[sgprLoopCounterL]   // 
v_add_co_u32 v208, vcc, 14, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v208, -0x1, v208, s[64:65]           // 


v_mov_b32 v[vgprG2LB+8], v219          // 
v_lshlrev_b32 v[vgprG2LB+8], 3, v[vgprG2LB+8]      // 
v_add_co_u32 v[vgprG2LB+8], vcc, 0, v[vgprG2LB+8]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+8], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+8], vcc, 0, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v[vgprG2LB+8], -0x1, v[vgprG2LB+8], s[64:65] // 
v_mov_b32 v209, v219                  // 
v_lshlrev_b32 v209, 3, v209                        // 
v_add_co_u32 v209, vcc, 1, v209                    // 
v_cmp_lt_u32 s[64:65], v209, s[sgprLoopCounterL]   // 
v_add_co_u32 v209, vcc, 2, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v209, -0x1, v209, s[64:65]           // 
v_mov_b32 v[vgprG2LB+9], v219          // 
v_lshlrev_b32 v[vgprG2LB+9], 3, v[vgprG2LB+9]      // 
v_add_co_u32 v[vgprG2LB+9], vcc, 2, v[vgprG2LB+9]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+9], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+9], vcc, 4, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v[vgprG2LB+9], -0x1, v[vgprG2LB+9], s[64:65] // 
v_mov_b32 v210, v219                   // 
v_lshlrev_b32 v210, 3, v210                        // 
v_add_co_u32 v210, vcc, 3, v210                    // 
v_cmp_lt_u32 s[64:65], v210, s[sgprLoopCounterL]   // 
v_add_co_u32 v210, vcc, 6, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v210, -0x1, v210, s[64:65]           // 
v_mov_b32 v[vgprG2LB+10], v219          // 
v_lshlrev_b32 v[vgprG2LB+10], 3, v[vgprG2LB+10]      // 
v_add_co_u32 v[vgprG2LB+10], vcc, 4, v[vgprG2LB+10]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+10], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+10], vcc, 8, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v[vgprG2LB+10], -0x1, v[vgprG2LB+10], s[64:65] // 
v_mov_b32 v211, v219                   // 
v_lshlrev_b32 v211, 3, v211                        // 
v_add_co_u32 v211, vcc, 5, v211                    // 
v_cmp_lt_u32 s[64:65], v211, s[sgprLoopCounterL]   // 
v_add_co_u32 v211, vcc, 10, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v211, -0x1, v211, s[64:65]           // 
v_mov_b32 v[vgprG2LB+11], v219          // 
v_lshlrev_b32 v[vgprG2LB+11], 3, v[vgprG2LB+11]      // 
v_add_co_u32 v[vgprG2LB+11], vcc, 6, v[vgprG2LB+11]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+11], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+11], vcc, 12, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v[vgprG2LB+11], -0x1, v[vgprG2LB+11], s[64:65] // 
v_mov_b32 v212, v219                   // 
v_lshlrev_b32 v212, 3, v212                        // 
v_add_co_u32 v212, vcc, 7, v212                    // 
v_cmp_lt_u32 s[64:65], v212, s[sgprLoopCounterL]   // 
v_add_co_u32 v212, vcc, 14, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v212, -0x1, v212, s[64:65]           // 


v_mov_b32 v[vgprG2LB+12], v219          // 
v_lshlrev_b32 v[vgprG2LB+12], 3, v[vgprG2LB+12]      // 
v_add_co_u32 v[vgprG2LB+12], vcc, 32, v[vgprG2LB+12]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+12], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+12], vcc, 0, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v[vgprG2LB+12], -0x1, v[vgprG2LB+12], s[64:65] // 
v_mov_b32 v213, v219                  // 
v_lshlrev_b32 v213, 3, v213                        // 
v_add_co_u32 v213, vcc, 33, v213                    // 
v_cmp_lt_u32 s[64:65], v213, s[sgprLoopCounterL]   // 
v_add_co_u32 v213, vcc, 2, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v213, -0x1, v213, s[64:65]           // 
v_mov_b32 v[vgprG2LB+13], v219          // 
v_lshlrev_b32 v[vgprG2LB+13], 3, v[vgprG2LB+13]      // 
v_add_co_u32 v[vgprG2LB+13], vcc, 34, v[vgprG2LB+13]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+13], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+13], vcc, 4, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v[vgprG2LB+13], -0x1, v[vgprG2LB+13], s[64:65] // 
v_mov_b32 v214, v219                   // 
v_lshlrev_b32 v214, 3, v214                        // 
v_add_co_u32 v214, vcc, 35, v214                    // 
v_cmp_lt_u32 s[64:65], v214, s[sgprLoopCounterL]   // 
v_add_co_u32 v214, vcc, 6, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v214, -0x1, v214, s[64:65]           // 
v_mov_b32 v[vgprG2LB+14], v219          // 
v_lshlrev_b32 v[vgprG2LB+14], 3, v[vgprG2LB+14]      // 
v_add_co_u32 v[vgprG2LB+14], vcc, 36, v[vgprG2LB+14]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+14], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+14], vcc, 8, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v[vgprG2LB+14], -0x1, v[vgprG2LB+14], s[64:65] // 
v_mov_b32 v215, v219                   // 
v_lshlrev_b32 v215, 3, v215                        // 
v_add_co_u32 v215, vcc, 37, v215                    // 
v_cmp_lt_u32 s[64:65], v215, s[sgprLoopCounterL]   // 
v_add_co_u32 v215, vcc, 10, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v215, -0x1, v215, s[64:65]           // 
v_mov_b32 v[vgprG2LB+15], v219          // 
v_lshlrev_b32 v[vgprG2LB+15], 3, v[vgprG2LB+15]      // 
v_add_co_u32 v[vgprG2LB+15], vcc, 38, v[vgprG2LB+15]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+15], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+15], vcc, 12, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v[vgprG2LB+15], -0x1, v[vgprG2LB+15], s[64:65] // 
v_mov_b32 v216, v219                   // 
v_lshlrev_b32 v216, 3, v216                        // 
v_add_co_u32 v216, vcc, 39, v216                    // 
v_cmp_lt_u32 s[64:65], v216, s[sgprLoopCounterL]   // 
v_add_co_u32 v216, vcc, 14, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v216, -0x1, v216, s[64:65]           // 


s_add_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 63 // tail add
s_lshr_b32 s[sgprLoopCounterL], s[sgprLoopCounterL], 6 // 


/* Update M0 for DTLDS */



/* Update M0 for DTLDS */


/* global read b */

/* g2l=0, load component 0 */
buffer_load_short_d16 v[vgprG2LB+0+0], v[vgprG2LB+0+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_short_d16_hi v201, v201, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v201 // HasEccHalf: pack
/* g2l=0, load component 2 */
buffer_load_short_d16 v[vgprG2LB+0+1], v[vgprG2LB+0+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_short_d16_hi v202, v202, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v202 // HasEccHalf: pack
/* g2l=0, load component 4 */
buffer_load_short_d16 v[vgprG2LB+0+2], v[vgprG2LB+0+2], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_short_d16_hi v203, v203, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+2], v[vgprG2LB+0+2], v203 // HasEccHalf: pack
/* g2l=0, load component 6 */
buffer_load_short_d16 v[vgprG2LB+0+3], v[vgprG2LB+0+3], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_short_d16_hi v204, v204, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+3], v[vgprG2LB+0+3], v204 // HasEccHalf: pack
/* g2l=4, load component 0 */
buffer_load_short_d16 v[vgprG2LB+4+0], v[vgprG2LB+4+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_short_d16_hi v205, v205, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+0], v[vgprG2LB+4+0], v205 // HasEccHalf: pack
/* g2l=4, load component 2 */
buffer_load_short_d16 v[vgprG2LB+4+1], v[vgprG2LB+4+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 3 */
buffer_load_short_d16_hi v206, v206, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+1], v[vgprG2LB+4+1], v206 // HasEccHalf: pack
/* g2l=4, load component 4 */
buffer_load_short_d16 v[vgprG2LB+4+2], v[vgprG2LB+4+2], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 5 */
buffer_load_short_d16_hi v207, v207, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+2], v[vgprG2LB+4+2], v207 // HasEccHalf: pack
/* g2l=4, load component 6 */
buffer_load_short_d16 v[vgprG2LB+4+3], v[vgprG2LB+4+3], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 7 */
buffer_load_short_d16_hi v208, v208, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+3], v[vgprG2LB+4+3], v208 // HasEccHalf: pack


/* g2l=0, load component 0 */
buffer_load_short_d16 v[vgprG2LB+8+0], v[vgprG2LB+8+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_short_d16_hi v209, v209, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+8+0], v[vgprG2LB+8+0], v209 // HasEccHalf: pack
/* g2l=0, load component 2 */
buffer_load_short_d16 v[vgprG2LB+8+1], v[vgprG2LB+8+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_short_d16_hi v210, v210, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+8+1], v[vgprG2LB+8+1], v210 // HasEccHalf: pack
/* g2l=0, load component 4 */
buffer_load_short_d16 v[vgprG2LB+8+2], v[vgprG2LB+8+2], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_short_d16_hi v211, v211, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+8+2], v[vgprG2LB+8+2], v211 // HasEccHalf: pack
/* g2l=0, load component 6 */
buffer_load_short_d16 v[vgprG2LB+8+3], v[vgprG2LB+8+3], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_short_d16_hi v212, v212, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+8+3], v[vgprG2LB+8+3], v212 // HasEccHalf: pack
/* g2l=4, load component 0 */
buffer_load_short_d16 v[vgprG2LB+12+0], v[vgprG2LB+12+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_short_d16_hi v213, v213, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+12+0], v[vgprG2LB+12+0], v213 // HasEccHalf: pack
/* g2l=4, load component 2 */
buffer_load_short_d16 v[vgprG2LB+12+1], v[vgprG2LB+12+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 3 */
buffer_load_short_d16_hi v214, v214, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+12+1], v[vgprG2LB+12+1], v214 // HasEccHalf: pack
/* g2l=4, load component 4 */
buffer_load_short_d16 v[vgprG2LB+12+2], v[vgprG2LB+12+2], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 5 */
buffer_load_short_d16_hi v215, v215, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+12+2], v[vgprG2LB+12+2], v215 // HasEccHalf: pack
/* g2l=4, load component 6 */
buffer_load_short_d16 v[vgprG2LB+12+3], v[vgprG2LB+12+3], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 7 */
buffer_load_short_d16_hi v216, v216, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+12+3], v[vgprG2LB+12+3], v216 // HasEccHalf: pack


s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=02wait for global read

s_barrier //

v_mov_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprG2LB+0:vgprG2LB+1]
v_mov_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprG2LB+2:vgprG2LB+3]
v_mov_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprG2LB+4:vgprG2LB+5]
v_mov_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprG2LB+6:vgprG2LB+7]
v_mov_b64 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+9], v[vgprG2LB+8:vgprG2LB+9]
v_mov_b64 v[vgprValuB_X0_I0+10:vgprValuB_X0_I0+11], v[vgprG2LB+10:vgprG2LB+11]
v_mov_b64 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+13], v[vgprG2LB+12:vgprG2LB+13]
v_mov_b64 v[vgprValuB_X0_I0+14:vgprValuB_X0_I0+15], v[vgprG2LB+14:vgprG2LB+15]

/* local write a */


/* local write b */


/* Recalc local read offsets */


s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-15wait for local write

s_barrier //


/* TailLoop: Check VGPR.checkin for INT8 LW */


/* local read reset offsets a */


/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrA], 0x7fff, v[vgprLocalReadAddrA] // reset Red,Blk -> Red


/* local read reset offsets b */


/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrB], 0x7fff, v[vgprLocalReadAddrB] // reset Red,Blk -> Red


/* local read init pointers a */


/* localReadInitPointers */


/* local read init pointers b */


/* localReadInitPointers */


/* tail loop: macs */

TailLoopBeginL_5:


/* local read a */

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x1000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x1200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x2000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x2200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x3000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x3200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x4000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x4200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x5000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x5200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x6000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x6200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x7000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x7200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0

s_waitcnt lgkmcnt(0)
MAC_32x4x2_X0_I0
s_barrier


ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x40 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x1040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x1240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x2040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x2240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x3040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x3240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x4040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x4240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x5040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x5240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x6040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x6240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x7040 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrA_forLdsWrapTailLoop] offset:0x7240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0

s_waitcnt lgkmcnt(0)
MAC_32x4x2_X0_I1
s_barrier



/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x1 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x1 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 TailLoopBeginL_5                    // restart LoopL
TailLoopEndL_6:

SkipTailLoopL_7:

Summation_End_23:
/* endSummation: add vgpr [128...198) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
.set ShadowLimitA, UNDEF
.set ShadowLimitB, UNDEF
.set LocalWriteAddrA, UNDEF
.set LocalWriteAddrB, UNDEF



/* not-LocalSplitU: global write indices */

/* Cache this rank tile's final-output row map in LDS after MMAC is done. */
v_cmp_lt_u32 vcc, v[vgprSerial], s97
s_and_saveexec_b64 s[54:55], vcc
_v_add_u32 v253, s93, v[vgprSerial]
v_lshlrev_b32 v253, 0x2, v253
global_load_dword v252, v253, s[94:95] glc
s_waitcnt vmcnt(0)
v_lshlrev_b32 v253, 0x2, v[vgprSerial]
ds_write_b32 v253, v252
s_mov_b64 exec, s[54:55]
s_waitcnt lgkmcnt(0)
s_barrier

/* computeStoreVgprs */
v_and_b32 v128, 63, v[vgprSerial]                  // v0 = v[vgprSerial] & 63
v_lshrrev_b32 v129, 6, v[vgprSerial]               // v4 = v[vgprSerial] / 64
v_and_b32 v132, 15, v128                           // v5 = v0 & 15
v_lshrrev_b32 v133, 4, v128                        // v3 = v0 / 16
v_and_b32 v134, 0, v129                            // v4&1
s_mov_b32 s54, 32                                 //  
v_mul_lo_u32 v134, s54, v134                       //  
v_lshlrev_b32 v128, 0, v132                        // ((v[vgprSerial]&63)&15)*2
_v_add_co_u32 v128, vcc, v128, v134                // row
v_lshrrev_b32 v134, 0, v129                        // v4>>1
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

                                           // 
s_mov_b32 s[sgprSrdC+0], s94                     // rowmap SRD base low for direct-output
s_mov_b32 s[sgprSrdC+1], s95                     // rowmap SRD base high
s_mov_b32 s[sgprSrdC+2], 0xfffffffe              // rowmap SRD limit
s_mov_b32 s[sgprSrdC+3], Srd127_96               // rowmap SRD format
s_mov_b32 s7, s93                                // shadow sorted_start for rowmap D macro
s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 GW_Beta_38                          // Branch if Beta is not zero
s_branch GW_B0_E1_29                              // direct-output: force vector-address edge write

s_and_b32 s54, 255, s[sgprSizeI]                   // s54 = s[sgprSizeI] % 256
s_add_u32 s56, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s56                // wg0 >= nwg0-1 ?
s_cselect_b32 s54, s54, 0                          // set rMT0
s_cmpk_gt_u32 s54, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B0_E1_29                         // jump if edges required
s_and_b32 s54, 255, s[sgprSizeJ]                   // s54 = s[sgprSizeJ] % 256
s_add_u32 s56, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s56                // wg1 >= nwg1-1
s_cselect_b32 s54, s54, 0                          // set rMT1
s_cmpk_gt_u32 s54, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B0_E1_29                         // jump if edges required
GW_B0_E0_26:

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
v_mov_b32 v250, 0x7fff0000                         // fp32 Nan
v_mov_b32 v251, 0x7fff                             // rounding bias for bfloat16

FP32_TO_BF16 vgprValuC+0       // convert C to bf16
buffer_store_short v0, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+1       // convert C to bf16
buffer_store_short v1, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+2       // convert C to bf16
buffer_store_short v2, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+3       // convert C to bf16
buffer_store_short v3, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+4       // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v4, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+5       // convert C to bf16
buffer_store_short v5, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+6       // convert C to bf16
buffer_store_short v6, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+7       // convert C to bf16
buffer_store_short v7, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+8       // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v8, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+9       // convert C to bf16
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
FP32_TO_BF16 vgprValuC+23    // convert C to bf16
buffer_store_short v23, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+24     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
s_mul_i32 s54, s[sgprStrideD1J], 32                // scale StrideD *= numRows(16) * bpe
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v80, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+81     // convert C to bf16
buffer_store_short v81, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+82    // convert C to bf16
buffer_store_short v82, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+83     // convert C to bf16
buffer_store_short v83, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+84    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v84, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+85     // convert C to bf16
buffer_store_short v85, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+86     // convert C to bf16
buffer_store_short v86, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+87     // convert C to bf16
buffer_store_short v87, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+88     // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
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
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v96, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+97     // convert C to bf16
buffer_store_short v97, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+98     // convert C to bf16
buffer_store_short v98, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+99     // convert C to bf16
buffer_store_short v99, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+100   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v100, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+101   // convert C to bf16
buffer_store_short v101, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+102   // convert C to bf16
buffer_store_short v102, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+103   // convert C to bf16
buffer_store_short v103, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+104   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v104, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+105   // convert C to bf16
buffer_store_short v105, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+106   // convert C to bf16
buffer_store_short v106, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+107   // convert C to bf16
buffer_store_short v107, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+108   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v108, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+109   // convert C to bf16
buffer_store_short v109, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+110   // convert C to bf16
buffer_store_short v110, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+111   // convert C to bf16
buffer_store_short v111, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+112   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v112, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+113   // convert C to bf16
buffer_store_short v113, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+114   // convert C to bf16
buffer_store_short v114, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+115   // convert C to bf16
buffer_store_short v115, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+116   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v116, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+117   // convert C to bf16
buffer_store_short v117, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+118   // convert C to bf16
buffer_store_short v118, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+119   // convert C to bf16
buffer_store_short v119, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+120   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v120, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+121   // convert C to bf16
buffer_store_short v121, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+122   // convert C to bf16
buffer_store_short v122, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+123   // convert C to bf16
buffer_store_short v123, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+124   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v124, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+125   // convert C to bf16
buffer_store_short v125, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+126   // convert C to bf16
buffer_store_short v126, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
FP32_TO_BF16 vgprValuC+127   // convert C to bf16
buffer_store_short v127, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
s_branch label_GW_End_37                           // jump to end
GW_B0_E1_29:

/* The 128 edge stores touch only eight logical output rows per lane.  Load
 * those row-map entries once, in parallel, and retain them in post-MMAC
 * scratch VGPRs for the complete epilogue. */
v_lshlrev_b32 v253, 0x2, v129
ds_read_b32 v224, v253 offset:0
ds_read_b32 v225, v253 offset:16
ds_read_b32 v226, v253 offset:32
ds_read_b32 v227, v253 offset:48
ds_read_b32 v228, v253 offset:64
ds_read_b32 v229, v253 offset:80
ds_read_b32 v230, v253 offset:96
ds_read_b32 v231, v253 offset:112
s_waitcnt lgkmcnt(0)
v_mov_b32 v253, 0x80000000
v_cmp_gt_i32 vcc, v224, -1
v_mul_lo_u32 v224, s[sgprStrideD1J], v224
v_cndmask_b32 v224, v253, v224, vcc
v_cmp_gt_i32 vcc, v225, -1
v_mul_lo_u32 v225, s[sgprStrideD1J], v225
v_cndmask_b32 v225, v253, v225, vcc
v_cmp_gt_i32 vcc, v226, -1
v_mul_lo_u32 v226, s[sgprStrideD1J], v226
v_cndmask_b32 v226, v253, v226, vcc
v_cmp_gt_i32 vcc, v227, -1
v_mul_lo_u32 v227, s[sgprStrideD1J], v227
v_cndmask_b32 v227, v253, v227, vcc
v_cmp_gt_i32 vcc, v228, -1
v_mul_lo_u32 v228, s[sgprStrideD1J], v228
v_cndmask_b32 v228, v253, v228, vcc
v_cmp_gt_i32 vcc, v229, -1
v_mul_lo_u32 v229, s[sgprStrideD1J], v229
v_cndmask_b32 v229, v253, v229, vcc
v_cmp_gt_i32 vcc, v230, -1
v_mul_lo_u32 v230, s[sgprStrideD1J], v230
v_cndmask_b32 v230, v253, v230, vcc
v_cmp_gt_i32 vcc, v231, -1
v_mul_lo_u32 v231, s[sgprStrideD1J], v231
v_cndmask_b32 v231, v253, v231, vcc

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
GLOBAL_ROWMAP_D_ADDR_REUSE 134, 128, 60, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 135, 128, 62, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 136, 128, 64, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 137, 128, 66, 227 // direct-output D rowmap CTA cache
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,4) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 138, 132, 68, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,5) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 139, 132, 70, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,6) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 140, 132, 72, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,7) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 141, 132, 74, 227 // direct-output D rowmap CTA cache
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 142, 132, 76, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 143, 132, 78, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 144, 132, 80, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 145, 132, 82, 227 // direct-output D rowmap CTA cache
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,4) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 146, 132, 84, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,5) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 147, 132, 86, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,6) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 148, 132, 88, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,7) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 149, 132, 90, 227 // direct-output D rowmap CTA cache
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 150, 132, 92, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 151, 132, 94, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 152, 132, 96, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 153, 132, 98, 227 // direct-output D rowmap CTA cache
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
v_mov_b32 v250, 0x7fff0000                         // fp32 Nan
v_mov_b32 v251, 0x7fff                             // rounding bias for bfloat16

FP32_TO_BF16 vgprValuC+0       // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 0, 134, 60 // store D
FP32_TO_BF16 vgprValuC+1       // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 1, 135, 62 // store D
FP32_TO_BF16 vgprValuC+2       // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 2, 136, 64 // store D
FP32_TO_BF16 vgprValuC+3       // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 3, 137, 66 // store D
FP32_TO_BF16 vgprValuC+4       // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 4, 138, 68 // store D
FP32_TO_BF16 vgprValuC+5       // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 5, 139, 70 // store D
FP32_TO_BF16 vgprValuC+6       // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 6, 140, 72 // store D
FP32_TO_BF16 vgprValuC+7       // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 7, 141, 74 // store D
FP32_TO_BF16 vgprValuC+8       // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 8, 142, 76 // store D
FP32_TO_BF16 vgprValuC+9       // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 9, 143, 78 // store D
FP32_TO_BF16 vgprValuC+10     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 10, 144, 80 // store D
FP32_TO_BF16 vgprValuC+11     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 11, 145, 82 // store D
FP32_TO_BF16 vgprValuC+12     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 12, 146, 84 // store D
FP32_TO_BF16 vgprValuC+13     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 13, 147, 86 // store D
FP32_TO_BF16 vgprValuC+14     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 14, 148, 88 // store D
FP32_TO_BF16 vgprValuC+15     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 15, 149, 90 // store D
FP32_TO_BF16 vgprValuC+16     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 16, 150, 92 // store D
FP32_TO_BF16 vgprValuC+17     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 17, 151, 94 // store D
FP32_TO_BF16 vgprValuC+18     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 18, 152, 96 // store D
FP32_TO_BF16 vgprValuC+19     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 19, 153, 98 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,1,4:vw1); (0,0,1,5:vw1); (0,0,1,6:vw1); (0,0,1,7:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,1,1,2:vw1); (0,1,1,3:vw1); (0,1,1,4:vw1); (0,1,1,5:vw1); (0,1,1,6:vw1); (0,1,1,7:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,2,4:vw1); (0,0,2,5:vw1); (0,0,2,6:vw1); (0,0,2,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,1,0,4) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 134, 132, 60, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,5) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 135, 132, 62, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,6) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 136, 132, 64, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,7) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 137, 132, 66, 227 // direct-output D rowmap CTA cache
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 138, 132, 68, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,1) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=1
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 139, 132, 70, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,2) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=2
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 140, 132, 72, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,3) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=3
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 141, 132, 74, 227 // direct-output D rowmap CTA cache
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,4) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 142, 132, 76, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,5) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 143, 132, 78, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,6) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 144, 132, 80, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,7) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 145, 132, 82, 227 // direct-output D rowmap CTA cache
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
s_mov_b32 s54, 128                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 146, 132, 84, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 147, 132, 86, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 148, 132, 88, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 149, 132, 90, 227 // direct-output D rowmap CTA cache
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,4) */
s_mov_b32 s54, 144                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 150, 132, 92, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,5) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 151, 132, 94, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,6) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 152, 132, 96, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,7) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 153, 132, 98, 227 // direct-output D rowmap CTA cache
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
FP32_TO_BF16 vgprValuC+20     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 20, 134, 60 // store D
FP32_TO_BF16 vgprValuC+21     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 21, 135, 62 // store D
FP32_TO_BF16 vgprValuC+22     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 22, 136, 64 // store D
FP32_TO_BF16 vgprValuC+23     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 23, 137, 66 // store D
FP32_TO_BF16 vgprValuC+24     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 24, 138, 68 // store D
FP32_TO_BF16 vgprValuC+25     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 25, 139, 70 // store D
FP32_TO_BF16 vgprValuC+26     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 26, 140, 72 // store D
FP32_TO_BF16 vgprValuC+27     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 27, 141, 74 // store D
FP32_TO_BF16 vgprValuC+28     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 28, 142, 76 // store D
FP32_TO_BF16 vgprValuC+29     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 29, 143, 78 // store D
FP32_TO_BF16 vgprValuC+30     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 30, 144, 80 // store D
FP32_TO_BF16 vgprValuC+31     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 31, 145, 82 // store D
FP32_TO_BF16 vgprValuC+32     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 32, 146, 84 // store D
FP32_TO_BF16 vgprValuC+33     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 33, 147, 86 // store D
FP32_TO_BF16 vgprValuC+34     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 34, 148, 88 // store D
FP32_TO_BF16 vgprValuC+35     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 35, 149, 90 // store D
FP32_TO_BF16 vgprValuC+36     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 36, 150, 92 // store D
FP32_TO_BF16 vgprValuC+37     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 37, 151, 94 // store D
FP32_TO_BF16 vgprValuC+38     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 38, 152, 96 // store D
FP32_TO_BF16 vgprValuC+39    // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 39, 153, 98 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,1,2,0:vw1); (0,1,2,1:vw1); (0,1,2,2:vw1); (0,1,2,3:vw1); (0,1,2,4:vw1); (0,1,2,5:vw1); (0,1,2,6:vw1); (0,1,2,7:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,0,3,4:vw1); (0,0,3,5:vw1); (0,0,3,6:vw1); (0,0,3,7:vw1); (0,1,3,0:vw1); (0,1,3,1:vw1); (0,1,3,2:vw1); (0,1,3,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
s_mov_b32 s54, 160                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 134, 132, 60, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,1) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 135, 132, 62, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,2) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 136, 132, 64, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,3) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 137, 132, 66, 227 // direct-output D rowmap CTA cache
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,4) */
s_mov_b32 s54, 176                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 138, 132, 68, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,5) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 139, 132, 70, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,6) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 140, 132, 72, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,7) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 141, 132, 74, 227 // direct-output D rowmap CTA cache
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 142, 132, 76, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 143, 132, 78, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 144, 132, 80, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 145, 132, 82, 227 // direct-output D rowmap CTA cache
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,4) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=0 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 146, 132, 84, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,5) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 147, 132, 86, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,6) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 148, 132, 88, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,7) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 149, 132, 90, 227 // direct-output D rowmap CTA cache
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
s_mov_b32 s54, 224                                  // coordOffset0 d0=1 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 150, 132, 92, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,1) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 151, 132, 94, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,2) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 152, 132, 96, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,3) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 153, 132, 98, 227 // direct-output D rowmap CTA cache
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
FP32_TO_BF16 vgprValuC+40     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 40, 134, 60 // store D
FP32_TO_BF16 vgprValuC+41     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 41, 135, 62 // store D
FP32_TO_BF16 vgprValuC+42     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 42, 136, 64 // store D
FP32_TO_BF16 vgprValuC+43     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 43, 137, 66 // store D
FP32_TO_BF16 vgprValuC+44     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 44, 138, 68 // store D
FP32_TO_BF16 vgprValuC+45     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 45, 139, 70 // store D
FP32_TO_BF16 vgprValuC+46     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 46, 140, 72 // store D
FP32_TO_BF16 vgprValuC+47     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 47, 141, 74 // store D
FP32_TO_BF16 vgprValuC+48     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 48, 142, 76 // store D
FP32_TO_BF16 vgprValuC+49     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 49, 143, 78 // store D
FP32_TO_BF16 vgprValuC+50     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 50, 144, 80 // store D
FP32_TO_BF16 vgprValuC+51     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 51, 145, 82 // store D
FP32_TO_BF16 vgprValuC+52     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 52, 146, 84 // store D
FP32_TO_BF16 vgprValuC+53     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 53, 147, 86 // store D
FP32_TO_BF16 vgprValuC+54     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 54, 148, 88 // store D
FP32_TO_BF16 vgprValuC+55     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 55, 149, 90 // store D
FP32_TO_BF16 vgprValuC+56     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 56, 150, 92 // store D
FP32_TO_BF16 vgprValuC+57     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 57, 151, 94 // store D
FP32_TO_BF16 vgprValuC+58     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 58, 152, 96 // store D
FP32_TO_BF16 vgprValuC+59     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 59, 153, 98 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (0,1,3,4:vw1); (0,1,3,5:vw1); (0,1,3,6:vw1); (0,1,3,7:vw1); (0,0,4,0:vw1); (0,0,4,1:vw1); (0,0,4,2:vw1); (0,0,4,3:vw1); (0,0,4,4:vw1); (0,0,4,5:vw1); (0,0,4,6:vw1); (0,0,4,7:vw1); (0,1,4,0:vw1); (0,1,4,1:vw1); (0,1,4,2:vw1); (0,1,4,3:vw1); (0,1,4,4:vw1); (0,1,4,5:vw1); (0,1,4,6:vw1); (0,1,4,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,3,1,4) */
s_mov_b32 s54, 240                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 134, 132, 60, 224 // direct-output D rowmap CTA cache
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,5) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 135, 132, 62, 225 // direct-output D rowmap CTA cache
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,6) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 136, 132, 64, 226 // direct-output D rowmap CTA cache
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,7) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 137, 132, 66, 227 // direct-output D rowmap CTA cache
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset


/* (d1,vc1,d0,vc0)=(0,4,0,0) */
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 138, 128, 68, 228 // direct-output D rowmap CTA cache
s_mul_i32 s[sgprEdgeCheck+1], s[sgprStrideD1J], 32 // add offset stride
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,1) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 139, 128, 70, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,2) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 140, 128, 72, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,3) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 141, 128, 74, 231 // direct-output D rowmap CTA cache
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,4) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 142, 132, 76, 228 // direct-output D rowmap CTA cache
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,5) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 143, 132, 78, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,6) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 144, 132, 80, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,7) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 145, 132, 82, 231 // direct-output D rowmap CTA cache
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 146, 132, 84, 228 // direct-output D rowmap CTA cache
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,1) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 147, 132, 86, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,2) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 148, 132, 88, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,3) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 149, 132, 90, 231 // direct-output D rowmap CTA cache
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,4) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 150, 132, 92, 228 // direct-output D rowmap CTA cache
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,5) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 151, 132, 94, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,6) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 152, 132, 96, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,7) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 153, 132, 98, 231 // direct-output D rowmap CTA cache
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
FP32_TO_BF16 vgprValuC+60     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 60, 134, 60 // store D
FP32_TO_BF16 vgprValuC+61     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 61, 135, 62 // store D
FP32_TO_BF16 vgprValuC+62     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 62, 136, 64 // store D
FP32_TO_BF16 vgprValuC+63     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 63, 137, 66 // store D
FP32_TO_BF16 vgprValuC+64     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 64, 138, 68 // store D
FP32_TO_BF16 vgprValuC+65     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 65, 139, 70 // store D
FP32_TO_BF16 vgprValuC+66     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 66, 140, 72 // store D
FP32_TO_BF16 vgprValuC+67     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 67, 141, 74 // store D
FP32_TO_BF16 vgprValuC+68     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 68, 142, 76 // store D
FP32_TO_BF16 vgprValuC+69     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 69, 143, 78 // store D
FP32_TO_BF16 vgprValuC+70     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 70, 144, 80 // store D
FP32_TO_BF16 vgprValuC+71     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 71, 145, 82 // store D
FP32_TO_BF16 vgprValuC+72     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 72, 146, 84 // store D
FP32_TO_BF16 vgprValuC+73     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 73, 147, 86 // store D
FP32_TO_BF16 vgprValuC+74     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 74, 148, 88 // store D
FP32_TO_BF16 vgprValuC+75     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 75, 149, 90 // store D
FP32_TO_BF16 vgprValuC+76     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 76, 150, 92 // store D
FP32_TO_BF16 vgprValuC+77     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 77, 151, 94 // store D
FP32_TO_BF16 vgprValuC+78     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 78, 152, 96 // store D
FP32_TO_BF16 vgprValuC+79     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 79, 153, 98 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #4 (d1,d0,vc1,vc0) = */
/*    (0,0,5,0:vw1); (0,0,5,1:vw1); (0,0,5,2:vw1); (0,0,5,3:vw1); (0,0,5,4:vw1); (0,0,5,5:vw1); (0,0,5,6:vw1); (0,0,5,7:vw1); (0,1,5,0:vw1); (0,1,5,1:vw1); (0,1,5,2:vw1); (0,1,5,3:vw1); (0,1,5,4:vw1); (0,1,5,5:vw1); (0,1,5,6:vw1); (0,1,5,7:vw1); (0,0,6,0:vw1); (0,0,6,1:vw1); (0,0,6,2:vw1); (0,0,6,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 134, 132, 60, 228 // direct-output D rowmap CTA cache
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,1) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 135, 132, 62, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,2) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 136, 132, 64, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,3) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 137, 132, 66, 231 // direct-output D rowmap CTA cache
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,4) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 138, 132, 68, 228 // direct-output D rowmap CTA cache
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,5) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 139, 132, 70, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,6) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 140, 132, 72, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,7) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 141, 132, 74, 231 // direct-output D rowmap CTA cache
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,0) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 142, 132, 76, 228 // direct-output D rowmap CTA cache
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,1) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=1
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 143, 132, 78, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,2) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=2
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 144, 132, 80, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,3) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=3
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 145, 132, 82, 231 // direct-output D rowmap CTA cache
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,4) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 146, 132, 84, 228 // direct-output D rowmap CTA cache
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,5) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 147, 132, 86, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,6) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 148, 132, 88, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,7) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 149, 132, 90, 231 // direct-output D rowmap CTA cache
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
s_mov_b32 s54, 128                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 150, 132, 92, 228 // direct-output D rowmap CTA cache
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,1) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 151, 132, 94, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,2) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 152, 132, 96, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,3) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 153, 132, 98, 231 // direct-output D rowmap CTA cache
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
FP32_TO_BF16 vgprValuC+80     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 80, 134, 60 // store D
FP32_TO_BF16 vgprValuC+81     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 81, 135, 62 // store D
FP32_TO_BF16 vgprValuC+82     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 82, 136, 64 // store D
FP32_TO_BF16 vgprValuC+83     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 83, 137, 66 // store D
FP32_TO_BF16 vgprValuC+84     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 84, 138, 68 // store D
FP32_TO_BF16 vgprValuC+85     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 85, 139, 70 // store D
FP32_TO_BF16 vgprValuC+86     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 86, 140, 72 // store D
FP32_TO_BF16 vgprValuC+87     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 87, 141, 74 // store D
FP32_TO_BF16 vgprValuC+88     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 88, 142, 76 // store D
FP32_TO_BF16 vgprValuC+89     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 89, 143, 78 // store D
FP32_TO_BF16 vgprValuC+90     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 90, 144, 80 // store D
FP32_TO_BF16 vgprValuC+91     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 91, 145, 82 // store D
FP32_TO_BF16 vgprValuC+92     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 92, 146, 84 // store D
FP32_TO_BF16 vgprValuC+93     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 93, 147, 86 // store D
FP32_TO_BF16 vgprValuC+94     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 94, 148, 88 // store D
FP32_TO_BF16 vgprValuC+95     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 95, 149, 90 // store D
FP32_TO_BF16 vgprValuC+96     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 96, 150, 92 // store D
FP32_TO_BF16 vgprValuC+97     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 97, 151, 94 // store D
FP32_TO_BF16 vgprValuC+98     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 98, 152, 96 // store D
FP32_TO_BF16 vgprValuC+99     // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 99, 153, 98 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #5 (d1,d0,vc1,vc0) = */
/*    (0,0,6,4:vw1); (0,0,6,5:vw1); (0,0,6,6:vw1); (0,0,6,7:vw1); (0,1,6,0:vw1); (0,1,6,1:vw1); (0,1,6,2:vw1); (0,1,6,3:vw1); (0,1,6,4:vw1); (0,1,6,5:vw1); (0,1,6,6:vw1); (0,1,6,7:vw1); (0,0,7,0:vw1); (0,0,7,1:vw1); (0,0,7,2:vw1); (0,0,7,3:vw1); (0,0,7,4:vw1); (0,0,7,5:vw1); (0,0,7,6:vw1); (0,0,7,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,6,0,4) */
s_mov_b32 s54, 144                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 134, 132, 60, 228 // direct-output D rowmap CTA cache
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,5) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 135, 132, 62, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,6) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 136, 132, 64, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,7) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 137, 132, 66, 231 // direct-output D rowmap CTA cache
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,0) */
s_mov_b32 s54, 160                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 138, 132, 68, 228 // direct-output D rowmap CTA cache
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,1) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 139, 132, 70, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,2) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 140, 132, 72, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,3) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 141, 132, 74, 231 // direct-output D rowmap CTA cache
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,4) */
s_mov_b32 s54, 176                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 142, 132, 76, 228 // direct-output D rowmap CTA cache
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,5) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 143, 132, 78, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,6) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 144, 132, 80, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,7) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 145, 132, 82, 231 // direct-output D rowmap CTA cache
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 146, 132, 84, 228 // direct-output D rowmap CTA cache
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,1) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 147, 132, 86, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,2) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 148, 132, 88, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,3) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 149, 132, 90, 231 // direct-output D rowmap CTA cache
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,4) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=0 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 150, 132, 92, 228 // direct-output D rowmap CTA cache
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,5) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 151, 132, 94, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,6) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 152, 132, 96, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,7) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 153, 132, 98, 231 // direct-output D rowmap CTA cache
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
FP32_TO_BF16 vgprValuC+100   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 100, 134, 60 // store D
FP32_TO_BF16 vgprValuC+101   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 101, 135, 62 // store D
FP32_TO_BF16 vgprValuC+102   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 102, 136, 64 // store D
FP32_TO_BF16 vgprValuC+103   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 103, 137, 66 // store D
FP32_TO_BF16 vgprValuC+104   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 104, 138, 68 // store D
FP32_TO_BF16 vgprValuC+105   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 105, 139, 70 // store D
FP32_TO_BF16 vgprValuC+106   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 106, 140, 72 // store D
FP32_TO_BF16 vgprValuC+107   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 107, 141, 74 // store D
FP32_TO_BF16 vgprValuC+108   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 108, 142, 76 // store D
FP32_TO_BF16 vgprValuC+109   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 109, 143, 78 // store D
FP32_TO_BF16 vgprValuC+110   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 110, 144, 80 // store D
FP32_TO_BF16 vgprValuC+111   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 111, 145, 82 // store D
FP32_TO_BF16 vgprValuC+112   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 112, 146, 84 // store D
FP32_TO_BF16 vgprValuC+113   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 113, 147, 86 // store D
FP32_TO_BF16 vgprValuC+114   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 114, 148, 88 // store D
FP32_TO_BF16 vgprValuC+115   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 115, 149, 90 // store D
FP32_TO_BF16 vgprValuC+116  // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 116, 150, 92 // store D
FP32_TO_BF16 vgprValuC+117   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 117, 151, 94 // store D
FP32_TO_BF16 vgprValuC+118   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 118, 152, 96 // store D
FP32_TO_BF16 vgprValuC+119   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 119, 153, 98 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #6 (d1,d0,vc1,vc0) = */
/*    (0,1,7,0:vw1); (0,1,7,1:vw1); (0,1,7,2:vw1); (0,1,7,3:vw1); (0,1,7,4:vw1); (0,1,7,5:vw1); (0,1,7,6:vw1); (0,1,7,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,7,1,0) */
s_mov_b32 s54, 224                                  // coordOffset0 d0=1 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 134, 132, 60, 228 // direct-output D rowmap CTA cache
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,1) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 135, 132, 62, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,2) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 136, 132, 64, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,3) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 137, 132, 66, 231 // direct-output D rowmap CTA cache
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,4) */
s_mov_b32 s54, 240                                  // coordOffset0 d0=1 vc0=3
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 138, 132, 68, 228 // direct-output D rowmap CTA cache
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,5) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 139, 132, 70, 229 // direct-output D rowmap CTA cache
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,6) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 140, 132, 72, 230 // direct-output D rowmap CTA cache
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,7) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
GLOBAL_ROWMAP_D_ADDR_REUSE 141, 132, 74, 231 // direct-output D rowmap CTA cache
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
FP32_TO_BF16 vgprValuC+120   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 120, 134, 60 // store D
FP32_TO_BF16 vgprValuC+121   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 121, 135, 62 // store D
FP32_TO_BF16 vgprValuC+122   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 122, 136, 64 // store D
FP32_TO_BF16 vgprValuC+123   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 123, 137, 66 // store D
FP32_TO_BF16 vgprValuC+124   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 124, 138, 68 // store D
FP32_TO_BF16 vgprValuC+125   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 125, 139, 70 // store D
FP32_TO_BF16 vgprValuC+126   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 126, 140, 72 // store D
FP32_TO_BF16 vgprValuC+127   // convert C to bf16
BUFFER_STORE_SHORT_D_CLIPPED 127, 141, 74 // store D
s_branch label_GW_End_37                           // jump to end
GW_Beta_38:
s_and_b32 s54, 255, s[sgprSizeI]                   // s54 = s[sgprSizeI] % 256
s_add_u32 s56, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s56                // wg0 >= nwg0-1 ?
s_cselect_b32 s54, s54, 0                          // set rMT0
s_cmpk_gt_u32 s54, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B1_E1_36                         // jump if edges required
s_and_b32 s54, 255, s[sgprSizeJ]                   // s54 = s[sgprSizeJ] % 256
s_add_u32 s56, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s56                // wg1 >= nwg1-1
s_cselect_b32 s54, s54, 0                          // set rMT1
s_cmpk_gt_u32 s54, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B1_E1_36                         // jump if edges required
GW_B1_E0_33:

/* edge=0, allocate 2 sgpr. perBatch=2 perElement=0 elementsPerBatch=80 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSharedMask=1 optSrdIncForRow=1 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,0,4:vw1); (0,0,0,5:vw1); (0,0,0,6:vw1); (0,0,0,7:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1); (0,1,0,4:vw1); (0,1,0,5:vw1); (0,1,0,6:vw1); (0,1,0,7:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,0,1,4:vw1); (0,0,1,5:vw1); (0,0,1,6:vw1); (0,0,1,7:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,1,1,2:vw1); (0,1,1,3:vw1); (0,1,1,4:vw1); (0,1,1,5:vw1); (0,1,1,6:vw1); (0,1,1,7:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,2,4:vw1); (0,0,2,5:vw1); (0,0,2,6:vw1); (0,0,2,7:vw1); (0,1,2,0:vw1); (0,1,2,1:vw1); (0,1,2,2:vw1); (0,1,2,3:vw1); (0,1,2,4:vw1); (0,1,2,5:vw1); (0,1,2,6:vw1); (0,1,2,7:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,0,3,4:vw1); (0,0,3,5:vw1); (0,0,3,6:vw1); (0,0,3,7:vw1); (0,1,3,0:vw1); (0,1,3,1:vw1); (0,1,3,2:vw1); (0,1,3,3:vw1); (0,1,3,4:vw1); (0,1,3,5:vw1); (0,1,3,6:vw1); (0,1,3,7:vw1); (0,0,4,0:vw1); (0,0,4,1:vw1); (0,0,4,2:vw1); (0,0,4,3:vw1); (0,0,4,4:vw1); (0,0,4,5:vw1); (0,0,4,6:vw1); (0,0,4,7:vw1); (0,1,4,0:vw1); (0,1,4,1:vw1); (0,1,4,2:vw1); (0,1,4,3:vw1); (0,1,4,4:vw1); (0,1,4,5:vw1); (0,1,4,6:vw1); (0,1,4,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
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
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
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
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
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
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
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
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
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
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
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
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
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
buffer_load_short_d16 v188, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,0,6) */
buffer_load_short_d16 v189, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,0,7) */
buffer_load_short_d16 v190, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v191, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,1) */
buffer_load_short_d16 v192, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,2) */
buffer_load_short_d16 v193, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,3) */
buffer_load_short_d16 v194, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v195, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,5) */
buffer_load_short_d16 v196, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,6) */
buffer_load_short_d16 v197, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,3,1,7) */
buffer_load_short_d16 v201, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
_v_add_lshl_u32 v134, v130, v128, 0x1              // 
s_mul_i32 s54, s[sgprStrideC1J], 32                // scale StrideC *= numRows(16) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v202, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,1) */
buffer_load_short_d16 v203, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,2) */
buffer_load_short_d16 v204, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,3) */
buffer_load_short_d16 v205, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v206, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,5) */
buffer_load_short_d16 v207, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,6) */
buffer_load_short_d16 v208, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,0,7) */
buffer_load_short_d16 v209, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,0) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v210, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,1) */
buffer_load_short_d16 v211, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,2) */
buffer_load_short_d16 v212, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,3) */
buffer_load_short_d16 v213, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v214, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,5) */
buffer_load_short_d16 v215, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,6) */
buffer_load_short_d16 v216, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,4,1,7) */
buffer_load_short_d16 v217, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v128, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128

/* rC *= alpha batchEements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 0, 4), (0, 0, 0, 5), (0, 0, 0, 6), (0, 0, 0, 7), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (0, 1, 0, 4), (0, 1, 0, 5), (0, 1, 0, 6), (0, 1, 0, 7), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 0, 1, 4), (0, 0, 1, 5), (0, 0, 1, 6), (0, 0, 1, 7), (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 1, 3), (0, 1, 1, 4), (0, 1, 1, 5), (0, 1, 1, 6), (0, 1, 1, 7), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 2, 4), (0, 0, 2, 5), (0, 0, 2, 6), (0, 0, 2, 7), (0, 1, 2, 0), (0, 1, 2, 1), (0, 1, 2, 2), (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 5), (0, 1, 2, 6), (0, 1, 2, 7), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 0, 3, 4), (0, 0, 3, 5), (0, 0, 3, 6), (0, 0, 3, 7), (0, 1, 3, 0), (0, 1, 3, 1), (0, 1, 3, 2), (0, 1, 3, 3), (0, 1, 3, 4), (0, 1, 3, 5), (0, 1, 3, 6), (0, 1, 3, 7), (0, 0, 4, 0), (0, 0, 4, 1), (0, 0, 4, 2), (0, 0, 4, 3), (0, 0, 4, 4), (0, 0, 4, 5), (0, 0, 4, 6), (0, 0, 4, 7), (0, 1, 4, 0), (0, 1, 4, 1), (0, 1, 4, 2), (0, 1, 4, 3), (0, 1, 4, 4), (0, 1, 4, 5), (0, 1, 4, 6), (0, 1, 4, 7)] */
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

/* apply mask, calc new C and issue writes */
v_mov_b32 v250, 0x7fff0000                         // fp32 Nan
v_mov_b32 v251, 0x7fff                             // rounding bias for bfloat16

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 0 + 0 - 1
v_lshlrev_b32 v252, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+0], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+0                                    // convert C to bf16
buffer_store_short v0, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 1 + 1 - 1
v_lshlrev_b32 v252, 16, v136                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+1], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+1                                    // convert C to bf16
buffer_store_short v1, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 2 + 2 - 1
v_lshlrev_b32 v252, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+2], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+2                                    // convert C to bf16
buffer_store_short v2, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 3 + 3 - 1
v_lshlrev_b32 v252, 16, v138                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+3], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+3                                    // convert C to bf16
buffer_store_short v3, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 4 + 4 - 1
v_lshlrev_b32 v252, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+4], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+4                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v4, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 5 + 5 - 1
v_lshlrev_b32 v252, 16, v140                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+5], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+5                                    // convert C to bf16
buffer_store_short v5, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 6 + 6 - 1
v_lshlrev_b32 v252, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+6], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+6                                    // convert C to bf16
buffer_store_short v6, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 7 + 7 - 1
v_lshlrev_b32 v252, 16, v142                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+7], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+7                                    // convert C to bf16
buffer_store_short v7, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 8 + 8 - 1
v_lshlrev_b32 v252, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+8], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+8                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v8, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 9 + 9 - 1
v_lshlrev_b32 v252, 16, v144                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+9], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+9                                    // convert C to bf16
buffer_store_short v9, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 10 + 10 - 1
v_lshlrev_b32 v252, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+10], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+10                                    // convert C to bf16
buffer_store_short v10, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 11 + 11 - 1
v_lshlrev_b32 v252, 16, v146                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+11], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+11                                    // convert C to bf16
buffer_store_short v11, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 12 + 12 - 1
v_lshlrev_b32 v252, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+12], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+12                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v12, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 13 + 13 - 1
v_lshlrev_b32 v252, 16, v148                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+13], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+13                                    // convert C to bf16
buffer_store_short v13, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 14 + 14 - 1
v_lshlrev_b32 v252, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+14], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+14                                    // convert C to bf16
buffer_store_short v14, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 15 + 15 - 1
v_lshlrev_b32 v252, 16, v150                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+15], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+15                                    // convert C to bf16
buffer_store_short v15, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 16 + 16 - 1
v_lshlrev_b32 v252, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+16], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+16                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v16, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 17 + 17 - 1
v_lshlrev_b32 v252, 16, v152                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+17], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+17                                    // convert C to bf16
buffer_store_short v17, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 18 + 18 - 1
v_lshlrev_b32 v252, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+18], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+18                                    // convert C to bf16
buffer_store_short v18, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 19 + 19 - 1
v_lshlrev_b32 v252, 16, v154                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+19], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+19                                    // convert C to bf16
buffer_store_short v19, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 20 + 20 - 1
v_lshlrev_b32 v252, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+20], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+20                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v20, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 21 + 21 - 1
v_lshlrev_b32 v252, 16, v156                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+21], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+21                                    // convert C to bf16
buffer_store_short v21, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 22 + 22 - 1
v_lshlrev_b32 v252, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+22], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+22                                    // convert C to bf16
buffer_store_short v22, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 23 + 23 - 1
v_lshlrev_b32 v252, 16, v158                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+23], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+23                                    // convert C to bf16
buffer_store_short v23, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 24 + 24 - 1
v_lshlrev_b32 v252, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+24], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+24                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v24, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 25 + 25 - 1
v_lshlrev_b32 v252, 16, v160                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+25], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+25                                    // convert C to bf16
buffer_store_short v25, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 26 + 26 - 1
v_lshlrev_b32 v252, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+26], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+26                                    // convert C to bf16
buffer_store_short v26, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 27 + 27 - 1
v_lshlrev_b32 v252, 16, v162                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+27], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+27                                    // convert C to bf16
buffer_store_short v27, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 28 + 28 - 1
v_lshlrev_b32 v252, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+28], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+28                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v28, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 29 + 29 - 1
v_lshlrev_b32 v252, 16, v164                        // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+29], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+29                                    // convert C to bf16
buffer_store_short v29, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 30 + 30 - 1
v_lshlrev_b32 v252, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+30], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+30                                    // convert C to bf16
buffer_store_short v30, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 31 + 31 - 1
v_lshlrev_b32 v252, 16, v166                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+31], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+31                                    // convert C to bf16
buffer_store_short v31, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 32 + 32 - 1
v_lshlrev_b32 v252, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+32], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+32                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v32, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 33 + 33 - 1
v_lshlrev_b32 v252, 16, v168                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+33], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+33                                    // convert C to bf16
buffer_store_short v33, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 34 + 34 - 1
v_lshlrev_b32 v252, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+34], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+34                                    // convert C to bf16
buffer_store_short v34, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 35 + 35 - 1
v_lshlrev_b32 v252, 16, v170                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+35], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+35                                    // convert C to bf16
buffer_store_short v35, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 36 + 36 - 1
v_lshlrev_b32 v252, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+36], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+36                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v36, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 37 + 37 - 1
v_lshlrev_b32 v252, 16, v172                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+37], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+37                                    // convert C to bf16
buffer_store_short v37, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 38 + 38 - 1
v_lshlrev_b32 v252, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+38], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+38                                    // convert C to bf16
buffer_store_short v38, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 39 + 39 - 1
v_lshlrev_b32 v252, 16, v174                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+39], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+39                                    // convert C to bf16
buffer_store_short v39, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 40 + 40 - 1
v_lshlrev_b32 v252, 16, v175                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+40], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+40                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v40, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 41 + 41 - 1
v_lshlrev_b32 v252, 16, v176                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+41], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+41                                    // convert C to bf16
buffer_store_short v41, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 42 + 42 - 1
v_lshlrev_b32 v252, 16, v177                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+42], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+42                                    // convert C to bf16
buffer_store_short v42, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 43 + 43 - 1
v_lshlrev_b32 v252, 16, v178                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+43], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+43                                    // convert C to bf16
buffer_store_short v43, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 44 + 44 - 1
v_lshlrev_b32 v252, 16, v179                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+44], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+44                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v44, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 45 + 45 - 1
v_lshlrev_b32 v252, 16, v180                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+45], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+45                                    // convert C to bf16
buffer_store_short v45, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 46 + 46 - 1
v_lshlrev_b32 v252, 16, v181                        // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+46], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+46                                    // convert C to bf16
buffer_store_short v46, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 47 + 47 - 1
v_lshlrev_b32 v252, 16, v182                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+47], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+47                                    // convert C to bf16
buffer_store_short v47, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 48 + 48 - 1
v_lshlrev_b32 v252, 16, v183                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+48], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+48                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v48, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 49 + 49 - 1
v_lshlrev_b32 v252, 16, v184                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+49], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+49                                    // convert C to bf16
buffer_store_short v49, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 50 + 50 - 1
v_lshlrev_b32 v252, 16, v185                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+50], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+50                                    // convert C to bf16
buffer_store_short v50, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 51 + 51 - 1
v_lshlrev_b32 v252, 16, v186                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+51], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+51                                    // convert C to bf16
buffer_store_short v51, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 52 + 52 - 1
v_lshlrev_b32 v252, 16, v187                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+52], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+52                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v52, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 53 + 53 - 1
v_lshlrev_b32 v252, 16, v188                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+53], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+53                                    // convert C to bf16
buffer_store_short v53, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 54 + 54 - 1
v_lshlrev_b32 v252, 16, v189                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+54], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+54                                    // convert C to bf16
buffer_store_short v54, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 55 + 55 - 1
v_lshlrev_b32 v252, 16, v190                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+55], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+55                                    // convert C to bf16
buffer_store_short v55, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 56 + 56 - 1
v_lshlrev_b32 v252, 16, v191                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+56], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+56                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v56, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 57 + 57 - 1
v_lshlrev_b32 v252, 16, v192                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+57], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+57                                    // convert C to bf16
buffer_store_short v57, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 58 + 58 - 1
v_lshlrev_b32 v252, 16, v193                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+58], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+58                                    // convert C to bf16
buffer_store_short v58, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 59 + 59 - 1
v_lshlrev_b32 v252, 16, v194                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+59], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+59                                    // convert C to bf16
buffer_store_short v59, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 60 + 60 - 1
v_lshlrev_b32 v252, 16, v195                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+60], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+60                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v60, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 61 + 61 - 1
v_lshlrev_b32 v252, 16, v196                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+61], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+61                                    // convert C to bf16
buffer_store_short v61, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 62 + 62 - 1
v_lshlrev_b32 v252, 16, v197                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+62], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+62                                    // convert C to bf16
buffer_store_short v62, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 63 + 63 - 1
v_lshlrev_b32 v252, 16, v201                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+63], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+63                                    // convert C to bf16
buffer_store_short v63, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 64 + 64 - 1
v_lshlrev_b32 v252, 16, v202                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+64], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+64                                    // convert C to bf16
_v_add_lshl_u32 v134, v131, v128, 0x1              // inner add offset
s_mul_i32 s54, s[sgprStrideD1J], 32                // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_short v64, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 65 + 65 - 1
v_lshlrev_b32 v252, 16, v203                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+65], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+65                                    // convert C to bf16
buffer_store_short v65, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 66 + 66 - 1
v_lshlrev_b32 v252, 16, v204                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+66], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+66                                    // convert C to bf16
buffer_store_short v66, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 67 + 67 - 1
v_lshlrev_b32 v252, 16, v205                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+67], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+67                                    // convert C to bf16
buffer_store_short v67, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 68 + 68 - 1
v_lshlrev_b32 v252, 16, v206                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+68], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+68                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v68, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 69 + 69 - 1
v_lshlrev_b32 v252, 16, v207                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+69], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+69                                    // convert C to bf16
buffer_store_short v69, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 70 + 70 - 1
v_lshlrev_b32 v252, 16, v208                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+70], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+70                                    // convert C to bf16
buffer_store_short v70, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 71 + 71 - 1
v_lshlrev_b32 v252, 16, v209                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+71], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+71                                    // convert C to bf16
buffer_store_short v71, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 72 + 72 - 1
v_lshlrev_b32 v252, 16, v210                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+72], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+72                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v72, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 73 + 73 - 1
v_lshlrev_b32 v252, 16, v211                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+73], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+73                                    // convert C to bf16
buffer_store_short v73, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 74 + 74 - 1
v_lshlrev_b32 v252, 16, v212                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+74], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+74                                    // convert C to bf16
buffer_store_short v74, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 75 + 75 - 1
v_lshlrev_b32 v252, 16, v213                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+75], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+75                                    // convert C to bf16
buffer_store_short v75, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 76 + 76 - 1
v_lshlrev_b32 v252, 16, v214                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+76], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+76                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v76, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 77 + 77 - 1
v_lshlrev_b32 v252, 16, v215                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+77], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+77                                    // convert C to bf16
buffer_store_short v77, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 78 + 78 - 1
v_lshlrev_b32 v252, 16, v216                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+78], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+78                                    // convert C to bf16
buffer_store_short v78, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(63)                                // wait C (interleaved) 79 = 80 - 79 + 79 - 1
v_lshlrev_b32 v252, 16, v217                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+79], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+79                                    // convert C to bf16
buffer_store_short v79, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
/* optSingleColVgpr=1 optSharedColVgpr=0 optSharedMask=1 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,5,0:vw1); (0,0,5,1:vw1); (0,0,5,2:vw1); (0,0,5,3:vw1); (0,0,5,4:vw1); (0,0,5,5:vw1); (0,0,5,6:vw1); (0,0,5,7:vw1); (0,1,5,0:vw1); (0,1,5,1:vw1); (0,1,5,2:vw1); (0,1,5,3:vw1); (0,1,5,4:vw1); (0,1,5,5:vw1); (0,1,5,6:vw1); (0,1,5,7:vw1); (0,0,6,0:vw1); (0,0,6,1:vw1); (0,0,6,2:vw1); (0,0,6,3:vw1); (0,0,6,4:vw1); (0,0,6,5:vw1); (0,0,6,6:vw1); (0,0,6,7:vw1); (0,1,6,0:vw1); (0,1,6,1:vw1); (0,1,6,2:vw1); (0,1,6,3:vw1); (0,1,6,4:vw1); (0,1,6,5:vw1); (0,1,6,6:vw1); (0,1,6,7:vw1); (0,0,7,0:vw1); (0,0,7,1:vw1); (0,0,7,2:vw1); (0,0,7,3:vw1); (0,0,7,4:vw1); (0,0,7,5:vw1); (0,0,7,6:vw1); (0,0,7,7:vw1); (0,1,7,0:vw1); (0,1,7,1:vw1); (0,1,7,2:vw1); (0,1,7,3:vw1); (0,1,7,4:vw1); (0,1,7,5:vw1); (0,1,7,6:vw1); (0,1,7,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
_v_add_lshl_u32 v134, v130, v128, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128
v_add_co_u32 v134, vcc, 0x60, v134                 // add offset 
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,0,1) */
buffer_load_short_d16 v136, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,0,2) */
buffer_load_short_d16 v137, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,0,3) */
buffer_load_short_d16 v138, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,0,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v139, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,0,5) */
buffer_load_short_d16 v140, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,0,6) */
buffer_load_short_d16 v141, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,0,7) */
buffer_load_short_d16 v142, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,0) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v143, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,1) */
buffer_load_short_d16 v144, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,2) */
buffer_load_short_d16 v145, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,3) */
buffer_load_short_d16 v146, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v147, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,5) */
buffer_load_short_d16 v148, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,6) */
buffer_load_short_d16 v149, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,5,1,7) */
buffer_load_short_d16 v150, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v151, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,1) */
buffer_load_short_d16 v152, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,2) */
buffer_load_short_d16 v153, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,3) */
buffer_load_short_d16 v154, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v155, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,5) */
buffer_load_short_d16 v156, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,6) */
buffer_load_short_d16 v157, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,0,7) */
buffer_load_short_d16 v158, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,0) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v159, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,1) */
buffer_load_short_d16 v160, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,2) */
buffer_load_short_d16 v161, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,3) */
buffer_load_short_d16 v162, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v163, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,5) */
buffer_load_short_d16 v164, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,6) */
buffer_load_short_d16 v165, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,6,1,7) */
buffer_load_short_d16 v166, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v167, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,0,1) */
buffer_load_short_d16 v168, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,0,2) */
buffer_load_short_d16 v169, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,0,3) */
buffer_load_short_d16 v170, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,0,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v171, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,0,5) */
buffer_load_short_d16 v172, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,0,6) */
buffer_load_short_d16 v173, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,0,7) */
buffer_load_short_d16 v174, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,0) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v175, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,1) */
buffer_load_short_d16 v176, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,2) */
buffer_load_short_d16 v177, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,3) */
buffer_load_short_d16 v178, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,4) */
_v_add_co_u32 v134, vcc, v134, 32                  // load not edge offset
buffer_load_short_d16 v179, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,5) */
buffer_load_short_d16 v180, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,6) */
buffer_load_short_d16 v181, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,7,1,7) */
buffer_load_short_d16 v182, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v128, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128
v_add_co_u32 v134, vcc, 0x60, v134                 // add offset 

/* rC *= alpha batchEements=[(0, 0, 5, 0), (0, 0, 5, 1), (0, 0, 5, 2), (0, 0, 5, 3), (0, 0, 5, 4), (0, 0, 5, 5), (0, 0, 5, 6), (0, 0, 5, 7), (0, 1, 5, 0), (0, 1, 5, 1), (0, 1, 5, 2), (0, 1, 5, 3), (0, 1, 5, 4), (0, 1, 5, 5), (0, 1, 5, 6), (0, 1, 5, 7), (0, 0, 6, 0), (0, 0, 6, 1), (0, 0, 6, 2), (0, 0, 6, 3), (0, 0, 6, 4), (0, 0, 6, 5), (0, 0, 6, 6), (0, 0, 6, 7), (0, 1, 6, 0), (0, 1, 6, 1), (0, 1, 6, 2), (0, 1, 6, 3), (0, 1, 6, 4), (0, 1, 6, 5), (0, 1, 6, 6), (0, 1, 6, 7), (0, 0, 7, 0), (0, 0, 7, 1), (0, 0, 7, 2), (0, 0, 7, 3), (0, 0, 7, 4), (0, 0, 7, 5), (0, 0, 7, 6), (0, 0, 7, 7), (0, 1, 7, 0), (0, 1, 7, 1), (0, 1, 7, 2), (0, 1, 7, 3), (0, 1, 7, 4), (0, 1, 7, 5), (0, 1, 7, 6), (0, 1, 7, 7)] */
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

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 0 + 0 - 1
v_lshlrev_b32 v252, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+80], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+80                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v80, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 1 + 1 - 1
v_lshlrev_b32 v252, 16, v136                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+81], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+81                                    // convert C to bf16
buffer_store_short v81, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 2 + 2 - 1
v_lshlrev_b32 v252, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+82], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+82                                    // convert C to bf16
buffer_store_short v82, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 3 + 3 - 1
v_lshlrev_b32 v252, 16, v138                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+83], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+83                                    // convert C to bf16
buffer_store_short v83, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 4 + 4 - 1
v_lshlrev_b32 v252, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+84], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+84                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v84, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 5 + 5 - 1
v_lshlrev_b32 v252, 16, v140                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+85], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+85                                    // convert C to bf16
buffer_store_short v85, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 6 + 6 - 1
v_lshlrev_b32 v252, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+86], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+86                                    // convert C to bf16
buffer_store_short v86, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 7 + 7 - 1
v_lshlrev_b32 v252, 16, v142                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+87], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+87                                    // convert C to bf16
buffer_store_short v87, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 8 + 8 - 1
v_lshlrev_b32 v252, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+88], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+88                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v88, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 9 + 9 - 1
v_lshlrev_b32 v252, 16, v144                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+89], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+89                                    // convert C to bf16
buffer_store_short v89, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 10 + 10 - 1
v_lshlrev_b32 v252, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+90], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+90                                    // convert C to bf16
buffer_store_short v90, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 11 + 11 - 1
v_lshlrev_b32 v252, 16, v146                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+91], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+91                                    // convert C to bf16
buffer_store_short v91, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 12 + 12 - 1
v_lshlrev_b32 v252, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+92], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+92                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v92, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 13 + 13 - 1
v_lshlrev_b32 v252, 16, v148                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+93], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+93                                    // convert C to bf16
buffer_store_short v93, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 14 + 14 - 1
v_lshlrev_b32 v252, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+94], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+94                                    // convert C to bf16
buffer_store_short v94, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 15 + 15 - 1
v_lshlrev_b32 v252, 16, v150                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+95], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+95                                    // convert C to bf16
buffer_store_short v95, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 16 + 16 - 1
v_lshlrev_b32 v252, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+96], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+96                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v96, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 17 + 17 - 1
v_lshlrev_b32 v252, 16, v152                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+97], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+97                                    // convert C to bf16
buffer_store_short v97, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 18 + 18 - 1
v_lshlrev_b32 v252, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+98], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+98                                    // convert C to bf16
buffer_store_short v98, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 19 + 19 - 1
v_lshlrev_b32 v252, 16, v154                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+99], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+99                                    // convert C to bf16
buffer_store_short v99, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 20 + 20 - 1
v_lshlrev_b32 v252, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+100], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+100                                    // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v100, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 21 + 21 - 1
v_lshlrev_b32 v252, 16, v156                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+101], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+101                                    // convert C to bf16
buffer_store_short v101, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 22 + 22 - 1
v_lshlrev_b32 v252, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+102], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+102                                   // convert C to bf16
buffer_store_short v102, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 23 + 23 - 1
v_lshlrev_b32 v252, 16, v158                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+103], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+103                                   // convert C to bf16
buffer_store_short v103, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 24 + 24 - 1
v_lshlrev_b32 v252, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+104], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+104                                   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v104, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 25 + 25 - 1
v_lshlrev_b32 v252, 16, v160                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+105], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+105                                   // convert C to bf16
buffer_store_short v105, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 26 + 26 - 1
v_lshlrev_b32 v252, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+106], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+106                                   // convert C to bf16
buffer_store_short v106, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 27 + 27 - 1
v_lshlrev_b32 v252, 16, v162                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+107], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+107                                   // convert C to bf16
buffer_store_short v107, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 28 + 28 - 1
v_lshlrev_b32 v252, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+108], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+108                                   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v108, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 29 + 29 - 1
v_lshlrev_b32 v252, 16, v164                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+109], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+109                                   // convert C to bf16
buffer_store_short v109, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 30 + 30 - 1
v_lshlrev_b32 v252, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+110], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+110                                   // convert C to bf16
buffer_store_short v110, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 31 + 31 - 1
v_lshlrev_b32 v252, 16, v166                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+111], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+111                                   // convert C to bf16
buffer_store_short v111, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 32 + 32 - 1
v_lshlrev_b32 v252, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+112], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+112                                   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v112, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 33 + 33 - 1
v_lshlrev_b32 v252, 16, v168                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+113], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+113                                   // convert C to bf16
buffer_store_short v113, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 34 + 34 - 1
v_lshlrev_b32 v252, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+114], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+114                                   // convert C to bf16
buffer_store_short v114, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 35 + 35 - 1
v_lshlrev_b32 v252, 16, v170                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+115], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+115                                   // convert C to bf16
buffer_store_short v115, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 36 + 36 - 1
v_lshlrev_b32 v252, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+116], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+116                                   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v116, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 37 + 37 - 1
v_lshlrev_b32 v252, 16, v172                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+117], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+117                                   // convert C to bf16
buffer_store_short v117, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 38 + 38 - 1
v_lshlrev_b32 v252, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+118], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+118                                   // convert C to bf16
buffer_store_short v118, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 39 + 39 - 1
v_lshlrev_b32 v252, 16, v174                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+119], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+119                                   // convert C to bf16
buffer_store_short v119, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 40 + 40 - 1
v_lshlrev_b32 v252, 16, v175                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+120], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+120                                   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v120, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 41 + 41 - 1
v_lshlrev_b32 v252, 16, v176                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+121], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+121                                   // convert C to bf16
buffer_store_short v121, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 42 + 42 - 1
v_lshlrev_b32 v252, 16, v177                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+122], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+122                                   // convert C to bf16
buffer_store_short v122, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 43 + 43 - 1
v_lshlrev_b32 v252, 16, v178                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+123], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+123                                   // convert C to bf16
buffer_store_short v123, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 44 + 44 - 1
v_lshlrev_b32 v252, 16, v179                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+124], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+124                                   // convert C to bf16
v_add_co_u32 v134, vcc, 0x20, v134                 // add offset1
buffer_store_short v124, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 45 + 45 - 1
v_lshlrev_b32 v252, 16, v180                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+125], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+125                                   // convert C to bf16
buffer_store_short v125, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 46 + 46 - 1
v_lshlrev_b32 v252, 16, v181                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+126], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+126                                   // convert C to bf16
buffer_store_short v126, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D

s_waitcnt vmcnt(47)                                // wait C (interleaved) 47 = 48 - 47 + 47 - 1
v_lshlrev_b32 v252, 16, v182                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+127], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+127                                   // convert C to bf16
buffer_store_short v127, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
s_branch label_GW_End_37                           // jump to end
GW_B1_E1_36:

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
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v150, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDC clip if OOB. offset
buffer_load_short_d16 v151, v150, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v158, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDC clip if OOB. offset
buffer_load_short_d16 v159, v158, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v158, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,5) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v166, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDC clip if OOB. offset
buffer_load_short_d16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
v_mov_b32 v250, 0x7fff0000                         // fp32 Nan
v_mov_b32 v251, 0x7fff                             // rounding bias for bfloat16

v_lshlrev_b32 v252, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+0], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+0                                    // convert C to bf16
buffer_store_short v0, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+1], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+1                                    // convert C to bf16
buffer_store_short v1, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+2], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+2                                    // convert C to bf16
buffer_store_short v2, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+3], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+3                                    // convert C to bf16
buffer_store_short v3, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+4], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+4                                    // convert C to bf16
buffer_store_short v4, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+5], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+5                                    // convert C to bf16
buffer_store_short v5, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+6], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+6                                    // convert C to bf16
buffer_store_short v6, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+7], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+7                                    // convert C to bf16
buffer_store_short v7, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+8], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+8                                    // convert C to bf16
buffer_store_short v8, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+9], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+9                                    // convert C to bf16
buffer_store_short v9, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+10], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+10                                    // convert C to bf16
buffer_store_short v10, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+11], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+11                                    // convert C to bf16
buffer_store_short v11, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+12], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+12                                    // convert C to bf16
buffer_store_short v12, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+13], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+13                                    // convert C to bf16
buffer_store_short v13, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+14], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+14                                    // convert C to bf16
buffer_store_short v14, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+15], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+15                                    // convert C to bf16
buffer_store_short v15, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+16], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+16                                    // convert C to bf16
buffer_store_short v16, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+17], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+17                                    // convert C to bf16
buffer_store_short v17, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+18], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+18                                    // convert C to bf16
buffer_store_short v18, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+19], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+19                                    // convert C to bf16
buffer_store_short v19, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,1,4:vw1); (0,0,1,5:vw1); (0,0,1,6:vw1); (0,0,1,7:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,1,1,2:vw1); (0,1,1,3:vw1); (0,1,1,4:vw1); (0,1,1,5:vw1); (0,1,1,6:vw1); (0,1,1,7:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,2,4:vw1); (0,0,2,5:vw1); (0,0,2,6:vw1); (0,0,2,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,1,0,4) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=4
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
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=5
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
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=6
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
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=7
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
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=0
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
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=1
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
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=2
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
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=3
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
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=4
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
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=5
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
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=6
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
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=7
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
s_mov_b32 s54, 128                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v158, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDC clip if OOB. offset
buffer_load_short_d16 v159, v158, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v158, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
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
/* (d1,vc1,d0,vc0)=(0,2,0,4) */
s_mov_b32 s54, 144                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v166, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDC clip if OOB. offset
buffer_load_short_d16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,5) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,2,0,6) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,2,0,7) */
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
v_lshlrev_b32 v252, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+20], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+20                                    // convert C to bf16
buffer_store_short v20, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+21], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+21                                    // convert C to bf16
buffer_store_short v21, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+22], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+22                                    // convert C to bf16
buffer_store_short v22, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+23], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+23                                    // convert C to bf16
buffer_store_short v23, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+24], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+24                                    // convert C to bf16
buffer_store_short v24, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+25], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+25                                    // convert C to bf16
buffer_store_short v25, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+26], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+26                                    // convert C to bf16
buffer_store_short v26, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+27], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+27                                    // convert C to bf16
buffer_store_short v27, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+28], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+28                                    // convert C to bf16
buffer_store_short v28, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+29], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+29                                    // convert C to bf16
buffer_store_short v29, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+30], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+30                                    // convert C to bf16
buffer_store_short v30, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+31], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+31                                   // convert C to bf16
buffer_store_short v31, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+32], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+32                                    // convert C to bf16
buffer_store_short v32, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+33], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+33                                    // convert C to bf16
buffer_store_short v33, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+34], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+34                                    // convert C to bf16
buffer_store_short v34, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+35], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+35                                    // convert C to bf16
buffer_store_short v35, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+36], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+36                                    // convert C to bf16
buffer_store_short v36, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+37], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+37                                    // convert C to bf16
buffer_store_short v37, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+38], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+38                                    // convert C to bf16
buffer_store_short v38, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+39], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+39                                    // convert C to bf16
buffer_store_short v39, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,1,2,0:vw1); (0,1,2,1:vw1); (0,1,2,2:vw1); (0,1,2,3:vw1); (0,1,2,4:vw1); (0,1,2,5:vw1); (0,1,2,6:vw1); (0,1,2,7:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,0,3,4:vw1); (0,0,3,5:vw1); (0,0,3,6:vw1); (0,0,3,7:vw1); (0,1,3,0:vw1); (0,1,3,1:vw1); (0,1,3,2:vw1); (0,1,3,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
s_mov_b32 s54, 160                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,1) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,2,1,2) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,2,1,3) */
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
/* (d1,vc1,d0,vc0)=(0,2,1,4) */
s_mov_b32 s54, 176                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,5) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,2,1,6) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,2,1,7) */
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
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v150, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDC clip if OOB. offset
buffer_load_short_d16 v151, v150, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
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
/* (d1,vc1,d0,vc0)=(0,3,0,4) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=0 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v158, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDC clip if OOB. offset
buffer_load_short_d16 v159, v158, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v158, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,5) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,3,0,6) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,3,0,7) */
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
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
s_mov_b32 s54, 224                                  // coordOffset0 d0=1 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v166, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDC clip if OOB. offset
buffer_load_short_d16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,1) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,3,1,2) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,3,1,3) */
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
v_lshlrev_b32 v252, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+40], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+40                                    // convert C to bf16
buffer_store_short v40, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+41], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+41                                    // convert C to bf16
buffer_store_short v41, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+42], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+42                                    // convert C to bf16
buffer_store_short v42, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+43], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+43                                    // convert C to bf16
buffer_store_short v43, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+44], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+44                                    // convert C to bf16
buffer_store_short v44, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+45], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+45                                    // convert C to bf16
buffer_store_short v45, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+46], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+46                                    // convert C to bf16
buffer_store_short v46, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+47], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+47                                    // convert C to bf16
buffer_store_short v47, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+48], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+48                                    // convert C to bf16
buffer_store_short v48, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+49], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+49                                    // convert C to bf16
buffer_store_short v49, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+50], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+50                                    // convert C to bf16
buffer_store_short v50, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+51], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+51                                    // convert C to bf16
buffer_store_short v51, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+52], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+52                                    // convert C to bf16
buffer_store_short v52, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+53], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+53                                    // convert C to bf16
buffer_store_short v53, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+54], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+54                                    // convert C to bf16
buffer_store_short v54, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+55], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+55                                    // convert C to bf16
buffer_store_short v55, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+56], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+56                                    // convert C to bf16
buffer_store_short v56, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+57], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+57                                    // convert C to bf16
buffer_store_short v57, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+58], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+58                                    // convert C to bf16
buffer_store_short v58, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+59], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+59                                    // convert C to bf16
buffer_store_short v59, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (0,1,3,4:vw1); (0,1,3,5:vw1); (0,1,3,6:vw1); (0,1,3,7:vw1); (0,0,4,0:vw1); (0,0,4,1:vw1); (0,0,4,2:vw1); (0,0,4,3:vw1); (0,0,4,4:vw1); (0,0,4,5:vw1); (0,0,4,6:vw1); (0,0,4,7:vw1); (0,1,4,0:vw1); (0,1,4,1:vw1); (0,1,4,2:vw1); (0,1,4,3:vw1); (0,1,4,4:vw1); (0,1,4,5:vw1); (0,1,4,6:vw1); (0,1,4,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,3,1,4) */
s_mov_b32 s54, 240                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,5) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,3,1,6) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
/* (d1,vc1,d0,vc0)=(0,3,1,7) */
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


/* (d1,vc1,d0,vc0)=(0,4,0,0) */
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
s_mul_i32 s[sgprEdgeCheck], s[sgprStrideC1J], 32   // add offset stride
v_add_co_u32 v142, vcc, s[sgprEdgeCheck], v142     // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
s_mul_i32 s[sgprEdgeCheck+1], s[sgprStrideD1J], 32 // add offset stride
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,1) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
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
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,4,0,5) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,4,0,6) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,4,0,7) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,4,1,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,4,1,1) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,4,1,2) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,4,1,3) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,4,1,4) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,4,1,5) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,4,1,6) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,4,1,7) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
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
v_lshlrev_b32 v252, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+60], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+60                                    // convert C to bf16
buffer_store_short v60, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+61], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+61                                    // convert C to bf16
buffer_store_short v61, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+62], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+62                                    // convert C to bf16
buffer_store_short v62, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+63], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+63                                    // convert C to bf16
buffer_store_short v63, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+64], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+64                                    // convert C to bf16
buffer_store_short v64, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+65], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+65                                    // convert C to bf16
buffer_store_short v65, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+66], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+66                                    // convert C to bf16
buffer_store_short v66, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+67], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+67                                    // convert C to bf16
buffer_store_short v67, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+68], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+68                                    // convert C to bf16
buffer_store_short v68, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+69], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+69                                    // convert C to bf16
buffer_store_short v69, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+70], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+70                                    // convert C to bf16
buffer_store_short v70, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+71], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+71                                    // convert C to bf16
buffer_store_short v71, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+72], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+72                                    // convert C to bf16
buffer_store_short v72, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+73], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+73                                    // convert C to bf16
buffer_store_short v73, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+74], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+74                                    // convert C to bf16
buffer_store_short v74, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+75], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+75                                    // convert C to bf16
buffer_store_short v75, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+76], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+76                                    // convert C to bf16
buffer_store_short v76, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+77], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+77                                    // convert C to bf16
buffer_store_short v77, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+78], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+78                                    // convert C to bf16
buffer_store_short v78, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+79], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+79                                    // convert C to bf16
buffer_store_short v79, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #4 (d1,d0,vc1,vc0) = */
/*    (0,0,5,0:vw1); (0,0,5,1:vw1); (0,0,5,2:vw1); (0,0,5,3:vw1); (0,0,5,4:vw1); (0,0,5,5:vw1); (0,0,5,6:vw1); (0,0,5,7:vw1); (0,1,5,0:vw1); (0,1,5,1:vw1); (0,1,5,2:vw1); (0,1,5,3:vw1); (0,1,5,4:vw1); (0,1,5,5:vw1); (0,1,5,6:vw1); (0,1,5,7:vw1); (0,0,6,0:vw1); (0,0,6,1:vw1); (0,0,6,2:vw1); (0,0,6,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
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
/* (d1,vc1,d0,vc0)=(0,5,0,1) */
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
/* (d1,vc1,d0,vc0)=(0,5,0,2) */
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
/* (d1,vc1,d0,vc0)=(0,5,0,3) */
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
/* (d1,vc1,d0,vc0)=(0,5,0,4) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,5,0,5) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,5,0,6) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,5,0,7) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,5,1,0) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,5,1,1) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=1
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,5,1,2) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=2
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,5,1,3) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=3
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,5,1,4) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=4
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
/* (d1,vc1,d0,vc0)=(0,5,1,5) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=5
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
/* (d1,vc1,d0,vc0)=(0,5,1,6) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=6
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
/* (d1,vc1,d0,vc0)=(0,5,1,7) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=7
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


/* (d1,vc1,d0,vc0)=(0,6,0,0) */
s_mov_b32 s54, 128                                 // coordOffset0 d0=1 vc0=4
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
/* (d1,vc1,d0,vc0)=(0,6,0,1) */
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
/* (d1,vc1,d0,vc0)=(0,6,0,2) */
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
/* (d1,vc1,d0,vc0)=(0,6,0,3) */
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
v_lshlrev_b32 v252, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+80], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+80                                    // convert C to bf16
buffer_store_short v80, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+81], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+81                                    // convert C to bf16
buffer_store_short v81, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+82], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+82                                    // convert C to bf16
buffer_store_short v82, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+83], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+83                                    // convert C to bf16
buffer_store_short v83, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+84], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+84                                    // convert C to bf16
buffer_store_short v84, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+85], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+85                                    // convert C to bf16
buffer_store_short v85, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+86], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+86                                    // convert C to bf16
buffer_store_short v86, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+87], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+87                                    // convert C to bf16
buffer_store_short v87, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+88], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+88                                    // convert C to bf16
buffer_store_short v88, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+89], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+89                                    // convert C to bf16
buffer_store_short v89, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+90], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+90                                    // convert C to bf16
buffer_store_short v90, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+91], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+91                                    // convert C to bf16
buffer_store_short v91, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+92], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+92                                    // convert C to bf16
buffer_store_short v92, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+93], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+93                                    // convert C to bf16
buffer_store_short v93, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+94], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+94                                    // convert C to bf16
buffer_store_short v94, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+95], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+95                                    // convert C to bf16
buffer_store_short v95, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+96], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+96                                    // convert C to bf16
buffer_store_short v96, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+97], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+97                                    // convert C to bf16
buffer_store_short v97, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+98], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+98                                    // convert C to bf16
buffer_store_short v98, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+99], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+99                                    // convert C to bf16
buffer_store_short v99, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #5 (d1,d0,vc1,vc0) = */
/*    (0,0,6,4:vw1); (0,0,6,5:vw1); (0,0,6,6:vw1); (0,0,6,7:vw1); (0,1,6,0:vw1); (0,1,6,1:vw1); (0,1,6,2:vw1); (0,1,6,3:vw1); (0,1,6,4:vw1); (0,1,6,5:vw1); (0,1,6,6:vw1); (0,1,6,7:vw1); (0,0,7,0:vw1); (0,0,7,1:vw1); (0,0,7,2:vw1); (0,0,7,3:vw1); (0,0,7,4:vw1); (0,0,7,5:vw1); (0,0,7,6:vw1); (0,0,7,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,6,0,4) */
s_mov_b32 s54, 144                                 // coordOffset0 d0=1 vc0=4
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
/* (d1,vc1,d0,vc0)=(0,6,0,5) */
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
/* (d1,vc1,d0,vc0)=(0,6,0,6) */
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
/* (d1,vc1,d0,vc0)=(0,6,0,7) */
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
/* (d1,vc1,d0,vc0)=(0,6,1,0) */
s_mov_b32 s54, 160                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,6,1,1) */
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
/* (d1,vc1,d0,vc0)=(0,6,1,2) */
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
/* (d1,vc1,d0,vc0)=(0,6,1,3) */
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
/* (d1,vc1,d0,vc0)=(0,6,1,4) */
s_mov_b32 s54, 176                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,6,1,5) */
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
/* (d1,vc1,d0,vc0)=(0,6,1,6) */
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
/* (d1,vc1,d0,vc0)=(0,6,1,7) */
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
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=1 vc0=4
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
/* (d1,vc1,d0,vc0)=(0,7,0,1) */
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
/* (d1,vc1,d0,vc0)=(0,7,0,2) */
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
/* (d1,vc1,d0,vc0)=(0,7,0,3) */
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
/* (d1,vc1,d0,vc0)=(0,7,0,4) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=0 vc0=4
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
/* (d1,vc1,d0,vc0)=(0,7,0,5) */
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
/* (d1,vc1,d0,vc0)=(0,7,0,6) */
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
/* (d1,vc1,d0,vc0)=(0,7,0,7) */
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
v_lshlrev_b32 v252, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+100], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+100                                    // convert C to bf16
buffer_store_short v100, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+101], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+101                                    // convert C to bf16
buffer_store_short v101, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+102], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+102                                    // convert C to bf16
buffer_store_short v102, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+103], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+103                                    // convert C to bf16
buffer_store_short v103, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+104], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+104                                    // convert C to bf16
buffer_store_short v104, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+105], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+105                                    // convert C to bf16
buffer_store_short v105, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+106], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+106                                    // convert C to bf16
buffer_store_short v106, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+107], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+107                                    // convert C to bf16
buffer_store_short v107, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v151                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+108], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+108                                    // convert C to bf16
buffer_store_short v108, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v153                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+109], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+109                                    // convert C to bf16
buffer_store_short v109, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v155                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+110], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+110                                    // convert C to bf16
buffer_store_short v110, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v157                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+111], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+111                                    // convert C to bf16
buffer_store_short v111, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v159                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+112], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+112                                    // convert C to bf16
buffer_store_short v112, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v161                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+113], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+113                                    // convert C to bf16
buffer_store_short v113, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v163                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+114], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+114                                    // convert C to bf16
buffer_store_short v114, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v165                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+115], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+115                                    // convert C to bf16
buffer_store_short v115, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v167                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+116], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+116                                    // convert C to bf16
buffer_store_short v116, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v169                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+117], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+117                                    // convert C to bf16
buffer_store_short v117, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v171                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+118], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+118                                    // convert C to bf16
buffer_store_short v118, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v173                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+119], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+119                                    // convert C to bf16
buffer_store_short v119, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #6 (d1,d0,vc1,vc0) = */
/*    (0,1,7,0:vw1); (0,1,7,1:vw1); (0,1,7,2:vw1); (0,1,7,3:vw1); (0,1,7,4:vw1); (0,1,7,5:vw1); (0,1,7,6:vw1); (0,1,7,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,7,1,0) */
s_mov_b32 s54, 224                                  // coordOffset0 d0=1 vc0=0
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
/* (d1,vc1,d0,vc0)=(0,7,1,1) */
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
/* (d1,vc1,d0,vc0)=(0,7,1,2) */
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
/* (d1,vc1,d0,vc0)=(0,7,1,3) */
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
/* (d1,vc1,d0,vc0)=(0,7,1,4) */
s_mov_b32 s54, 240                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
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
/* (d1,vc1,d0,vc0)=(0,7,1,5) */
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
/* (d1,vc1,d0,vc0)=(0,7,1,6) */
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
/* (d1,vc1,d0,vc0)=(0,7,1,7) */
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
v_lshlrev_b32 v252, 16, v135                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+120], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+120                                    // convert C to bf16
buffer_store_short v120, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v137                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+121], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+121                                    // convert C to bf16
buffer_store_short v121, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v139                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+122], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+122                                    // convert C to bf16
buffer_store_short v122, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v141                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+123], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+123                                    // convert C to bf16
buffer_store_short v123, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v143                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+124], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+124                                    // convert C to bf16
buffer_store_short v124, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v145                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+125], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+125                                    // convert C to bf16
buffer_store_short v125, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v147                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+126], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+126                                    // convert C to bf16
buffer_store_short v126, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
v_lshlrev_b32 v252, 16, v149                         // convert bf16 to fp32
_v_mac_f32 v[vgprValuC+127], v252, s[sgprBeta]        // finalSum = sum*alpha + C*beta
FP32_TO_BF16 vgprValuC+127                                    // convert C to bf16
buffer_store_short v127, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_branch label_GW_End_37                           // jump to end
label_GW_End_37:

label_0039:  /// KernelEnd
/* Direct-output logical N-group completion.
 *
 * The 0x60 workspace starts with one task-claim word per
 * (problem, physical-N-tile); num_done_groups counter words follow it.
 * All waves wait for routed output stores before lane 0 contributes one
 * completion for this claimed task. The last completion in a logical group
 * publishes wait_signal_value through the 0x68 flag array.
 */
s_load_dwordx2 s[90:91], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60
s_load_dwordx2 s[92:93], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68
s_waitcnt lgkmcnt(0)
s_or_b32 s80, s90, s91
s_cmp_eq_u32 s80, 0
s_cbranch_scc1 .L_moe_nchunk_signal_end
s_or_b32 s80, s92, s93
s_cmp_eq_u32 s80, 0
s_cbranch_scc1 .L_moe_nchunk_signal_end

s_waitcnt vmcnt(0) lgkmcnt(0)
buffer_wbinvl1_vol
s_barrier

v_cmp_eq_u32 vcc, v[vgprSerial], 0
s_and_saveexec_b64 s[98:99], vcc
s_cbranch_execz .L_moe_nchunk_signal_restore_exec

s_load_dword s80, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x4c
s_load_dword s83, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_load_dword s81, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x30
s_load_dword s82, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x34
s_load_dword s86, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x44
s_waitcnt lgkmcnt(0)
s_lshr_b32 s80, s80, 8                            // physical tiles per logical group
s_cmp_eq_u32 s80, 0
s_cbranch_scc1 .L_moe_nchunk_signal_restore_exec
s_cmp_eq_u32 s83, 0
s_cbranch_scc1 .L_moe_nchunk_signal_restore_exec

s_add_u32 s82, s82, 255
s_lshr_b32 s82, s82, 8                            // total physical N tiles
s_mul_i32 s89, s81, s82                           // claim words before counters
s_lshl_b32 s89, s89, 2
s_add_u32 s90, s90, s89                           // logical counter array base
s_addc_u32 s91, s91, 0

s_mov_b32 s84, 0                                  // logical group id
s_mov_b32 s85, s100                               // physical N-tile id remainder
.L_moe_nchunk_group_div:
s_cmp_lt_u32 s85, s80
s_cbranch_scc1 .L_moe_nchunk_group_div_done
s_sub_u32 s85, s85, s80
s_add_u32 s84, s84, 1
s_branch .L_moe_nchunk_group_div
.L_moe_nchunk_group_div_done:
s_cmp_ge_u32 s84, s83
s_cbranch_scc1 .L_moe_nchunk_signal_restore_exec

s_mul_i32 s87, s84, s80                          // first physical tile in group
s_sub_u32 s88, s82, s87                          // physical tiles remaining
s_cmp_lt_u32 s88, s80
s_cselect_b32 s87, s88, s80                      // actual physical tiles in group
s_mul_i32 s88, s81, s87                          // expected CTA completions

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
s_cbranch_scc0 .L_moe_nchunk_signal_restore_exec

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

.L_moe_nchunk_signal_restore_exec:
s_or_b64 exec, exec, s[98:99]
.L_moe_nchunk_signal_end:
s_endpgm                                           // Kernel End
