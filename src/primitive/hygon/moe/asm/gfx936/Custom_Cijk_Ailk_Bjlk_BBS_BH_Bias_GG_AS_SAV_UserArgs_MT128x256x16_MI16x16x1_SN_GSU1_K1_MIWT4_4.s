
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx936"
.text
.protected Custom_Cijk_Ailk_Bjlk_BBS_BH_Bias_GG_AS_SAV_UserArgs_MT128x256x16_MI16x16x1_SN_GSU1_K1_MIWT4_4
.globl Custom_Cijk_Ailk_Bjlk_BBS_BH_Bias_GG_AS_SAV_UserArgs_MT128x256x16_MI16x16x1_SN_GSU1_K1_MIWT4_4
.p2align 8
.type Custom_Cijk_Ailk_Bjlk_BBS_BH_Bias_GG_AS_SAV_UserArgs_MT128x256x16_MI16x16x1_SN_GSU1_K1_MIWT4_4,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Custom_Cijk_Ailk_Bjlk_BBS_BH_Bias_GG_AS_SAV_UserArgs_MT128x256x16_MI16x16x1_SN_GSU1_K1_MIWT4_4
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 210 // vgprs
  .amdhsa_next_free_sgpr 100 // sgprs
  .amdhsa_group_segment_fixed_size 65536 // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text
/* Num VGPR   =128 */
/* Num AccVGPR=0 */
/* Num SGPR   =72 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 16 x 4 */
/* SubGroup= 8 x 64 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=4, GlobalReadVectorWidthB=8 */
/* DirectToLdsA=True */
/* DirectToLdsB=True */
/* UseSgprForGRO=False */
.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: Custom_Cijk_Ailk_Bjlk_BBS_BH_Bias_GG_AS_SAV_UserArgs_MT128x256x16_MI16x16x1_SN_GSU1_K1_MIWT4_4
    .symbol: 'Custom_Cijk_Ailk_Bjlk_BBS_BH_Bias_GG_AS_SAV_UserArgs_MT128x256x16_MI16x16x1_SN_GSU1_K1_MIWT4_4.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            GroupedSizes
        .size:            4
        .offset:          0
        .value_kind:      by_value
        .value_type:      u32
      - .name:            DeviceArguments
        .size:            8
        .offset:          4
        .value_kind:      global_buffer
        .value_type:      void
        .address_space:   generic
      - .name:            argsPtr
        .size:            8
        .offset:          12
        .value_kind:      global_buffer
        .value_type:      void
        .address_space:   generic
      - .name:            WgTable
        .size:            4
        .offset:          20
        .value_kind:      by_value
        .value_type:      u32
      - .name:            GSU
        .size:            4
        .offset:          24
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   65536
    .kernarg_segment_align:      8
    .kernarg_segment_size:       32
    .max_flat_workgroup_size:    512
    .private_segment_fixed_size: 0
    .sgpr_count:                 100
    .sgpr_spill_count:           0
    .vgpr_count:                 210
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Custom_Cijk_Ailk_Bjlk_BBS_BH_Bias_GG_AS_SAV_UserArgs_MT128x256x16_MI16x16x1_SN_GSU1_K1_MIWT4_4:

/* Magic div and mod functions */
.macro V_MAGIC_DIV dstIdx:req dividend:req magicNumber:req magicShift:req magicA:req
    v_mul_hi_u32 v[\dstIdx+1] \dividend \magicNumber
    v_mul_lo_u32 v[\dstIdx+0] \dividend \magicA
    v_add_u32 v[\dstIdx+0] v[\dstIdx+0] v[\dstIdx+1]
    v_lshrrev_b32 v[\dstIdx+0] \magicShift v[\dstIdx+0]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-64), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprValuA_X0_I0, 64
.set vgprValuB_X0_I0, 72
.set vgprLocalWriteAddrA, 80
.set vgprLocalWriteAddrB, 81
.set vgprGlobalReadOffsetA, 82
.set vgprGlobalReadOffsetB, 83
.set vgprG2LA, 84
.set vgprG2LB, 86
.set vgprLocalReadAddrA, 90
.set vgprLocalReadAddrB, 92
.set vgprSerial, 94
.set vgprLocalReadAddrA_ori, 190
.set vgprLocalReadAddrB_ori, 191
.set vgprLocalReadAddrASwap, 192
.set vgprLocalReadAddrBSwap, 193
.set vgprLocalReadAddrASwap_ori, 194
.set vgprLocalReadAddrBSwap_ori, 195
.set vgprValuBias_M, 200
.set vgprValuScalAlpha_M, 204
/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprExternalArgAddress, 96
.set sgprLoopCounterL, 5
.set sgprOrigLoopCounter, 6
.set sgprSrdD, 8
.set sgprSrdC, 12
.set sgprNumWorkGroups0, 7
.set sgprNumWorkGroups1, 16
.set sgprSizesFree, 20
.set sgprSizesSum, 23
.set sgprAddressD, 24
.set sgprAddressC, 26
.set sgprAddressA, 28
.set sgprAddressB, 30
.set sgprStridesD, 32
.set sgprStridesC, 34
.set sgprStridesA, 36
.set sgprStridesB, 38
.set sgprAlpha, 40
.set sgprBeta, 41
.set sgprGSU, 42
.set sgprSrdA, 44
.set sgprSrdB, 48
.set sgprShadowLimitA, 18
.set sgprShadowLimitB, 52
.set sgprGlobalReadIncsA, 17
.set sgprGlobalReadIncsB, 43
.set sgprLocalWriteAddrA, 54
.set sgprLocalWriteAddrB, 55
.set sgprLocalWriteAddrA_ori, 70
.set sgprLocalWriteAddrB_ori, 71
.set sgprMask, 72
.set sgprWaveiD, 73

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
.set constStrideB1J, 1
.set sgprStrideBL, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 128
.set MT1, 256
.set DepthU, 16
.set BpeA, 2
.set BpeALog2, 1
.set BpeB, 2
.set BpeBLog2, 1
.set BpeAGR, 2
.set BpeAGRLog2, 1
.set BpeBGR, 2
.set BpeBGRLog2, 1
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 4
.set SrdShiftLeftB, 8
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xffffffff
.set BufferOOB, 0x80000000

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
    v_mul_lo_u32 v[\vgprTmp+0] s[sgprStrideAL] v[\vgprOffsetL] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0] vcc v[\vgprOffset0I] v[\vgprTmp+0] // accumulate K lower
    v_add_u32 v[\vgprAddr+0] 0x4 v[\vgprAddr+0]      // add prepad for pointer shift
    v_lshlrev_b32 v[\vgprAddr+0] 0x1 v[\vgprAddr+0]  // offset *= bytes/element
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffset1J:req vgprOffsetL:req vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0] s[sgprStrideBL] v[\vgprOffsetL] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0] vcc v[\vgprOffset1J] v[\vgprTmp+0] // accumulate K lower
    v_add_u32 v[\vgprAddr+0] 0x8 v[\vgprAddr+0]      // add prepad for pointer shift
    v_lshlrev_b32 v[\vgprAddr+0] 0x1 v[\vgprAddr+0]  // offset *= bytes/element
.endm

/* Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor; */
.macro DYNAMIC_VECTOR_DIVIDE vQuotient vRemainder vDividend vDivisor vTmp0 vTmp1 sTmp
    v_cvt_f32_u32 v[\vQuotient] v[\vDivisor]
    v_rcp_f32 v[\vQuotient] v[\vQuotient]
    v_mul_f32 v[\vQuotient] 0x4f800000 v[\vQuotient]
    v_cvt_u32_f32 v[\vQuotient] v[\vQuotient]
    v_mul_lo_u32 v[\vRemainder] v[\vDivisor] v[\vQuotient]
    v_mul_hi_u32 v[\vTmp0] v[\vDivisor] v[\vQuotient]
    v_sub_co_u32 v[\vTmp1] vcc 0x0 v[\vRemainder]
    v_cmp_ne_i32 s[\sTmp:\sTmp+1] 0x0 v[\vTmp0]
    v_cndmask_b32 v[\vRemainder] v[\vTmp1] v[\vRemainder] s[\sTmp:\sTmp+1]
    v_mul_hi_u32 v[\vRemainder] v[\vRemainder] v[\vQuotient]
    v_sub_co_u32 v[\vTmp0] vcc v[\vQuotient] v[\vRemainder]
    v_add_co_u32 v[\vQuotient] vcc v[\vQuotient] v[\vRemainder]
    v_cndmask_b32 v[\vQuotient] v[\vQuotient] v[\vTmp0] s[\sTmp:\sTmp+1]
    v_mul_hi_u32 v[\vQuotient] v[\vQuotient] v[\vDividend]
    v_mul_lo_u32 v[\vRemainder] v[\vQuotient] v[\vDivisor]
    v_sub_co_u32 v[\vTmp0] vcc v[\vDividend] v[\vRemainder]
    v_cmp_ge_u32 s[\sTmp:\sTmp+1] v[\vDividend] v[\vRemainder]
    v_add_co_u32 v[\vRemainder] vcc 0x1 v[\vQuotient]
    v_add_co_u32 v[\vTmp1] vcc -1 v[\vQuotient]
    v_cmp_le_u32 vcc v[\vDivisor] v[\vTmp0]
    s_and_b64 vcc s[\sTmp:\sTmp+1] vcc
    v_cndmask_b32 v[\vQuotient] v[\vQuotient] v[\vRemainder] vcc
    v_cndmask_b32 v[\vQuotient] v[\vTmp1] v[\vQuotient] s[\sTmp:\sTmp+1]
    v_cmp_ne_i32 vcc 0x0 v[\vDivisor]
    v_cndmask_b32 v[\vQuotient] -1 v[\vQuotient] vcc // final result
    v_mul_lo_u32 v[\vRemainder] v[\vQuotient] v[\vDivisor]
    v_sub_co_u32 v[\vRemainder] vcc v[\vDividend] v[\vRemainder] // final result
.endm

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Grouped Gemm: Load num of Gemms */
s_load_dword s27, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0

/* Grouped Gemm: Load GSU data */
s_load_dword s[sgprGSU], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x18

/* Grouped Gemm: Load address of external kernel arguments */
s_load_dwordx2 s[sgprExternalArgAddress:sgprExternalArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x4

/* Grouped Gemm: Load address of kernel arguments */
s_load_dwordx2 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xc
s_waitcnt lgkmcnt(0)
s_mov_b32 m0, 0x10000                              // LDS clamp at 65536 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
/* Check if custom structure pointer is null */
s_cmp_eq_u64 s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0 // s[ExternalArgAddress] == 0 ?
s_cbranch_scc0 label_IsExternalValid               // branch if s[ExternalArgAddress] != 0
s_mov_b32 s26, 124
s_mul_i32 s22, s27, 4
s_mov_b64 s[16:17], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s26, 196
s_mov_b32 s22, 0x0
s_mov_b64 s[16:17], s[sgprExternalArgAddress:sgprExternalArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s24, 1
s_mov_b32 s23, 0
s_load_dwordx4 s[12:15], s[16:17], s22
s_cmpk_eq_u32 s27, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s20, s12, 7                             // s20 = s12 / 128
s_and_b32 s18, 127, s12                            // s18 = s12 % 128
s_addc_u32 s20, s20, 0x0
s_lshr_b32 s21, s13, 8                             // s21 = s13 / 256
s_and_b32 s18, 255, s13                            // s18 = s13 % 256
s_addc_u32 s21, s21, 0x0
s_mul_i32 s20, s20, s21
s_mul_i32 s20, s20, s14
s_mul_i32 s20, s20, s[sgprGSU]
s_add_u32 s23, s23, s20
s_cmp_lt_u32 s[sgprWorkGroup0], s23
s_cbranch_scc1 label_FOUND
s_add_u32 s22, s22, s26
s_load_dwordx4 s[12:15], s[16:17], s22
s_add_u32 s24, s24, 1
s_cmp_lt_u32 s24, s27
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s20, s12, 7                             // s20 = s12 / 128
s_and_b32 s18, 127, s12                            // s18 = s12 % 128
s_addc_u32 s20, s20, 0x0
s_lshr_b32 s21, s13, 8                             // s21 = s13 / 256
s_and_b32 s18, 255, s13                            // s18 = s13 % 256
s_addc_u32 s21, s21, 0x0
s_mul_i32 s20, s20, s21
s_mul_i32 s20, s20, s14
s_mul_i32 s20, s20, s[sgprGSU]
s_add_u32 s23, s23, s20

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s11, s24, 1
s_mov_b32 s74, s11                                  // FTrain: preserve grouped GEMM index for rank-signal metadata
s_sub_u32 s8, s23, s20
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s8
/* Check if custom structure pointer is null */
s_cmp_eq_u64 s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0 // s[ExternalArgAddress] == 0 ?
s_cbranch_scc0 label_LoadExternalStruct            // branch if s[ExternalArgAddress] != 0

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s27, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s11, s11, 124
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s11
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0

/* Load Kernel Args */
s_load_dwordx16 s[20:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0
s_load_dwordx4 s[36:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40
s_load_dwordx2 s[40:41], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s11, s11, 196
s_add_u32 s[sgprExternalArgAddress], s[sgprExternalArgAddress], s11
s_addc_u32 s[sgprExternalArgAddress+1], s[sgprExternalArgAddress+1], 0x0
s_load_dwordx16 s[20:35], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x0
s_load_dwordx4 s[36:39], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x40
s_load_dword s40, s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x50
// Read Beta
s_load_dword s41, s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x60
label_LoadExternalStructEnd:

/* init: add vgpr [64...144) to pool */
/* init: add vgpr [0...64) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v0, 63, v[vgprSerial]                    // v0 = v[vgprSerial] % 64
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // v4 = v[vgprSerial] / 64
v_and_b32 v5, 3, v0                                // v5 = v[vgprSerial] % 64 % 4
v_lshrrev_b32 v3, 2, v0                            // v3 = v[vgprSerial] / 64/ 4


/* local read addresses: final offsets a */

v_and_b32 v8, 1, v4                                // v[vgprSerial] / 64 % 2
s_mov_b32 s62, 32                                  // 
v_mul_lo_u32 v8, s62, v8                           // A_stride*(MT_A(32)) v8 = v[vgprSerial] / 64 % 2 x 32

v_and_b32 v1, 7, v0		                           // v1 = v[vgprSerial] % 64 % 8
v_lshrrev_b32 v2, 3, v0                            // v2 = v[vgprSerial] % 64 / 8
                                                   // 
v_and_b32 v6, 3, v1                                // v6 = v[vgprSerial] % 64 % 8 % 4
v_lshrrev_b32 v7, 2, v1                            // v7 = v[vgprSerial] % 64 % 8 / 4
                                                   // 
v_lshlrev_b32 v6, 3, v6                            // v6 = v[vgprSerial] % 64 % 8 % 4 x 8
v_lshlrev_b32 v10, 8, v7                           // v10 = v[vgprSerial] % 64 % 8 / 4 x 256
v_lshlrev_b32 v11, 5, v7                           // v11 = v[vgprSerial] % 64 % 8 / 4 x 32
                                                   //
v_and_b32 v12, 15, v0                              // v12 = v[vgprSerial] % 64 % 16
v_lshrrev_b32 v13, 4, v0                           // v13 = v[vgprSerial] % 64 / 16 
v_lshrrev_b32 v12, 3, v12                          // v12 = v[vgprSerial] % 64 % 16 / 8
                                                   //
v_lshlrev_b32 v12, 7, v12                          // v12 = v[vgprSerial] % 64 % 16 / 8 x 128
v_lshlrev_b32 v13, 9, v13                         // v13 = v[vgprSerial] % 64 / 16 x 512
                                                   //
v_add_co_u32 v6, vcc, v11, v6                      // v6 = v[vgprSerial] % 64 % 8 % 4 x 8 + v[vgprSerial] % 64 % 8 / 4 x 32
v_add_co_u32 v6, vcc, v6, v10                      // v6 = v[vgprSerial] % 64 % 8 % 4 x 8 + v[vgprSerial] % 64 % 8 / 4 x 32 + v[vgprSerial] % 64 % 8 / 4 x 512
v_add_co_u32 v6, vcc, v6, v12                      // v6 = v[vgprSerial] % 64 % 8 % 4 x 8 + v[vgprSerial] % 64 % 8 / 4 x 32 + v[vgprSerial] % 64 % 8 / 4 x 512 + v[vgprSerial] % 64 % 16 / 8 x 256
v_add_co_u32 v6, vcc, v6, v13                      // v6 = v[vgprSerial] % 64 % 8 % 4 x 8 + v[vgprSerial] % 64 % 8 / 4 x 32 + v[vgprSerial] % 64 % 8 / 4 x 512 + v[vgprSerial] % 64 % 16 / 8 x 256 + v[vgprSerial] % 64 / 16 x 1024
v_add_lshl_u32 v[vgprLocalReadAddrA], v6, v8, 0x1  // Final offset 1: LDS-Av [vgprLocalReadAddrA] = (v[vgprSerial] % 64 % 8 % 4 x 8 + v[vgprSerial] % 64 % 8 / 4 x 32 + v[vgprSerial] % 64 % 8 / 4 x 512 + v[vgprSerial] % 64 % 16 / 8 x 256 + v[vgprSerial] % 64 / 16 x 1024+ v[vgprSerial] / 64 % 2 x 32)x 2

v_lshrrev_b32 v7, 2, v0                            // v7 = v[vgprSerial] % 64 / 4
v_and_b32 v7, 3, v7                                // v7 = v[vgprSerial] % 64 / 4 % 4
v_cmp_eq_u32 s[64:65], v7, 3                       // v[vgprSerial] % 64 / 4 % 4 == 3
                                                   //
v_and_b32 v2, 1, v4                                // v2 = v[vgprSerial] / 64 % 2
v_cmp_eq_u32 s[62:63], v2, 1                       // v[vgprSerial] / 64 % 2 == 1?
                                                   // 
s_and_b64 s[62:63], s[62:63], s[64:65]             // v[vgprSerial] % 64 / 4 % 4 == 3 ? & v[vgprSerial] / 64 % 2 == 1?
v_mov_b32 v7,0x100                                 // v7 = 256
                                                   //
v_cndmask_b32 v7, 0, v7, s[62:63]                  // 
                                                   //
v_sub_i32 v6, v6, v7
v_add_lshl_u32 v[vgprLocalReadAddrASwap], v6, v8, 0x1  // Final offset 1: LDS-A 閿熻鍑ゆ嫹閿熼ズ锟� 256

/* local read addresses: final offsets b */
v_lshrrev_b32 v9, 1, v4                            // v9 = v[vgprSerial] / 64 / 2
s_mov_b32 s62, 32                                  //  
v_mul_lo_u32 v9, s62, v9                           // B_stride*(MT_B(32))  v9 = v[vgprSerial] / 64 / 2 x 32

v_and_b32 v1, 7, v0		                           // v1 = v[vgprSerial] % 64 % 8
v_lshrrev_b32 v2, 3, v0                            // v2 = v[vgprSerial] % 64 / 8
                                                   // 
v_and_b32 v6, 3, v1                                // v6 = v[vgprSerial] % 64 % 8 % 4
v_lshrrev_b32 v7, 2, v1                            // v7 = v[vgprSerial] % 64 % 8 / 4
                                                   // 
v_lshlrev_b32 v6, 3, v6                            // v6 = v[vgprSerial] % 64 % 8 % 4 x 8
v_lshlrev_b32 v10, 9, v7                           // v10 = v[vgprSerial] % 64 % 8 / 4 x 512
v_lshlrev_b32 v11, 5, v7                           // v11 = v[vgprSerial] % 64 % 8 / 4 x 32
                                                   // 
v_and_b32 v12, 15, v0                              // v12 = v[vgprSerial] % 64 % 16
v_lshrrev_b32 v13, 4, v0                           // v13 = v[vgprSerial] % 64 / 16
v_lshrrev_b32 v12, 3, v12                          // v12 = v[vgprSerial] % 64 % 16 / 8
                                                   // 
v_lshlrev_b32 v12, 8, v12                          // v12 = v[vgprSerial] % 64 % 16 / 8 x 512
v_lshlrev_b32 v13, 10, v13                         // v13 = v[vgprSerial] % 64 / 16 x 1024
                                                   // 
v_add_co_u32 v6, vcc, v11, v6                      // 
v_add_co_u32 v6, vcc, v6, v10                      // 
v_add_co_u32 v6, vcc, v6, v12                      // v6 = v[vgprSerial] % 64 % 8 % 4 x 8 +v[vgprSerial] % 64 % 8 / 4 x 32+v[vgprSerial] % 64 % 8 / 4 x 512 +v[vgprSerial] % 64 % 16 / 8 x 512
v_add_co_u32 v6, vcc, v6, v13                      // v6 = v[vgprSerial] % 64 % 8 % 4 x 8 +v[vgprSerial] % 64 % 8 / 4 x 32+v[vgprSerial] % 64 % 8 / 4 x 512 +v[vgprSerial] % 64 % 16 / 8 x 512 +v[vgprSerial] % 64 / 16 x 1024

v_add_lshl_u32 v[vgprLocalReadAddrB], v6, v9, 0x1  // Final offset 1: LDS-B (v[vgprSerial] % 64 % 8 % 4 x 8 +v[vgprSerial] % 64 % 8 / 4 x 32+v[vgprSerial] % 64 % 8 / 4 x 512 +v[vgprSerial] % 64 % 16 / 8 x 512 +v[vgprSerial] % 64 / 16 x 1024 + v[vgprSerial] / 64 / 2 x 32) x 2

v_lshrrev_b32 v7, 2, v0                            // v7 = v[vgprSerial] % 64 / 4
v_and_b32 v7, 3, v7                                // v7 = v[vgprSerial] % 64 / 4 % 4
v_cmp_eq_u32 s[64:65], v7, 3                       // v[vgprSerial] % 64 / 4 % 4 == 3

v_lshrrev_b32 v2, 1, v4                            // v[vgprSerial] / 64  /2 
v_cmp_eq_u32 s[62:63], v2, 3                        // v[vgprSerial] / 64  /2  == 3

s_and_b64 s[62:63], s[62:63], s[64:65]
v_mov_b32 v7,0x200
v_cndmask_b32 v7, 0, v7, s[62:63]

v_sub_i32 v6, v6, v7
v_add_lshl_u32 v[vgprLocalReadAddrBSwap], v6, v9, 0x1  // Final offset 1: LDS-B   - 512

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB], vcc, 0x1000, v[vgprLocalReadAddrB] //  += LdsOffsetB (lower)  8192
v_add_co_u32 v[vgprLocalReadAddrBSwap], vcc, 0x1000, v[vgprLocalReadAddrBSwap] //  += LdsOffsetB (lower)

v_mov_b32 v[vgprLocalReadAddrA_ori], v[vgprLocalReadAddrA]
v_mov_b32 v[vgprLocalReadAddrB_ori], v[vgprLocalReadAddrB]

v_mov_b32 v[vgprLocalReadAddrASwap_ori], v[vgprLocalReadAddrASwap]
v_mov_b32 v[vgprLocalReadAddrBSwap_ori], v[vgprLocalReadAddrBSwap]

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 32 */
/* v1 = A-unroll = serial/LVCA */
v_lshrrev_b32 v1, 5, v[vgprSerial]                 // v1 = v[vgprSerial] / 32
v_and_b32 v0, 31, v[vgprSerial]                    // v0 = v[vgprSerial] % 32
/* tile *= glvw */
v_lshlrev_b32 v0, 0x2, v0                          // v0 = v0 * 4
/* LVCB = 32 */
/* v3 = B-unroll = serial/LVCB */
v_lshrrev_b32 v3, 5, v[vgprSerial]                 // v3 = v[vgprSerial] / 32
v_and_b32 v2, 31, v[vgprSerial]                    // v2 = v[vgprSerial] % 32
/* tile *= glvw */
v_lshlrev_b32 v2, 0x3, v2                          // v2 = v2 * 8
/* lwaUnrollAssignmentA = v1 */
/* lwaUnrollAssignmentB = v3 */

v_lshrrev_b32 v[vgprValuA_X0_I0], 6, v[vgprSerial]
v_readfirstlane_b32 s[sgprWaveiD], v[vgprValuA_X0_I0]


/* local write addresses: first offset a */
v_mul_u32_u24 v4, 0x80, v1                         // lwAL**(MTA + PAD)
v_add_lshl_u32 v4, v0, v4, 0x1                     // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpe
v_readfirstlane_b32 s[sgprLocalWriteAddrA], v4     // Copy lds write address VGPR to SGPR
v_mov_b32 v[vgprLocalWriteAddrA], v4               // Copy lds write address for handle Tail Loop When use DirectToLds

s_and_b32 s64, s[sgprWaveiD], 1                   // 
s_lshl_b32 s64, s64, 18                           // 
s_or_b32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA], s64 // 

/* local write addresses: first offset b */
v_mul_u32_u24 v4, 0x100, v3                        // lwBL**(MTB + PAD)
v_add_lshl_u32 v4, v2, v4, 0x1                     // lwFOB = (lwBB + lwBL*(MT1J+PAD))*bpe
v_add_co_u32 v4, vcc, 0x1000, v4                   // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=2048*2
v_readfirstlane_b32 s[sgprLocalWriteAddrB], v4     // Copy lds write address VGPR to SGPR
v_mov_b32 v[vgprLocalWriteAddrB], v4               // Copy lds write address for handle Tail Loop When use DirectToLds

s_and_b32 s64, s[sgprWaveiD], 1                   // 
s_lshl_b32 s64, s64, 18                           // 
s_or_b32 s[sgprLocalWriteAddrB], s[sgprLocalWriteAddrB], s64 // 

s_mov_b32 s[sgprLocalWriteAddrA_ori], s[sgprLocalWriteAddrA]
s_mov_b32 s[sgprLocalWriteAddrB_ori], s[sgprLocalWriteAddrB]

s_waitcnt lgkmcnt(0)                               // wait for 92 bytes of kern args
label_stop:
v_mov_b32 v6, MT0                                  // set MT0 into sgpr
v_mov_b32 v5, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v4, v6                               // v4 = ceil(v5 / v6)
v_rcp_iflag_f32 v4, v4                             // v4 = ceil(v5 / v6)
v_cvt_f32_u32 v7, v5                               // v4 = ceil(v5 / v6)
v_mul_f32 v4, v4, v7                               // v4 = ceil(v5 / v6)
v_cvt_u32_f32 v4, v4                               // v4 = ceil(v5 / v6)
v_mul_u32_u24 v7, v4, v6                           // v4 = ceil(v5 / v6)
v_sub_u32 v7, v5, v7                               // v4 = ceil(v5 / v6)
v_cmp_ne_u32 vcc, v7, 0                            // v4 = ceil(v5 / v6)
v_addc_co_u32 v4, vcc, v4, 0, vcc                  // ceil
v_mov_b32 v6, MT1                                  // set MT1 into sgpr
v_mov_b32 v5, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v4      // set back to numWorkGroup0
v_cvt_f32_u32 v4, v6                               // v4 = ceil(v5 / v6)
v_rcp_iflag_f32 v4, v4                             // v4 = ceil(v5 / v6)
v_cvt_f32_u32 v7, v5                               // v4 = ceil(v5 / v6)
v_mul_f32 v4, v4, v7                               // v4 = ceil(v5 / v6)
v_cvt_u32_f32 v4, v4                               // v4 = ceil(v5 / v6)
v_mul_u32_u24 v7, v4, v6                           // v4 = ceil(v5 / v6)
v_sub_u32 v7, v5, v7                               // v4 = ceil(v5 / v6)
v_cmp_ne_u32 vcc, v7, 0                            // v4 = ceil(v5 / v6)
v_addc_co_u32 v4, vcc, v4, 0, vcc                  // ceil
v_readfirstlane_b32 s[sgprNumWorkGroups1], v4      // set back to numWorkGroup1

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0x0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

/* Grouped Gemm: remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s56, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_mul_i32 s56, s56, s[sgprGSU]
v_cvt_f32_u32 v4, s56                              // s56 = s[sgprWorkGroup0] / s56
v_rcp_iflag_f32 v4, v4                             // s56 = s[sgprWorkGroup0] / s56
v_cvt_f32_u32 v5, s[sgprWorkGroup0]                // s56 = s[sgprWorkGroup0] / s56
v_mul_f32 v4, v4, v5                               // s56 = s[sgprWorkGroup0] / s56
v_cvt_u32_f32 v4, v4                               // s56 = s[sgprWorkGroup0] / s56
v_mul_u32_u24 v5, v4, s56                          // s56 = s[sgprWorkGroup0] / s56
v_sub_u32 v5, s[sgprWorkGroup0], v5                // s56 = s[sgprWorkGroup0] / s56
v_cmpx_eq_u32 exec, v5, s56                        // s56 = s[sgprWorkGroup0] / s56
v_add_u32 v4, 1, v4                                // s56 = s[sgprWorkGroup0] / s56
s_mov_b64 exec, -1                                 // s56 = s[sgprWorkGroup0] / s56
v_readfirstlane_b32 s56, v4
s_mov_b32 s[sgprWorkGroup2], s56
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s56, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s56, s56, s[sgprWorkGroup2]
s_mul_i32 s56, s56, s[sgprGSU]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s56
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v4, s[sgprNumWorkGroups0]            // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v4, v4                             // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v5, s[sgprWorkGroup0]                // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v4, v4, v5                               // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v4, v4                               // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v5, v4, s[sgprNumWorkGroups0]        // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_u32 v5, s[sgprWorkGroup0], v5                // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmpx_eq_u32 exec, v5, s[sgprNumWorkGroups0]      // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_u32 v4, 1, v4                                // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b64 exec, -1                                 // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_readfirstlane_b32 s56, v4
s_mov_b32 s[sgprWorkGroup1], s56
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s56, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s56

/* Early stop if wg exceed */
s_cmp_ge_u32 s[sgprWorkGroup2], s[sgprSizesFree+2]
s_cbranch_scc0 label_NoEarlyStop_wgExceed
label_EarlyStop_if_wg_exceed:
s_endpgm
label_NoEarlyStop_wgExceed:

s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 8  // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // pre-pad to make room for possible pointer shift

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if s[Alpha] != 0
s_mov_b32 s[sgprSizesSum+0], 0x0                   // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
s_mov_b32 s60, 0x10000001L                         // magic number for WGM==8
s_mul_hi_u32 s57, s[sgprWorkGroup1], s60           // s_magic mul
s_mul_i32 s56, s[sgprWorkGroup1], s60              // s_magic mul
s_lshr_b64 s[56:57], s[56:57], 31                  // sMagicDiv
s_mul_i32 s57, s56, 8                              // quotient * non-magic divisor
s_sub_u32 s57, s[sgprWorkGroup1], s57              // WorkGroup1=remainder
s_mul_i32 s57, s57, s[sgprNumWorkGroups0]          // (wg1 % WGM)*nwg0
s_add_u32 s57, s57, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*nwg0
s_mul_hi_u32 s59, s[sgprNumWorkGroups1], s60       // s_magic mul
s_mul_i32 s58, s[sgprNumWorkGroups1], s60          // s_magic mul
s_lshr_b64 s[58:59], s[58:59], 31                  // sMagicDiv
s_mul_i32 s59, 8, s58                              // quotient * non-magic divisor
s_sub_u32 s60, s[sgprNumWorkGroups1], s59          // WorkGroup1=remainder
s_cmp_eq_u32 s60, 0                                // remainder == 0 ?
s_cmov_b32 s60, 8                                  // remainder = WGM if remainder == 0
s_cmp_ge_u32 s56, s58                              // blockId >= numFullBlocks ?
s_cselect_b32 s58, s60, 8
v_cvt_f32_u32 v4, s58                              // s[sgprWorkGroup0] = s57 / s58
v_rcp_iflag_f32 v4, v4                             // s[sgprWorkGroup0] = s57 / s58
v_cvt_f32_u32 v5, s57                              // s[sgprWorkGroup0] = s57 / s58
v_mul_f32 v4, v4, v5                               // s[sgprWorkGroup0] = s57 / s58
v_cvt_u32_f32 v4, v4                               // s[sgprWorkGroup0] = s57 / s58
v_mul_u32_u24 v5, v4, s58                          // s[sgprWorkGroup0] = s57 / s58
v_sub_u32 v5, s57, v5                              // s[sgprWorkGroup0] = s57 / s58
v_cmpx_eq_u32 exec, v5, s58                        // s[sgprWorkGroup0] = s57 / s58
v_add_u32 v4, 1, v4                                // s[sgprWorkGroup0] = s57 / s58
v_mov_b32 v5, 0                                    // s[sgprWorkGroup1] = s57 % s58
s_mov_b64 exec, -1                                 // s[sgprWorkGroup0] = s57 / s58
v_readfirstlane_b32 s[sgprWorkGroup0], v4
v_readfirstlane_b32 s[sgprWorkGroup1], v5
s_mul_i32 s56, s56, 8                              // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s56 // wg1 += blockId * WGM

/* global read addresses: tile offset assignment a */
/* graTileAssignmentA = v0 */

/* global read addresses: tile offset assignment b */
/* graTileAssignmentB = v2 */

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
s_mul_i32 s56, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_sub_u32 s56, s[sgprSizeI], s56                   // edge = Size0I - WG*MT
s_sub_u32 s56, s56, 4                              // edge -= margin(4)
v_mov_b32 v8, s56                                  // edge vgpr = Size0I- WG*MT - margin(4)
v_min_i32 v4, v8, v4                               // offset = (offset < edge) ? offset(v4) : edge(v8)

/* global read addresses: shift b */
s_mul_i32 s56, s[sgprWorkGroup1], 256              // WorkGroup[01] * MT
s_sub_u32 s56, s[sgprSizeJ], s56                   // edge = Size1J - WG*MT
s_sub_u32 s56, s56, 8                              // edge -= margin(8)
v_mov_b32 v8, s56                                  // edge vgpr = Size1J- WG*MT - margin(8)
v_min_i32 v5, v8, v5                               // offset = (offset < edge) ? offset(v5) : edge(v8)

/* global read addresses: final offsets a */
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  4,  6, 8 // gROA_0_0_0_0

/* global read addresses: final offsets b */
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  5,  7, 8 // gROB_0_0_0_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s59, s[sgprWorkGroup0], 128           // WorkGroup[01] * MT
s_mul_i32 s58, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_mov_b32 s60, 1                                   // Init tensor size
s_mov_b32 s61, 0                                   // init tensor size
s_sub_u32 s56, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s57, constStrideA0I, s56              // stride x (size-1)
s_mul_i32 s56, constStrideA0I, s56                 // stride x (size-1)
s_add_u32 s60, s60, s56                            // sum tensor size
s_addc_u32 s61, s61, s57                           // sum tensor size
s_sub_u32 s56, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s57, s[sgprStrideAL], s56             // stride x (size-1)
s_mul_i32 s56, s[sgprStrideAL], s56                // stride x (size-1)
s_add_u32 s60, s60, s56                            // sum tensor size
s_addc_u32 s61, s61, s57                           // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s60, s58          // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s61, s59         // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 8 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s57, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s56, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s58, s58, s56                            // accum wg term to tilestart
s_addc_u32 s59, s59, s57                           // accum wg term to tilestart
s_lshl_b64 s[58:59], s[58:59], 0x1                 // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s58    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s59   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s59, s[sgprWorkGroup1], 256           // WorkGroup[01] * MT
s_mul_i32 s58, s[sgprWorkGroup1], 256              // WorkGroup[01] * MT
s_mov_b32 s60, 1                                   // Init tensor size
s_mov_b32 s61, 0                                   // init tensor size
s_sub_u32 s56, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s57, constStrideB1J, s56              // stride x (size-1)
s_mul_i32 s56, constStrideB1J, s56                 // stride x (size-1)
s_add_u32 s60, s60, s56                            // sum tensor size
s_addc_u32 s61, s61, s57                           // sum tensor size
s_sub_u32 s56, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s57, s[sgprStrideBL], s56             // stride x (size-1)
s_mul_i32 s56, s[sgprStrideBL], s56                // stride x (size-1)
s_add_u32 s60, s60, s56                            // sum tensor size
s_addc_u32 s61, s61, s57                           // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s60, s58          // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s61, s59         // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s57, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s56, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s58, s58, s56                            // accum wg term to tilestart
s_addc_u32 s59, s59, s57                           // accum wg term to tilestart
s_lshl_b64 s[58:59], s[58:59], 0x1                 // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s58    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s59   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: increments a */
s_mul_i32 s56, s[sgprGSU], DepthU*BpeAGR
s_mul_i32 s[sgprGlobalReadIncsA+0], s56, s[sgprStrideAL] // incrA unrollIdx)

/* global read addresses: increments b */
s_mul_i32 s56, s[sgprGSU], DepthU*BpeBGR
s_mul_i32 s[sgprGlobalReadIncsB+0], s56, s[sgprStrideBL] // incrB unrollIdx)
/* declare loop num iterations */
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 4 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 16
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
/* local read addresses: init pointers a */

/* localReadInitPointers */
/* local read addresses: init pointers b */

	/* localReadInitPointers */

	/* FTrain rank-signal wait through argsPtr metadata.
	   Layout:
	     0x00: int* barrier_ptr
	     0x08: int* owner_rank_by_gemm
	     0x10: int  rank
	     0x14: int  wait_signal_value
	     0x18: int  enabled
	*/
	s_cmp_eq_u64 s[sgprKernArgAddress:sgprKernArgAddress+1], 0
	s_cbranch_scc1 label_FTrainRankSignalWaitDone
	s_load_dword s75, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x18
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s75, 0
	s_cbranch_scc1 label_FTrainRankSignalWaitDone
	s_load_dwordx4 s[76:79], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0
	s_load_dwordx2 s[80:81], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s82, s74, 2
	s_add_u32 s78, s78, s82
	s_addc_u32 s79, s79, 0
	s_load_dword s83, s[78:79], 0x0
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s83, s80
	s_cbranch_scc1 label_FTrainRankSignalWaitDone
	v_mov_b32 v24, s83
	v_lshlrev_b32 v24, 2, v24
label_FTrainRankSignalWaitLoop:
	global_load_dword v25, v24, s[76:77] glc
	s_waitcnt vmcnt(0)
	buffer_wbinvl1_vol
	v_cmp_ge_i32 vcc, v25, s81
	s_cbranch_vccz label_FTrainRankSignalWaitLoop
	s_barrier
label_FTrainRankSignalWaitDone:

	/* prefetch: global -> local */
	s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
	s_cbranch_scc1 label_ShadowInitStart               // skip to ShadowInitStart iter b/c numIter==0

///////////////////////////////////////
// prefetch load 1
s_add_u32 m0, s[sgprLocalWriteAddrA], 0            // m0 <- LDS write address
/* before DirectToLds load, ensure prior ds_reads have finished */
s_barrier
buffer_load_dwordx2 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0             // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32

s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              
s_cbranch_scc1 label_ShadowInitStart

///////////////////////////////////////
// prefetch load 2
s_add_u32 m0, s[sgprLocalWriteAddrA], 0x4000            // m0 <- LDS write address
buffer_load_dwordx2 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x4000            // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32


s_cmp_eq_u32 s[sgprLoopCounterL], 0x2
s_cbranch_scc1 label_ShadowInitStart

///////////////////////////////////////
// prefetch load 3
s_add_u32 m0, s[sgprLocalWriteAddrA], 0x8000            // m0 <- LDS write address
buffer_load_dwordx2 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x8000            // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32

///////////////////////////////////////////

/******************************************/
/* End setupNewTile                       */
/******************************************/
label_ShadowInitStart:
s_mov_b32 s[sgprSrdD+0], s[sgprAddressD+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdD+1], s[sgprAddressD+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdD+2], 0x80000000
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s[sgprSrdC+0], s[sgprAddressC+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdC+1], s[sgprAddressC+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdC+2], 0x80000000
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s58, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s57, s58, s[sgprStrideC1J]            // ScaleC s58 by Stride
s_mul_i32 s56, s58, s[sgprStrideC1J]               // ScaleC s58 by Stride
s_lshl_b64 s[56:57], s[56:57], 1                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s56    // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s57   // add hi to SRD
s_mul_hi_u32 s57, s58, s[sgprStrideD1J]            // ScaleD s58 by Stride
s_mul_i32 s56, s58, s[sgprStrideD1J]               // ScaleD s58 by Stride
s_lshl_b64 s[56:57], s[56:57], 1                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s56    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s57   // add hi to SRD

s_mul_hi_u32 s57, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s56, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[56:57], s[56:57], 1                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s56        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s57       // add hi to SRD
s_mul_hi_u32 s57, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s56, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[56:57], s[56:57], 1                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s56        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s57       // add hi to SRD


/* initC: remove ValuC vgpr buffer [0...64) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [64...80) from pool */
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
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_EZNLIDVUI30QB5OH_0   // Only branch on scc1
s_getpc_b64 s[56:57]                               // addr of next instr
s_add_i32 s58, label_PrefetchGlobalLastIterEnd, 0x4 // target branch offset
s_add_u32 s56, s56, s58                            // add target branch offset
s_addc_u32 s57, s57, 0                             // add high and carry
s_setpc_b64 s[56:57]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_EZNLIDVUI30QB5OH_0:
s_waitcnt vmcnt(0)
s_mov_b32 s[sgprMask], 0x10000

s_waitcnt vmcnt(0)                                 // 8wait for global read

s_waitcnt lgkmcnt(0)                               // for local write
s_barrier                                          // for global read

/* local write a */

/* local write b */

/* local write swap a */

/* (EPS=1) local write swap internal offset -> 16384 */

/* local write swap b */

/* (EPS=1) local write swap internal offset -> 16384 */

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
s_cmp_lt_u32 s[sgprLoopCounterL], 0x4              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL

s_cmp_ge_i32 s[sgprWaveiD], 4
s_cbranch_scc1 label_LoopBeginL_2

label_LoopBeginL_1:

/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/


/* iter 0 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrASwap] offset:128 // ds_read_m Method for load lds data

/* local read b */
ds_read_m32x16_b16 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrBSwap] offset:256 // ds_read_m Method for load lds data

s_add_u32 m0, s[sgprLocalWriteAddrA], 0xc000        // m0 <- LDS write address
buffer_load_dwordx2 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0xc000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32

s_waitcnt vmcnt(4)
s_barrier
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
//s_nop 1
v_mmac_f32_16x16x16_bf16 v[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
v_mmac_f32_16x16x16_bf16 v[4:7], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
v_mmac_f32_16x16x16_bf16 v[8:11], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
v_mmac_f32_16x16x16_bf16 v[12:15], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[12:15] // left value = v[12+0:15+0]
v_mmac_f32_16x16x16_bf16 v[16:19], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[16:19] // left value = v[16+0:19+0]
v_mmac_f32_16x16x16_bf16 v[20:23], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[20:23] // left value = v[20+0:23+0]
v_mmac_f32_16x16x16_bf16 v[24:27], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[24:27] // left value = v[24+0:27+0]
v_mmac_f32_16x16x16_bf16 v[28:31], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[28:31] // left value = v[28+0:31+0]
v_mmac_f32_16x16x16_bf16 v[32:35], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[32:35] // left value = v[32+0:35+0]
v_mmac_f32_16x16x16_bf16 v[36:39], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[36:39] // left value = v[36+0:39+0]
v_mmac_f32_16x16x16_bf16 v[40:43], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[40:43] // left value = v[40+0:43+0]
v_mmac_f32_16x16x16_bf16 v[44:47], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[44:47] // left value = v[44+0:47+0]
v_mmac_f32_16x16x16_bf16 v[48:51], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[48:51] // left value = v[48+0:51+0]
v_mmac_f32_16x16x16_bf16 v[52:55], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[52:55] // left value = v[52+0:55+0]
v_mmac_f32_16x16x16_bf16 v[56:59], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[56:59] // left value = v[56+0:59+0]
v_mmac_f32_16x16x16_bf16 v[60:63], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[60:63] // left value = v[60+0:63+0]

/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x3              // counterL==1
s_cbranch_scc1 label_LoopEndL_oddexit_31              // exit LoopL

/* iter 1 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x4000 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrASwap] offset:0x4080 // ds_read_m Method for load lds data

/* local read b */
ds_read_m32x16_b16 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0x4000 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrBSwap] offset:0x4100 // ds_read_m Method for load lds data

s_add_u32 m0, s[sgprLocalWriteAddrA], 0        // m0 <- LDS write address
buffer_load_dwordx2 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32

s_waitcnt vmcnt(4)
s_barrier
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
//s_nop 1
v_mmac_f32_16x16x16_bf16 v[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
v_mmac_f32_16x16x16_bf16 v[4:7], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
v_mmac_f32_16x16x16_bf16 v[8:11], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
v_mmac_f32_16x16x16_bf16 v[12:15], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[12:15] // left value = v[12+0:15+0]
v_mmac_f32_16x16x16_bf16 v[16:19], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[16:19] // left value = v[16+0:19+0]
v_mmac_f32_16x16x16_bf16 v[20:23], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[20:23] // left value = v[20+0:23+0]
v_mmac_f32_16x16x16_bf16 v[24:27], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[24:27] // left value = v[24+0:27+0]
v_mmac_f32_16x16x16_bf16 v[28:31], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[28:31] // left value = v[28+0:31+0]
v_mmac_f32_16x16x16_bf16 v[32:35], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[32:35] // left value = v[32+0:35+0]
v_mmac_f32_16x16x16_bf16 v[36:39], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[36:39] // left value = v[36+0:39+0]
v_mmac_f32_16x16x16_bf16 v[40:43], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[40:43] // left value = v[40+0:43+0]
v_mmac_f32_16x16x16_bf16 v[44:47], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[44:47] // left value = v[44+0:47+0]
v_mmac_f32_16x16x16_bf16 v[48:51], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[48:51] // left value = v[48+0:51+0]
v_mmac_f32_16x16x16_bf16 v[52:55], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[52:55] // left value = v[52+0:55+0]
v_mmac_f32_16x16x16_bf16 v[56:59], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[56:59] // left value = v[56+0:59+0]
v_mmac_f32_16x16x16_bf16 v[60:63], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[60:63] // left value = v[60+0:63+0]

/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x3              // counterL==1
s_cbranch_scc1 label_LoopEndL_oddexit_32              // exit LoopL

/* iter 2 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x8000 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrASwap] offset:0x8080 // ds_read_m Method for load lds data

/* local read b */
ds_read_m32x16_b16 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0x8000 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrBSwap] offset:0x8100 // ds_read_m Method for load lds data

s_add_u32 m0, s[sgprLocalWriteAddrA], 0x4000        // m0 <- LDS write address
buffer_load_dwordx2 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x4000       // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32

s_waitcnt vmcnt(4)
s_barrier
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
//s_nop 1
v_mmac_f32_16x16x16_bf16 v[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
v_mmac_f32_16x16x16_bf16 v[4:7], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
v_mmac_f32_16x16x16_bf16 v[8:11], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
v_mmac_f32_16x16x16_bf16 v[12:15], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[12:15] // left value = v[12+0:15+0]
v_mmac_f32_16x16x16_bf16 v[16:19], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[16:19] // left value = v[16+0:19+0]
v_mmac_f32_16x16x16_bf16 v[20:23], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[20:23] // left value = v[20+0:23+0]
v_mmac_f32_16x16x16_bf16 v[24:27], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[24:27] // left value = v[24+0:27+0]
v_mmac_f32_16x16x16_bf16 v[28:31], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[28:31] // left value = v[28+0:31+0]
v_mmac_f32_16x16x16_bf16 v[32:35], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[32:35] // left value = v[32+0:35+0]
v_mmac_f32_16x16x16_bf16 v[36:39], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[36:39] // left value = v[36+0:39+0]
v_mmac_f32_16x16x16_bf16 v[40:43], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[40:43] // left value = v[40+0:43+0]
v_mmac_f32_16x16x16_bf16 v[44:47], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[44:47] // left value = v[44+0:47+0]
v_mmac_f32_16x16x16_bf16 v[48:51], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[48:51] // left value = v[48+0:51+0]
v_mmac_f32_16x16x16_bf16 v[52:55], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[52:55] // left value = v[52+0:55+0]
v_mmac_f32_16x16x16_bf16 v[56:59], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[56:59] // left value = v[56+0:59+0]
v_mmac_f32_16x16x16_bf16 v[60:63], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[60:63] // left value = v[60+0:63+0]

/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x3              // counterL==1
s_cbranch_scc1 label_LoopEndL_oddexit_33              // exit LoopL



/* iter 3 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0xc000 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrASwap] offset:0xc080 // ds_read_m Method for load lds data

/* local read b */
ds_read_m32x16_b16 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0xc000 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrBSwap] offset:0xc100 // ds_read_m Method for load lds data

s_add_u32 m0, s[sgprLocalWriteAddrA], 0x8000        // m0 <- LDS write address
buffer_load_dwordx2 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x8000       // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32

s_waitcnt vmcnt(4)
s_barrier
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
//s_nop 1
v_mmac_f32_16x16x16_bf16 v[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
v_mmac_f32_16x16x16_bf16 v[4:7], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
v_mmac_f32_16x16x16_bf16 v[8:11], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
v_mmac_f32_16x16x16_bf16 v[12:15], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[12:15] // left value = v[12+0:15+0]
v_mmac_f32_16x16x16_bf16 v[16:19], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[16:19] // left value = v[16+0:19+0]
v_mmac_f32_16x16x16_bf16 v[20:23], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[20:23] // left value = v[20+0:23+0]
v_mmac_f32_16x16x16_bf16 v[24:27], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[24:27] // left value = v[24+0:27+0]
v_mmac_f32_16x16x16_bf16 v[28:31], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[28:31] // left value = v[28+0:31+0]
v_mmac_f32_16x16x16_bf16 v[32:35], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[32:35] // left value = v[32+0:35+0]
v_mmac_f32_16x16x16_bf16 v[36:39], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[36:39] // left value = v[36+0:39+0]
v_mmac_f32_16x16x16_bf16 v[40:43], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[40:43] // left value = v[40+0:43+0]
v_mmac_f32_16x16x16_bf16 v[44:47], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[44:47] // left value = v[44+0:47+0]
v_mmac_f32_16x16x16_bf16 v[48:51], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[48:51] // left value = v[48+0:51+0]
v_mmac_f32_16x16x16_bf16 v[52:55], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[52:55] // left value = v[52+0:55+0]
v_mmac_f32_16x16x16_bf16 v[56:59], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[56:59] // left value = v[56+0:59+0]
v_mmac_f32_16x16x16_bf16 v[60:63], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[60:63] // left value = v[60+0:63+0]



/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x3              // counterL==1
s_cbranch_scc0 label_LoopBeginL_1                    // restart LoopL
s_branch label_LoopEndL_oddexit  /// unroll loop odditer exit

label_LoopBeginL_2:

/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/
/* iter 0 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrASwap] offset:128 // ds_read_m Method for load lds data

/* local read b */
ds_read_m32x16_b16 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrBSwap] offset:256 // ds_read_m Method for load lds data

s_add_u32 m0, s[sgprLocalWriteAddrA], 0xc000        // m0 <- LDS write address
buffer_load_dwordx2 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0xc000        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32


s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
//s_nop 1
v_mmac_f32_16x16x16_bf16 v[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
v_mmac_f32_16x16x16_bf16 v[4:7], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
v_mmac_f32_16x16x16_bf16 v[8:11], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
v_mmac_f32_16x16x16_bf16 v[12:15], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[12:15] // left value = v[12+0:15+0]
v_mmac_f32_16x16x16_bf16 v[16:19], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[16:19] // left value = v[16+0:19+0]
v_mmac_f32_16x16x16_bf16 v[20:23], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[20:23] // left value = v[20+0:23+0]
v_mmac_f32_16x16x16_bf16 v[24:27], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[24:27] // left value = v[24+0:27+0]
v_mmac_f32_16x16x16_bf16 v[28:31], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[28:31] // left value = v[28+0:31+0]
v_mmac_f32_16x16x16_bf16 v[32:35], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[32:35] // left value = v[32+0:35+0]
v_mmac_f32_16x16x16_bf16 v[36:39], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[36:39] // left value = v[36+0:39+0]
v_mmac_f32_16x16x16_bf16 v[40:43], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[40:43] // left value = v[40+0:43+0]
v_mmac_f32_16x16x16_bf16 v[44:47], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[44:47] // left value = v[44+0:47+0]
v_mmac_f32_16x16x16_bf16 v[48:51], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[48:51] // left value = v[48+0:51+0]
v_mmac_f32_16x16x16_bf16 v[52:55], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[52:55] // left value = v[52+0:55+0]
v_mmac_f32_16x16x16_bf16 v[56:59], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[56:59] // left value = v[56+0:59+0]
v_mmac_f32_16x16x16_bf16 v[60:63], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[60:63] // left value = v[60+0:63+0]
s_waitcnt vmcnt(4)
s_barrier
/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x3              // counterL==1
s_cbranch_scc1 label_LoopEndL_oddexit_31              // exit LoopL

/* iter 1 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x4000 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrASwap] offset:0x4080 // ds_read_m Method for load lds data

/* local read b */
ds_read_m32x16_b16 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0x4000 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrBSwap] offset:0x4100 // ds_read_m Method for load lds data

s_add_u32 m0, s[sgprLocalWriteAddrA], 0        // m0 <- LDS write address
buffer_load_dwordx2 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0        // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32


s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
//s_nop 1
v_mmac_f32_16x16x16_bf16 v[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
v_mmac_f32_16x16x16_bf16 v[4:7], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
v_mmac_f32_16x16x16_bf16 v[8:11], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
v_mmac_f32_16x16x16_bf16 v[12:15], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[12:15] // left value = v[12+0:15+0]
v_mmac_f32_16x16x16_bf16 v[16:19], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[16:19] // left value = v[16+0:19+0]
v_mmac_f32_16x16x16_bf16 v[20:23], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[20:23] // left value = v[20+0:23+0]
v_mmac_f32_16x16x16_bf16 v[24:27], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[24:27] // left value = v[24+0:27+0]
v_mmac_f32_16x16x16_bf16 v[28:31], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[28:31] // left value = v[28+0:31+0]
v_mmac_f32_16x16x16_bf16 v[32:35], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[32:35] // left value = v[32+0:35+0]
v_mmac_f32_16x16x16_bf16 v[36:39], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[36:39] // left value = v[36+0:39+0]
v_mmac_f32_16x16x16_bf16 v[40:43], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[40:43] // left value = v[40+0:43+0]
v_mmac_f32_16x16x16_bf16 v[44:47], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[44:47] // left value = v[44+0:47+0]
v_mmac_f32_16x16x16_bf16 v[48:51], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[48:51] // left value = v[48+0:51+0]
v_mmac_f32_16x16x16_bf16 v[52:55], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[52:55] // left value = v[52+0:55+0]
v_mmac_f32_16x16x16_bf16 v[56:59], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[56:59] // left value = v[56+0:59+0]
v_mmac_f32_16x16x16_bf16 v[60:63], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[60:63] // left value = v[60+0:63+0]
s_waitcnt vmcnt(4)
s_barrier
/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x3              // counterL==1
s_cbranch_scc1 label_LoopEndL_oddexit_32              // exit LoopL

/* iter 2 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0x8000 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrASwap] offset:0x8080 // ds_read_m Method for load lds data

/* local read b */
ds_read_m32x16_b16 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0x8000 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrBSwap] offset:0x8100 // ds_read_m Method for load lds data

s_add_u32 m0, s[sgprLocalWriteAddrA], 0x4000        // m0 <- LDS write address
buffer_load_dwordx2 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x4000       // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32


s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
//s_nop 1
v_mmac_f32_16x16x16_bf16 v[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
v_mmac_f32_16x16x16_bf16 v[4:7], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
v_mmac_f32_16x16x16_bf16 v[8:11], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
v_mmac_f32_16x16x16_bf16 v[12:15], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[12:15] // left value = v[12+0:15+0]
v_mmac_f32_16x16x16_bf16 v[16:19], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[16:19] // left value = v[16+0:19+0]
v_mmac_f32_16x16x16_bf16 v[20:23], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[20:23] // left value = v[20+0:23+0]
v_mmac_f32_16x16x16_bf16 v[24:27], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[24:27] // left value = v[24+0:27+0]
v_mmac_f32_16x16x16_bf16 v[28:31], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[28:31] // left value = v[28+0:31+0]
v_mmac_f32_16x16x16_bf16 v[32:35], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[32:35] // left value = v[32+0:35+0]
v_mmac_f32_16x16x16_bf16 v[36:39], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[36:39] // left value = v[36+0:39+0]
v_mmac_f32_16x16x16_bf16 v[40:43], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[40:43] // left value = v[40+0:43+0]
v_mmac_f32_16x16x16_bf16 v[44:47], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[44:47] // left value = v[44+0:47+0]
v_mmac_f32_16x16x16_bf16 v[48:51], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[48:51] // left value = v[48+0:51+0]
v_mmac_f32_16x16x16_bf16 v[52:55], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[52:55] // left value = v[52+0:55+0]
v_mmac_f32_16x16x16_bf16 v[56:59], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[56:59] // left value = v[56+0:59+0]
v_mmac_f32_16x16x16_bf16 v[60:63], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[60:63] // left value = v[60+0:63+0]
s_waitcnt vmcnt(4)
s_barrier
/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x3              // counterL==1
s_cbranch_scc1 label_LoopEndL_oddexit_33              // exit LoopL



/* iter 3 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0xc000 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrASwap] offset:0xc080 // ds_read_m Method for load lds data

/* local read b */
ds_read_m32x16_b16 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0xc000 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrBSwap] offset:0xc100 // ds_read_m Method for load lds data

s_add_u32 m0, s[sgprLocalWriteAddrA], 0x8000        // m0 <- LDS write address
buffer_load_dwordx2 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0
s_add_u32 m0, s[sgprLocalWriteAddrB], 0x8000       // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0, lds // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32


s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
//s_nop 1
v_mmac_f32_16x16x16_bf16 v[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
v_mmac_f32_16x16x16_bf16 v[4:7], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
v_mmac_f32_16x16x16_bf16 v[8:11], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
v_mmac_f32_16x16x16_bf16 v[12:15], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[12:15] // left value = v[12+0:15+0]
v_mmac_f32_16x16x16_bf16 v[16:19], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[16:19] // left value = v[16+0:19+0]
v_mmac_f32_16x16x16_bf16 v[20:23], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[20:23] // left value = v[20+0:23+0]
v_mmac_f32_16x16x16_bf16 v[24:27], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[24:27] // left value = v[24+0:27+0]
v_mmac_f32_16x16x16_bf16 v[28:31], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[28:31] // left value = v[28+0:31+0]
v_mmac_f32_16x16x16_bf16 v[32:35], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[32:35] // left value = v[32+0:35+0]
v_mmac_f32_16x16x16_bf16 v[36:39], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[36:39] // left value = v[36+0:39+0]
v_mmac_f32_16x16x16_bf16 v[40:43], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[40:43] // left value = v[40+0:43+0]
v_mmac_f32_16x16x16_bf16 v[44:47], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[44:47] // left value = v[44+0:47+0]
v_mmac_f32_16x16x16_bf16 v[48:51], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[48:51] // left value = v[48+0:51+0]
v_mmac_f32_16x16x16_bf16 v[52:55], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[52:55] // left value = v[52+0:55+0]
v_mmac_f32_16x16x16_bf16 v[56:59], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[56:59] // left value = v[56+0:59+0]
v_mmac_f32_16x16x16_bf16 v[60:63], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[60:63] // left value = v[60+0:63+0]
s_waitcnt vmcnt(4)
s_barrier


/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x3              // counterL==1
s_cbranch_scc0 label_LoopBeginL_2                    // restart LoopL
s_branch label_LoopEndL_oddexit  /// unroll loop odditer exit


/******************************************/
/* Unrolled Loop - End 2/2 (final)        */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
label_LoopEndL_oddexit_31:
v_xor_b32 v[vgprLocalReadAddrA], 0x4000, v[vgprLocalReadAddrA] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrASwap], 0x4000, v[vgprLocalReadAddrASwap] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB], 0x4000, v[vgprLocalReadAddrB] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrBSwap], 0x4000, v[vgprLocalReadAddrBSwap] // swap Red Blk
s_branch label_LoopEndL_oddexit

label_LoopEndL_oddexit_32:
v_xor_b32 v[vgprLocalReadAddrA], 0x8000, v[vgprLocalReadAddrA] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrASwap], 0x8000, v[vgprLocalReadAddrASwap] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB], 0x8000, v[vgprLocalReadAddrB] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrBSwap], 0x8000, v[vgprLocalReadAddrBSwap] // swap Red Blk
s_branch label_LoopEndL_oddexit

label_LoopEndL_oddexit_33:
v_xor_b32 v[vgprLocalReadAddrA], 0xc000, v[vgprLocalReadAddrA] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrASwap], 0xc000, v[vgprLocalReadAddrASwap] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB], 0xc000, v[vgprLocalReadAddrB] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrBSwap], 0xc000, v[vgprLocalReadAddrBSwap] // swap Red Blk
s_branch label_LoopEndL_oddexit

label_LoopEndL_oddexit:  /// unroll loop odditer exit
label_LoopEndL:

//////////////////////////
//////////////////////////last
s_waitcnt vmcnt(0)
s_barrier
s_mov_b32 s64, 0


last_loop:
/* iter 0 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrASwap] offset:128 // ds_read_m Method for load lds data

/* local read b */
ds_read_m32x16_b16 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrBSwap] offset:256 // ds_read_m Method for load lds data


v_add_co_u32 v[vgprLocalReadAddrA], 0x4000, v[vgprLocalReadAddrA]
v_cmp_lt_u32 s[62:63], v[vgprLocalReadAddrA], s[sgprMask]
v_cndmask_b32 v[vgprLocalReadAddrA], v[vgprLocalReadAddrA_ori], v[vgprLocalReadAddrA], s[62:63]

v_add_co_u32 v[vgprLocalReadAddrB], 0x4000, v[vgprLocalReadAddrB]
v_cmp_lt_u32 s[62:63], v[vgprLocalReadAddrB], s[sgprMask]
v_cndmask_b32 v[vgprLocalReadAddrB], v[vgprLocalReadAddrB_ori], v[vgprLocalReadAddrB], s[62:63]

v_add_co_u32 v[vgprLocalReadAddrASwap], 0x4000, v[vgprLocalReadAddrASwap]
v_cmp_lt_u32 s[62:63], v[vgprLocalReadAddrASwap], s[sgprMask]
v_cndmask_b32 v[vgprLocalReadAddrASwap], v[vgprLocalReadAddrASwap_ori], v[vgprLocalReadAddrASwap], s[62:63]

v_add_co_u32 v[vgprLocalReadAddrBSwap], 0x4000, v[vgprLocalReadAddrBSwap]
v_cmp_lt_u32 s[62:63], v[vgprLocalReadAddrBSwap], s[sgprMask]
v_cndmask_b32 v[vgprLocalReadAddrBSwap], v[vgprLocalReadAddrBSwap_ori], v[vgprLocalReadAddrBSwap], s[62:63]

s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
//s_nop 1
v_mmac_f32_16x16x16_bf16 v[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
v_mmac_f32_16x16x16_bf16 v[4:7], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
v_mmac_f32_16x16x16_bf16 v[8:11], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
v_mmac_f32_16x16x16_bf16 v[12:15], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[12:15] // left value = v[12+0:15+0]
v_mmac_f32_16x16x16_bf16 v[16:19], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[16:19] // left value = v[16+0:19+0]
v_mmac_f32_16x16x16_bf16 v[20:23], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[20:23] // left value = v[20+0:23+0]
v_mmac_f32_16x16x16_bf16 v[24:27], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[24:27] // left value = v[24+0:27+0]
v_mmac_f32_16x16x16_bf16 v[28:31], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[28:31] // left value = v[28+0:31+0]
v_mmac_f32_16x16x16_bf16 v[32:35], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[32:35] // left value = v[32+0:35+0]
v_mmac_f32_16x16x16_bf16 v[36:39], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[36:39] // left value = v[36+0:39+0]
v_mmac_f32_16x16x16_bf16 v[40:43], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[40:43] // left value = v[40+0:43+0]
v_mmac_f32_16x16x16_bf16 v[44:47], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[44:47] // left value = v[44+0:47+0]
v_mmac_f32_16x16x16_bf16 v[48:51], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[48:51] // left value = v[48+0:51+0]
v_mmac_f32_16x16x16_bf16 v[52:55], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[52:55] // left value = v[52+0:55+0]
v_mmac_f32_16x16x16_bf16 v[56:59], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[56:59] // left value = v[56+0:59+0]
v_mmac_f32_16x16x16_bf16 v[60:63], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[60:63] // left value = v[60+0:63+0]

s_add_u32 s64, s64, 1
s_cmp_lt_u32 s64, s[sgprLoopCounterL]
s_cbranch_scc1 last_loop

s_waitcnt vmcnt(0)
s_barrier
/******************************************/
/* Opt. NoLoadLoop - Begin                */
/******************************************/
s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 label_OptNLL_End                    // Branch if Beta is not zero

s_cmp_eq_u32 s[sgprAlpha], 1.0                     // Alpha == 1.0 ?
s_cbranch_scc0 label_OptNLL_End                    // branch if alpha != 1

s_and_b32 s56, 127, s[sgprSizeI]                   // s56 = s[sgprSizeI] % 128
s_add_u32 s57, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s57                // wg0 >= nwg0-1 ?
s_cselect_b32 s56, s56, 0                          // set rMT0
s_cmpk_gt_u32 s56, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required
s_and_b32 s56, 255, s[sgprSizeJ]                   // s56 = s[sgprSizeJ] % 256
s_add_u32 s57, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s57                // wg1 >= nwg1-1
s_cselect_b32 s56, s56, 0                          // set rMT1
s_cmpk_gt_u32 s56, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required

s_and_b32 s57, 15, s[sgprSizesSum+0]               // s57 = s[sgprSizesSum+0] % 16
s_cmp_eq_u32 s57, 0x0                              // numIterL == 0
s_cbranch_scc0 label_OptNLL_End                    // skip if tail loop required

/* Stores for OptNLL */
label_Summation_End_OptNLL:
/* endSummation: add vgpr [64...94) to pool */
/* load store sgprs */
.set sgprAddressScaleAlphaVec, 44
.set sgprAddressBias, 46
.set sgprBiasType, 48
.set sgprBiasStride, 49
.set sgpractivationAlpha, 50
.set sgpractivationBeta, 51
.set sgprActivationType, 52
/* Check if custom structure pointer is null */
s_cmp_eq_u64 s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0 // s[ExternalArgAddress] == 0 ?
s_cbranch_scc0 label_LoadExternalEpilogueStruct    // branch if s[ExternalArgAddress] != 0
s_load_dwordx8 s[44:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_load_dword s52, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x78
s_branch label_LoadExternalEpilogueStructEnd
label_LoadExternalEpilogueStruct:
s_load_dwordx4 s[44:47], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x90
s_load_dwordx2 s[48:49], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xa0
s_load_dwordx2 s[50:51], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xb8
s_load_dword s52, s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xc0
label_LoadExternalEpilogueStructEnd:
.set sgprSrdScaleAlphaVec, 56
.set sgprSrdBias, 60

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */
/* computeStoreVgprs */
v_lshrrev_b32 v68, 6, v[vgprSerial]                // v68 = v[vgprSerial] / 64
v_lshrrev_b32 v69, 1, v68                          // v69 = v68 / 2
s_mov_b32 s43, 0x20                                // tempStore tmpSgpr
v_mul_lo_u32 v69, s43, v69                         // wave coordination offset 1
v_and_b32 v65, 63, v[vgprSerial]                   // v65 = v[vgprSerial] % 64
v_lshrrev_b32 v65, 4, v65                          // v65 = v65 / 16
v_add_lshl_u32 v65, v69, v65, 0                    // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v66, v65, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v67, v65, s[sgprStrideD1J]            //  offset 1
v_and_b32 v64, 1, v68                              // v64 = v68 % 2
s_mov_b32 s43, 0x20                                // tempStore tmpSgpr
v_mul_lo_u32 v64, s43, v64                         // wave coordination offset 0
v_and_b32 v69, 15, v[vgprSerial]                   // v69 = v[vgprSerial] % 16
v_add_lshl_u32 v64, v69, v64, 0                    // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s43, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_add_u32 v64, s43, v64                            // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s43, 256, s[sgprWorkGroup1]              // wgp1 * MT1
v_add_u32 v65, s43, v65                            // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_waitcnt lgkmcnt(0)                               // wait for 36 bytes of kern args.
s_mov_b32 s[sgprSrdScaleAlphaVec+0], s[sgprAddressScaleAlphaVec+0] // init SRD base address (lower)
s_mov_b32 s[sgprSrdScaleAlphaVec+1], s[sgprAddressScaleAlphaVec+1] // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdScaleAlphaVec+3], Srd127_96     // Set bits 127_96 in post-loop SRD
s_cmp_eq_u64 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], 0 // s[AddressScaleAlphaVec] == 0 ?
s_cbranch_scc0 label_ScaleAlphaVecAddrValid        // branch if s[AddressScaleAlphaVec] != 0
s_mov_b32 s[sgprSrdScaleAlphaVec+2], 0
s_branch label_ScaleAlphaVecAddrValid_End
label_ScaleAlphaVecAddrValid:
s_mov_b32 s[sgprSrdScaleAlphaVec+2], s[sgprSizeI]
label_ScaleAlphaVecAddrValid_End:

s_mul_i32 s[sgprSrdScaleAlphaVec+2], 0x4, s[sgprSrdScaleAlphaVec+2] // ScaleAlphaVec scaled by BPE
s_add_u32 s43, s[sgprWorkGroup2], 0x1
s_mul_i32 s43, s[sgprBiasStride], s43              // stride * (wg+1)
s_cmp_eq_u32 s43, 0x0                              // bias stride = 0?
s_cselect_b32 s43, s[sgprSizeI], s43
s_mov_b32 s[sgprSrdBias+0], s[sgprAddressBias+0]   // init SRD base address (lower)
s_mov_b32 s[sgprSrdBias+1], s[sgprAddressBias+1]   // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdBias+3], Srd127_96              // Set bits 127_96 in post-loop SRD
s_cmp_eq_u64 s[sgprAddressBias:sgprAddressBias+1], 0 // s[AddressBias] == 0 ?
s_cbranch_scc0 label_BiasAddrValid                 // branch if s[AddressBias] != 0
s_mov_b32 s[sgprSrdBias+2], 0
s_branch label_BiasAddrValid_End
label_BiasAddrValid:
s_mov_b32 s[sgprSrdBias+2], s43
label_BiasAddrValid_End:

label_Load_Biasf32:
s_cmpk_lg_u32 s[sgprBiasType], 0                   // BiasType != 0
s_cbranch_scc1 label_Load_Biasbf16                 // Branch if true

/******************************************/
/* Read Bias to LDS                       */
/******************************************/
s_mul_i32 s[sgprSrdBias+2], 0x4, s[sgprSrdBias+2]  // scaled by BPE
s_mul_i32 s43, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_add_u32 v72, s43, v[vgprSerial]                  // coord 0 = wgp0 * MT0 + thread offset
s_mul_i32 s43, s[sgprBiasStride], s[sgprWorkGroup2] // Stride * WG
v_add_u32 v72, s43, v72                            // coord 0 = wgp0 * MT0 + thread offset + Stride * WG
v_lshlrev_b32 v72, 0x2, v72                        // Global bias address scaled by BPE
buffer_load_dword v68, v72, s[sgprSrdBias:sgprSrdBias+3], 0 offen offset:0 // load bias
v_lshlrev_b32 v72, 0x2, v[vgprSerial]              // Local bias address scaled by BPE
s_waitcnt vmcnt(0)                                 // wait for bias load
s_barrier                                          // Wait for all wavefronts
ds_write_b32 v72, v68 offset:0                     // store bias
s_branch label_Load_Bias_End                       // Branch to load bias end
label_Load_Biasbf16:
s_cmpk_lg_u32 s[sgprBiasType], 7                   // BiasType != 7
s_cbranch_scc1 label_Load_Bias_End                 // Branch if true

/******************************************/
/* Read Bias to LDS                       */
/******************************************/
s_mul_i32 s[sgprSrdBias+2], 0x2, s[sgprSrdBias+2]  // scaled by BPE
s_mul_i32 s43, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_add_u32 v72, s43, v[vgprSerial]                  // coord 0 = wgp0 * MT0 + thread offset
s_mul_i32 s43, s[sgprBiasStride], s[sgprWorkGroup2] // Stride * WG
v_add_u32 v72, s43, v72                            // coord 0 = wgp0 * MT0 + thread offset + Stride * WG
v_lshlrev_b32 v72, 0x1, v72                        // Global bias address scaled by BPE
buffer_load_short_d16 v68, v72, s[sgprSrdBias:sgprSrdBias+3], 0 offen offset:0 // load bias
v_lshlrev_b32 v72, 0x2, v[vgprSerial]              // Local bias address scaled by BPE
s_waitcnt vmcnt(0)                                 // wait for bias load
s_barrier                                          // Wait for all wavefronts
v_lshlrev_b32 v68, 16, v68                         // convert bf16 to fp32
ds_write_b32 v72, v68 offset:0                     // store bias
s_branch label_Load_Bias_End                       // Branch to load bias end
label_Load_Bias_End:


s_waitcnt lgkmcnt(0)                               // Wait for Bias LDS write
s_barrier                                          // Bias LDS write barrier

/******************************************/
/* Read ScaleAlphaVec to LDS              */
/******************************************/
s_mul_i32 s43, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_add_u32 v72, s43, v[vgprSerial]                  // coord 0 = wgp0 * MT0 + thread offset
//s_mul_i32 s43, s[sgprBiasStride], s[sgprWorkGroup2] // Stride * WG
//v_add_u32 v72, s43, v72                            // coord 0 = wgp0 * MT0 + thread offset + Stride * WG
v_lshlrev_b32 v72, 0x2, v72                        // Global ScaleAlphaVec address scaled by BPE
buffer_load_dword v68, v72, s[sgprSrdScaleAlphaVec:sgprSrdScaleAlphaVec+3], 0 offen offset:0 // load ScaleAlphaVec
v_lshlrev_b32 v72, 0x2, v[vgprSerial]              // Local ScaleAlphaVec address scaled by BPE
s_waitcnt vmcnt(0)                                 // wait for ScaleAlphaVec load
s_barrier                                          // Wait for all wavefronts
ds_write_b32 v72, v68 offset:0x1000                // store ScaleAlphaVec
s_branch label_Load_ScaleAlphaVec_End              // Branch to load ScaleAlphaVec end
label_Load_ScaleAlphaVec_End:

s_waitcnt lgkmcnt(0)                               // Wait for ScaleAlphaVec LDS write
s_barrier                                          // ScaleAlphaVec LDS write barrier

/******************************************/
/* Read Bias & ScaleAlphaVec from LDS     */
/******************************************/
s_mul_i32 s66, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v76, v64, s66
v_lshlrev_b32 v76, 0x2, v76                        // Bias & ScaleAlphaVec address scaled by BPE

ds_read_b32 v[vgprValuBias_M+0], v76 offset:0                      // load bias
ds_read_b32 v[vgprValuBias_M+1], v76 offset:64                     // load bias
ds_read_b32 v[vgprValuBias_M+2], v76 offset:256                    // load bias
ds_read_b32 v[vgprValuBias_M+3], v76 offset:320                    // load bias

s_mov_b32 s66, 4096
v_add_co_u32 v76, vcc, v76, s66

ds_read_b32 v[vgprValuScalAlpha_M+0], v76 offset:0                 // load ScaleAlphaVec
ds_read_b32 v[vgprValuScalAlpha_M+1], v76 offset:64                // load ScaleAlphaVec
ds_read_b32 v[vgprValuScalAlpha_M+2], v76 offset:256               // load ScaleAlphaVec
ds_read_b32 v[vgprValuScalAlpha_M+3], v76 offset:320               // load ScaleAlphaVec

s_waitcnt lgkmcnt(0)                               // Wait for Bias LDS write
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v[vgprValuScalAlpha_M+0], 1.0, v[vgprValuScalAlpha_M+0], s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v[vgprValuScalAlpha_M+1], 1.0, v[vgprValuScalAlpha_M+1], s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v[vgprValuScalAlpha_M+2], 1.0, v[vgprValuScalAlpha_M+2], s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v[vgprValuScalAlpha_M+3], 1.0, v[vgprValuScalAlpha_M+3], s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0


s_cmpk_eq_u32 s[sgprActivationType], 1             // activationType == 1
s_cbranch_scc1 label_To_Activation_Abs_VW1         // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 2             // activationType == 2
s_cbranch_scc1 label_To_Activation_Clippedrelu_VW1 // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 3             // activationType == 3
s_cbranch_scc1 label_To_Activation_Gelu_VW1        // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 4             // activationType == 4
s_cbranch_scc1 label_To_Activation_Leakyrelu_VW1   // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 5             // activationType == 5
s_cbranch_scc1 label_To_Activation_Relu_VW1        // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 6             // activationType == 6
s_cbranch_scc1 label_To_Activation_Sigmoid_VW1     // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 7             // activationType == 7
s_cbranch_scc1 label_To_Activation_Tanh_VW1        // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 9             // activationType == 9
s_cbranch_scc1 label_To_Activation_Geluscaling_VW1 // Branch if true
label_To_Activation_None_VW1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_None_VW1, 0x4      // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Abs_VW1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Abs_VW1, 0x4       // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Clippedrelu_VW1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Clippedrelu_VW1, 0x4 // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Gelu_VW1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Gelu_VW1, 0x4      // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Leakyrelu_VW1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Leakyrelu_VW1, 0x4 // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Relu_VW1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Relu_VW1, 0x4      // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Sigmoid_VW1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Sigmoid_VW1, 0x4   // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Tanh_VW1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Tanh_VW1, 0x4      // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Geluscaling_VW1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Geluscaling_VW1, 0x4 // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_ActivationSetPCAddrEnd:
label_GW_B0_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v74, v67, v64, 0x1                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64


/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16


v_mul_f32 v[vgprValuC+0], v[vgprValuScalAlpha_M+0], v[vgprValuC+0]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+0]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v0, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+0], v[vgprValuC+0] // check Nan
v_bfe_u32 v70, v[vgprValuC+0], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+0], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+0], v70, v72, s[66:67]
v_lshrrev_b32 v0, 16, v[vgprValuC+0]               // convert C to bf16
buffer_store_short v0, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D


v_mul_f32 v[vgprValuC+4], v[vgprValuScalAlpha_M+1], v[vgprValuC+4]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+4]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v4, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+4], v[vgprValuC+4] // check Nan
v_bfe_u32 v70, v[vgprValuC+4], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+4], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+4], v70, v72, s[66:67]
v_lshrrev_b32 v4, 16, v[vgprValuC+4]               // convert C to bf16
buffer_store_short v4, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D


v_mul_f32 v[vgprValuC+8], v[vgprValuScalAlpha_M+2], v[vgprValuC+8]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+8]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v8, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+8], v[vgprValuC+8] // check Nan
v_bfe_u32 v70, v[vgprValuC+8], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+8], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+8], v70, v72, s[66:67]
v_lshrrev_b32 v8, 16, v[vgprValuC+8]               // convert C to bf16
buffer_store_short v8, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D


v_mul_f32 v[vgprValuC+12], v[vgprValuScalAlpha_M+3], v[vgprValuC+12]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+12]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v12, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+12], v[vgprValuC+12] // check Nan
v_bfe_u32 v70, v[vgprValuC+12], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+12], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+12], v70, v72, s[66:67]
v_lshrrev_b32 v12, 16, v[vgprValuC+12]             // convert C to bf16
buffer_store_short v12, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+1], v[vgprValuScalAlpha_M+0], v[vgprValuC+1]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+1]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v1, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+1], v[vgprValuC+1] // check Nan
v_bfe_u32 v70, v[vgprValuC+1], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+1], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+1], v70, v72, s[66:67]
v_lshrrev_b32 v1, 16, v[vgprValuC+1]               // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v1, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+5], v[vgprValuScalAlpha_M+1], v[vgprValuC+5]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+5]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v5, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+5], v[vgprValuC+5] // check Nan
v_bfe_u32 v70, v[vgprValuC+5], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+5], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+5], v70, v72, s[66:67]
v_lshrrev_b32 v5, 16, v[vgprValuC+5]               // convert C to bf16
buffer_store_short v5, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+9], v[vgprValuScalAlpha_M+2], v[vgprValuC+9]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+9]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v9, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+9], v[vgprValuC+9] // check Nan
v_bfe_u32 v70, v[vgprValuC+9], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+9], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+9], v70, v72, s[66:67]
v_lshrrev_b32 v9, 16, v[vgprValuC+9]               // convert C to bf16
buffer_store_short v9, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+13], v[vgprValuScalAlpha_M+3], v[vgprValuC+13]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+13]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v13, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+13], v[vgprValuC+13] // check Nan
v_bfe_u32 v70, v[vgprValuC+13], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+13], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+13], v70, v72, s[66:67]
v_lshrrev_b32 v13, 16, v[vgprValuC+13]             // convert C to bf16
buffer_store_short v13, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+2], v[vgprValuScalAlpha_M+0], v[vgprValuC+2]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+2]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v2, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+2], v[vgprValuC+2] // check Nan
v_bfe_u32 v70, v[vgprValuC+2], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+2], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+2], v70, v72, s[66:67]
v_lshrrev_b32 v2, 16, v[vgprValuC+2]               // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v2, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+6], v[vgprValuScalAlpha_M+1], v[vgprValuC+6]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+6]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v6, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+6], v[vgprValuC+6] // check Nan
v_bfe_u32 v70, v[vgprValuC+6], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+6], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+6], v70, v72, s[66:67]
v_lshrrev_b32 v6, 16, v[vgprValuC+6]               // convert C to bf16
buffer_store_short v6, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+10], v[vgprValuScalAlpha_M+2], v[vgprValuC+10]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+10]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v10, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+10], v[vgprValuC+10] // check Nan
v_bfe_u32 v70, v[vgprValuC+10], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+10], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+10], v70, v72, s[66:67]
v_lshrrev_b32 v10, 16, v[vgprValuC+10]             // convert C to bf16
buffer_store_short v10, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+14], v[vgprValuScalAlpha_M+3], v[vgprValuC+14]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+14]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v14, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+14], v[vgprValuC+14] // check Nan
v_bfe_u32 v70, v[vgprValuC+14], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+14], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+14], v70, v72, s[66:67]
v_lshrrev_b32 v14, 16, v[vgprValuC+14]             // convert C to bf16
buffer_store_short v14, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+3], v[vgprValuScalAlpha_M+0], v[vgprValuC+3]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+3]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v3, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+3], v[vgprValuC+3] // check Nan
v_bfe_u32 v70, v[vgprValuC+3], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+3], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+3], v70, v72, s[66:67]
v_lshrrev_b32 v3, 16, v[vgprValuC+3]               // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v3, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+7], v[vgprValuScalAlpha_M+1], v[vgprValuC+7]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+7]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v7, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+7], v[vgprValuC+7] // check Nan
v_bfe_u32 v70, v[vgprValuC+7], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+7], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+7], v70, v72, s[66:67]
v_lshrrev_b32 v7, 16, v[vgprValuC+7]               // convert C to bf16
buffer_store_short v7, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+11], v[vgprValuScalAlpha_M+2], v[vgprValuC+11]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+11]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v11, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+11], v[vgprValuC+11] // check Nan
v_bfe_u32 v70, v[vgprValuC+11], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+11], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+11], v70, v72, s[66:67]
v_lshrrev_b32 v11, 16, v[vgprValuC+11]             // convert C to bf16
buffer_store_short v11, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+15], v[vgprValuScalAlpha_M+3], v[vgprValuC+15]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+15]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v15, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+15], v[vgprValuC+15] // check Nan
v_bfe_u32 v70, v[vgprValuC+15], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+15], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+15], v70, v72, s[66:67]
v_lshrrev_b32 v15, 16, v[vgprValuC+15]             // convert C to bf16
buffer_store_short v15, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #1 (d1,d0,vc1,vc0) = */
/*    (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16


v_mul_f32 v[vgprValuC+16], v[vgprValuScalAlpha_M+0], v[vgprValuC+16]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+16]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v16, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+16], v[vgprValuC+16] // check Nan
v_bfe_u32 v70, v[vgprValuC+16], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+16], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+16], v70, v72, s[66:67]
v_lshrrev_b32 v16, 16, v[vgprValuC+16]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v16, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D


v_mul_f32 v[vgprValuC+20], v[vgprValuScalAlpha_M+1], v[vgprValuC+20]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+20]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v20, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+20], v[vgprValuC+20] // check Nan
v_bfe_u32 v70, v[vgprValuC+20], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+20], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+20], v70, v72, s[66:67]
v_lshrrev_b32 v20, 16, v[vgprValuC+20]             // convert C to bf16
buffer_store_short v20, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D


v_mul_f32 v[vgprValuC+24], v[vgprValuScalAlpha_M+2], v[vgprValuC+24]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+24]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v24, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+24], v[vgprValuC+24] // check Nan
v_bfe_u32 v70, v[vgprValuC+24], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+24], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+24], v70, v72, s[66:67]
v_lshrrev_b32 v24, 16, v[vgprValuC+24]             // convert C to bf16
buffer_store_short v24, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D


v_mul_f32 v[vgprValuC+28], v[vgprValuScalAlpha_M+3], v[vgprValuC+28]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+28]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v28, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+28], v[vgprValuC+28] // check Nan
v_bfe_u32 v70, v[vgprValuC+28], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+28], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+28], v70, v72, s[66:67]
v_lshrrev_b32 v28, 16, v[vgprValuC+28]             // convert C to bf16
buffer_store_short v28, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+17], v[vgprValuScalAlpha_M+0], v[vgprValuC+17]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+17]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v17, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+17], v[vgprValuC+17] // check Nan
v_bfe_u32 v70, v[vgprValuC+17], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+17], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+17], v70, v72, s[66:67]
v_lshrrev_b32 v17, 16, v[vgprValuC+17]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v17, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+21], v[vgprValuScalAlpha_M+1], v[vgprValuC+21]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+21]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v21, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+21], v[vgprValuC+21] // check Nan
v_bfe_u32 v70, v[vgprValuC+21], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+21], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+21], v70, v72, s[66:67]
v_lshrrev_b32 v21, 16, v[vgprValuC+21]             // convert C to bf16
buffer_store_short v21, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+25], v[vgprValuScalAlpha_M+2], v[vgprValuC+25]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+25]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v25, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+25], v[vgprValuC+25] // check Nan
v_bfe_u32 v70, v[vgprValuC+25], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+25], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+25], v70, v72, s[66:67]
v_lshrrev_b32 v25, 16, v[vgprValuC+25]             // convert C to bf16
buffer_store_short v25, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+29], v[vgprValuScalAlpha_M+3], v[vgprValuC+29]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+29]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v29, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+29], v[vgprValuC+29] // check Nan
v_bfe_u32 v70, v[vgprValuC+29], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+29], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+29], v70, v72, s[66:67]
v_lshrrev_b32 v29, 16, v[vgprValuC+29]             // convert C to bf16
buffer_store_short v29, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+18], v[vgprValuScalAlpha_M+0], v[vgprValuC+18]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+18]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v18, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+18], v[vgprValuC+18] // check Nan
v_bfe_u32 v70, v[vgprValuC+18], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+18], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+18], v70, v72, s[66:67]
v_lshrrev_b32 v18, 16, v[vgprValuC+18]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v18, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+22], v[vgprValuScalAlpha_M+1], v[vgprValuC+22]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+22]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v22, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+22], v[vgprValuC+22] // check Nan
v_bfe_u32 v70, v[vgprValuC+22], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+22], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+22], v70, v72, s[66:67]
v_lshrrev_b32 v22, 16, v[vgprValuC+22]             // convert C to bf16
buffer_store_short v22, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+26], v[vgprValuScalAlpha_M+2], v[vgprValuC+26]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+26]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v26, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+26], v[vgprValuC+26] // check Nan
v_bfe_u32 v70, v[vgprValuC+26], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+26], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+26], v70, v72, s[66:67]
v_lshrrev_b32 v26, 16, v[vgprValuC+26]             // convert C to bf16
buffer_store_short v26, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+30], v[vgprValuScalAlpha_M+3], v[vgprValuC+30]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+30]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v30, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+30], v[vgprValuC+30] // check Nan
v_bfe_u32 v70, v[vgprValuC+30], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+30], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+30], v70, v72, s[66:67]
v_lshrrev_b32 v30, 16, v[vgprValuC+30]             // convert C to bf16
buffer_store_short v30, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+19], v[vgprValuScalAlpha_M+0], v[vgprValuC+19]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+19]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v19, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+19], v[vgprValuC+19] // check Nan
v_bfe_u32 v70, v[vgprValuC+19], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+19], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+19], v70, v72, s[66:67]
v_lshrrev_b32 v19, 16, v[vgprValuC+19]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v19, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+23], v[vgprValuScalAlpha_M+1], v[vgprValuC+23]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+23]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v23, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+23], v[vgprValuC+23] // check Nan
v_bfe_u32 v70, v[vgprValuC+23], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+23], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+23], v70, v72, s[66:67]
v_lshrrev_b32 v23, 16, v[vgprValuC+23]             // convert C to bf16
buffer_store_short v23, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+27], v[vgprValuScalAlpha_M+2], v[vgprValuC+27]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+27]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v27, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+27], v[vgprValuC+27] // check Nan
v_bfe_u32 v70, v[vgprValuC+27], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+27], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+27], v70, v72, s[66:67]
v_lshrrev_b32 v27, 16, v[vgprValuC+27]             // convert C to bf16
buffer_store_short v27, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+31], v[vgprValuScalAlpha_M+3], v[vgprValuC+31]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+31]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v31, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+31], v[vgprValuC+31] // check Nan
v_bfe_u32 v70, v[vgprValuC+31], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+31], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+31], v70, v72, s[66:67]
v_lshrrev_b32 v31, 16, v[vgprValuC+31]             // convert C to bf16
buffer_store_short v31, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #2 (d1,d0,vc1,vc0) = */
/*    (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (8,3,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (9,3,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (10,3,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (11,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(8,0,0,0) */

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16


v_mul_f32 v[vgprValuC+32], v[vgprValuScalAlpha_M+0], v[vgprValuC+32]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+32]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v32, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+32], v[vgprValuC+32] // check Nan
v_bfe_u32 v70, v[vgprValuC+32], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+32], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+32], v70, v72, s[66:67]
v_lshrrev_b32 v32, 16, v[vgprValuC+32]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 200                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v32, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D


v_mul_f32 v[vgprValuC+36], v[vgprValuScalAlpha_M+1], v[vgprValuC+36]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+36]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v36, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+36], v[vgprValuC+36] // check Nan
v_bfe_u32 v70, v[vgprValuC+36], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+36], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+36], v70, v72, s[66:67]
v_lshrrev_b32 v36, 16, v[vgprValuC+36]             // convert C to bf16
buffer_store_short v36, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D


v_mul_f32 v[vgprValuC+40], v[vgprValuScalAlpha_M+2], v[vgprValuC+40]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+40]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v40, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+40], v[vgprValuC+40] // check Nan
v_bfe_u32 v70, v[vgprValuC+40], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+40], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+40], v70, v72, s[66:67]
v_lshrrev_b32 v40, 16, v[vgprValuC+40]             // convert C to bf16
buffer_store_short v40, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D


v_mul_f32 v[vgprValuC+44], v[vgprValuScalAlpha_M+3], v[vgprValuC+44]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+44]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v44, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+44], v[vgprValuC+44] // check Nan
v_bfe_u32 v70, v[vgprValuC+44], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+44], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+44], v70, v72, s[66:67]
v_lshrrev_b32 v44, 16, v[vgprValuC+44]             // convert C to bf16
buffer_store_short v44, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+33], v[vgprValuScalAlpha_M+0], v[vgprValuC+33]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+33]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v33, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+33], v[vgprValuC+33] // check Nan
v_bfe_u32 v70, v[vgprValuC+33], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+33], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+33], v70, v72, s[66:67]
v_lshrrev_b32 v33, 16, v[vgprValuC+33]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v33, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+37], v[vgprValuScalAlpha_M+1], v[vgprValuC+37]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+37]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v37, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+37], v[vgprValuC+37] // check Nan
v_bfe_u32 v70, v[vgprValuC+37], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+37], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+37], v70, v72, s[66:67]
v_lshrrev_b32 v37, 16, v[vgprValuC+37]             // convert C to bf16
buffer_store_short v37, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+41], v[vgprValuScalAlpha_M+2], v[vgprValuC+41]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+41]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v41, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+41], v[vgprValuC+41] // check Nan
v_bfe_u32 v70, v[vgprValuC+41], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+41], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+41], v70, v72, s[66:67]
v_lshrrev_b32 v41, 16, v[vgprValuC+41]             // convert C to bf16
buffer_store_short v41, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+45], v[vgprValuScalAlpha_M+3], v[vgprValuC+45]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+45]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v45, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+45], v[vgprValuC+45] // check Nan
v_bfe_u32 v70, v[vgprValuC+45], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+45], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+45], v70, v72, s[66:67]
v_lshrrev_b32 v45, 16, v[vgprValuC+45]             // convert C to bf16
buffer_store_short v45, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+34], v[vgprValuScalAlpha_M+0], v[vgprValuC+34]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+34]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v34, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+34], v[vgprValuC+34] // check Nan
v_bfe_u32 v70, v[vgprValuC+34], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+34], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+34], v70, v72, s[66:67]
v_lshrrev_b32 v34, 16, v[vgprValuC+34]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v34, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+38], v[vgprValuScalAlpha_M+1], v[vgprValuC+38]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+38]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v38, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+38], v[vgprValuC+38] // check Nan
v_bfe_u32 v70, v[vgprValuC+38], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+38], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+38], v70, v72, s[66:67]
v_lshrrev_b32 v38, 16, v[vgprValuC+38]             // convert C to bf16
buffer_store_short v38, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+42], v[vgprValuScalAlpha_M+2], v[vgprValuC+42]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+42]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v42, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+42], v[vgprValuC+42] // check Nan
v_bfe_u32 v70, v[vgprValuC+42], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+42], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+42], v70, v72, s[66:67]
v_lshrrev_b32 v42, 16, v[vgprValuC+42]             // convert C to bf16
buffer_store_short v42, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+46], v[vgprValuScalAlpha_M+3], v[vgprValuC+46]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+46]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v46, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+46], v[vgprValuC+46] // check Nan
v_bfe_u32 v70, v[vgprValuC+46], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+46], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+46], v70, v72, s[66:67]
v_lshrrev_b32 v46, 16, v[vgprValuC+46]             // convert C to bf16
buffer_store_short v46, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+35], v[vgprValuScalAlpha_M+0], v[vgprValuC+35]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+35]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v35, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+35], v[vgprValuC+35] // check Nan
v_bfe_u32 v70, v[vgprValuC+35], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+35], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+35], v70, v72, s[66:67]
v_lshrrev_b32 v35, 16, v[vgprValuC+35]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v35, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+39], v[vgprValuScalAlpha_M+1], v[vgprValuC+39]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+39]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v39, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+39], v[vgprValuC+39] // check Nan
v_bfe_u32 v70, v[vgprValuC+39], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+39], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+39], v70, v72, s[66:67]
v_lshrrev_b32 v39, 16, v[vgprValuC+39]             // convert C to bf16
buffer_store_short v39, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+43], v[vgprValuScalAlpha_M+2], v[vgprValuC+43]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+43]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v43, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+43], v[vgprValuC+43] // check Nan
v_bfe_u32 v70, v[vgprValuC+43], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+43], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+43], v70, v72, s[66:67]
v_lshrrev_b32 v43, 16, v[vgprValuC+43]             // convert C to bf16
buffer_store_short v43, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+47], v[vgprValuScalAlpha_M+3], v[vgprValuC+47]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+47]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v47, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+47], v[vgprValuC+47] // check Nan
v_bfe_u32 v70, v[vgprValuC+47], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+47], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+47], v70, v72, s[66:67]
v_lshrrev_b32 v47, 16, v[vgprValuC+47]             // convert C to bf16
buffer_store_short v47, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #3 (d1,d0,vc1,vc0) = */
/*    (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (12,3,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (13,3,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (14,3,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (15,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16


v_mul_f32 v[vgprValuC+48], v[vgprValuScalAlpha_M+0], v[vgprValuC+48]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+48]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v48, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+48], v[vgprValuC+48] // check Nan
v_bfe_u32 v70, v[vgprValuC+48], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+48], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+48], v70, v72, s[66:67]
v_lshrrev_b32 v48, 16, v[vgprValuC+48]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v48, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D


v_mul_f32 v[vgprValuC+52], v[vgprValuScalAlpha_M+1], v[vgprValuC+52]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+52]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v52, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+52], v[vgprValuC+52] // check Nan
v_bfe_u32 v70, v[vgprValuC+52], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+52], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+52], v70, v72, s[66:67]
v_lshrrev_b32 v52, 16, v[vgprValuC+52]             // convert C to bf16
buffer_store_short v52, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D


v_mul_f32 v[vgprValuC+56], v[vgprValuScalAlpha_M+2], v[vgprValuC+56]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+56]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v56, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+56], v[vgprValuC+56] // check Nan
v_bfe_u32 v70, v[vgprValuC+56], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+56], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+56], v70, v72, s[66:67]
v_lshrrev_b32 v56, 16, v[vgprValuC+56]             // convert C to bf16
buffer_store_short v56, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D


v_mul_f32 v[vgprValuC+60], v[vgprValuScalAlpha_M+3], v[vgprValuC+60]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+60]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v60, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+60], v[vgprValuC+60] // check Nan
v_bfe_u32 v70, v[vgprValuC+60], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+60], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+60], v70, v72, s[66:67]
v_lshrrev_b32 v60, 16, v[vgprValuC+60]             // convert C to bf16
buffer_store_short v60, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+49], v[vgprValuScalAlpha_M+0], v[vgprValuC+49]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+49]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v49, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+49], v[vgprValuC+49] // check Nan
v_bfe_u32 v70, v[vgprValuC+49], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+49], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+49], v70, v72, s[66:67]
v_lshrrev_b32 v49, 16, v[vgprValuC+49]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v49, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+53], v[vgprValuScalAlpha_M+1], v[vgprValuC+53]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+53]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v53, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+53], v[vgprValuC+53] // check Nan
v_bfe_u32 v70, v[vgprValuC+53], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+53], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+53], v70, v72, s[66:67]
v_lshrrev_b32 v53, 16, v[vgprValuC+53]             // convert C to bf16
buffer_store_short v53, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+57], v[vgprValuScalAlpha_M+2], v[vgprValuC+57]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+57]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v57, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+57], v[vgprValuC+57] // check Nan
v_bfe_u32 v70, v[vgprValuC+57], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+57], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+57], v70, v72, s[66:67]
v_lshrrev_b32 v57, 16, v[vgprValuC+57]             // convert C to bf16
buffer_store_short v57, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+61], v[vgprValuScalAlpha_M+3], v[vgprValuC+61]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+61]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v61, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+61], v[vgprValuC+61] // check Nan
v_bfe_u32 v70, v[vgprValuC+61], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+61], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+61], v70, v72, s[66:67]
v_lshrrev_b32 v61, 16, v[vgprValuC+61]             // convert C to bf16
buffer_store_short v61, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+50], v[vgprValuScalAlpha_M+0], v[vgprValuC+50]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+50]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v50, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+50], v[vgprValuC+50] // check Nan
v_bfe_u32 v70, v[vgprValuC+50], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+50], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+50], v70, v72, s[66:67]
v_lshrrev_b32 v50, 16, v[vgprValuC+50]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v50, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+54], v[vgprValuScalAlpha_M+1], v[vgprValuC+54]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+54]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v54, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+54], v[vgprValuC+54] // check Nan
v_bfe_u32 v70, v[vgprValuC+54], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+54], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+54], v70, v72, s[66:67]
v_lshrrev_b32 v54, 16, v[vgprValuC+54]             // convert C to bf16
buffer_store_short v54, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+58], v[vgprValuScalAlpha_M+2], v[vgprValuC+58]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+58]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v58, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+58], v[vgprValuC+58] // check Nan
v_bfe_u32 v70, v[vgprValuC+58], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+58], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+58], v70, v72, s[66:67]
v_lshrrev_b32 v58, 16, v[vgprValuC+58]             // convert C to bf16
buffer_store_short v58, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+62], v[vgprValuScalAlpha_M+3], v[vgprValuC+62]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+62]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v62, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+62], v[vgprValuC+62] // check Nan
v_bfe_u32 v70, v[vgprValuC+62], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+62], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+62], v70, v72, s[66:67]
v_lshrrev_b32 v62, 16, v[vgprValuC+62]             // convert C to bf16
buffer_store_short v62, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+51], v[vgprValuScalAlpha_M+0], v[vgprValuC+51]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+51]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v51, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+51], v[vgprValuC+51] // check Nan
v_bfe_u32 v70, v[vgprValuC+51], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+51], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+51], v70, v72, s[66:67]
v_lshrrev_b32 v51, 16, v[vgprValuC+51]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v51, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+55], v[vgprValuScalAlpha_M+1], v[vgprValuC+55]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+55]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v55, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+55], v[vgprValuC+55] // check Nan
v_bfe_u32 v70, v[vgprValuC+55], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+55], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+55], v70, v72, s[66:67]
v_lshrrev_b32 v55, 16, v[vgprValuC+55]             // convert C to bf16
buffer_store_short v55, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+59], v[vgprValuScalAlpha_M+2], v[vgprValuC+59]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+59]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v59, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+59], v[vgprValuC+59] // check Nan
v_bfe_u32 v70, v[vgprValuC+59], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+59], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+59], v70, v72, s[66:67]
v_lshrrev_b32 v59, 16, v[vgprValuC+59]             // convert C to bf16
buffer_store_short v59, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+63], v[vgprValuScalAlpha_M+3], v[vgprValuC+63]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+63]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v63, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+63], v[vgprValuC+63] // check Nan
v_bfe_u32 v70, v[vgprValuC+63], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+63], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+63], v70, v72, s[66:67]
v_lshrrev_b32 v63, 16, v[vgprValuC+63]             // convert C to bf16
buffer_store_short v63, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_End:

s_endpgm                                           // Kernel End
label_OptNLL_End:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

label_PrefetchGlobalLastIterEnd:

/******************************************/
/* Tail Loop                              */
/******************************************/

/* Tail: add ValuA/B vgpr buffer [64...144) to pool */

/* local write reset offsets a */
v_and_b32 v[vgprLocalWriteAddrA], 0xf03fff, v[vgprLocalWriteAddrA] // reset to Red
s_and_b32 s[sgprLocalWriteAddrA], 0xf03fff, s[sgprLocalWriteAddrA] // reset to Red

/* local write reset offsets b */
v_and_b32 v[vgprLocalWriteAddrB], 0xf03fff, v[vgprLocalWriteAddrB] // reset to Red
s_and_b32 s[sgprLocalWriteAddrB], 0xf03fff, s[sgprLocalWriteAddrB] // reset to Red

s_lshr_b32 s62, s[sgprWaveiD], 2
s_lshl_b32 s62, 18, s62
s_or_b32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA], s62
s_or_b32 s[sgprLocalWriteAddrB], s[sgprLocalWriteAddrB], s62

s_mov_b32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA_ori]  // 
s_mov_b32 s[sgprLocalWriteAddrB], s[sgprLocalWriteAddrB_ori]  // 

// numIterL = (((sizeL % LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 15, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 16
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0              // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

s_sub_i32 s62, s[sgprLoopCounterL], 0              // 
s_cmp_gt_u32 s62, 16                               // 
s_cselect_b32 s62, 16, s62                         // 
s_mul_i32 s62, 32, s62                             // 
v_cmp_lt_u32 s[64:65], v[vgprSerial], s62          // 
v_cndmask_b32 v[vgprGlobalReadOffsetA+0], -0x1, v[vgprGlobalReadOffsetA+0], s[64:65] // 

s_sub_i32 s62, s[sgprLoopCounterL], 0              // 
s_cmp_gt_u32 s62, 16                               // 
s_cselect_b32 s62, 16, s62                         // 
s_mul_i32 s62, 32, s62                             // 
v_cmp_lt_u32 s[64:65], v[vgprSerial], s62          // 
v_cndmask_b32 v[vgprGlobalReadOffsetB+0], -0x1, v[vgprGlobalReadOffsetB+0], s[64:65] // 
s_add_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 15 // tail add
s_lshr_b32 s[sgprLoopCounterL], s[sgprLoopCounterL], 4 // 


/* Update M0 for DTLDS */

/* global read A */
s_mov_b32 m0, s[sgprLocalWriteAddrA]
buffer_load_dwordx2 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0  lds// load one buffer value

/* Update M0 for DTLDS */

/* global read B */
s_mov_b32 m0, s[sgprLocalWriteAddrB]
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 lds// load one buffer value
s_waitcnt vmcnt(0)                                 // 2wait for global read
s_barrier


/* Recalc local read offsets */

/* local read reset offsets a */

/* localReadResetOffsets */
/* handled internally */
v_mov_b32 v[vgprLocalReadAddrA], v[vgprLocalReadAddrA_ori] // reset Red,Blk -> Red
v_mov_b32 v[vgprLocalReadAddrASwap], v[vgprLocalReadAddrASwap_ori] // reset Red,Blk -> Red

/* local read reset offsets b */

/* localReadResetOffsets */
/* handled internally */
v_mov_b32 v[vgprLocalReadAddrB], v[vgprLocalReadAddrB_ori] // reset Red,Blk -> Red
v_mov_b32 v[vgprLocalReadAddrBSwap], v[vgprLocalReadAddrBSwap_ori] // reset Red,Blk -> Red

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */

/* tail loop: macs */
label_TailLoopBeginL:

/* Tail: remove ValuA/B vgpr buffer [64...80) from pool */

/* Tail: add address/G2L vgpr [80...94) to pool */

/* local read a */
ds_read_m32x16_b16 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrASwap] offset:128 // ds_read_m Method for load lds data

/* local read b */
ds_read_m32x16_b16 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // ds_read_m Method for load lds data
ds_read_m32x16_b16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrBSwap] offset:256 // ds_read_m Method for load lds data

/* local read inc a */
s_mov_b32 s56, 0x1000                              // inc
v_add_co_u32 v[vgprLocalReadAddrA], vcc, s56, v[vgprLocalReadAddrA] // lrA += 8192 (LSU*(MT+PAD)*bpe)
v_add_co_u32 v[vgprLocalReadAddrASwap], vcc, s56, v[vgprLocalReadAddrASwap] // lrA += 8192 (LSU*(MT+PAD)*bpe)

/* local read inc b */
s_mov_b32 s56, 0x2000                              // inc
v_add_co_u32 v[vgprLocalReadAddrB], vcc, s56, v[vgprLocalReadAddrB] // lrB += 8192 (LSU*(MT+PAD)*bpe)
v_add_co_u32 v[vgprLocalReadAddrBSwap], vcc, s56, v[vgprLocalReadAddrBSwap] // lrB += 8192 (LSU*(MT+PAD)*bpe)
s_waitcnt lgkmcnt(0)                               // 4wait for local read

//s_nop 1
v_mmac_f32_16x16x16_bf16 v[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
v_mmac_f32_16x16x16_bf16 v[4:7], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
v_mmac_f32_16x16x16_bf16 v[8:11], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
v_mmac_f32_16x16x16_bf16 v[12:15], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[12:15] // left value = v[12+0:15+0]
v_mmac_f32_16x16x16_bf16 v[16:19], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[16:19] // left value = v[16+0:19+0]
v_mmac_f32_16x16x16_bf16 v[20:23], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[20:23] // left value = v[20+0:23+0]
v_mmac_f32_16x16x16_bf16 v[24:27], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[24:27] // left value = v[24+0:27+0]
v_mmac_f32_16x16x16_bf16 v[28:31], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[28:31] // left value = v[28+0:31+0]
v_mmac_f32_16x16x16_bf16 v[32:35], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[32:35] // left value = v[32+0:35+0]
v_mmac_f32_16x16x16_bf16 v[36:39], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[36:39] // left value = v[36+0:39+0]
v_mmac_f32_16x16x16_bf16 v[40:43], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[40:43] // left value = v[40+0:43+0]
v_mmac_f32_16x16x16_bf16 v[44:47], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[44:47] // left value = v[44+0:47+0]
v_mmac_f32_16x16x16_bf16 v[48:51], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[48:51] // left value = v[48+0:51+0]
v_mmac_f32_16x16x16_bf16 v[52:55], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[52:55] // left value = v[52+0:55+0]
v_mmac_f32_16x16x16_bf16 v[56:59], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[56:59] // left value = v[56+0:59+0]
v_mmac_f32_16x16x16_bf16 v[60:63], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[60:63] // left value = v[60+0:63+0]

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x1 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x1 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 label_TailLoopBeginL                // restart LoopL
label_TailLoopEndL:
label_SkipTailLoopL:

/* Tail: remove address/G2L [80...94) from pool */
label_Summation_End_1N6LYVALRN7FPA74_0:
/* endSummation: add vgpr [64...94) to pool */
.set sgprGlobalReadIncsB, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitB, UNDEF
.set sgprLocalWriteAddrA, UNDEF
.set sgprLocalWriteAddrB, UNDEF
/* load store sgprs */
.set sgprAddressScaleAlphaVec, 44
.set sgprAddressBias, 46
.set sgprBiasType, 48
.set sgprBiasStride, 49
.set sgpractivationAlpha, 50
.set sgpractivationBeta, 51
.set sgprActivationType, 52
/* Check if custom structure pointer is null */
s_cmp_eq_u64 s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0 // s[ExternalArgAddress] == 0 ?
s_cbranch_scc0 label_LoadExternalEpilogueStruct_1  // branch if s[ExternalArgAddress] != 0
s_load_dwordx8 s[44:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_load_dword s52, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x78
s_branch label_LoadExternalEpilogueStructEnd_1
label_LoadExternalEpilogueStruct_1:
s_load_dwordx4 s[44:47], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x90
s_load_dwordx2 s[48:49], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xa0
s_load_dwordx2 s[50:51], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xb8
s_load_dword s52, s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xc0
label_LoadExternalEpilogueStructEnd_1:
.set sgprSrdScaleAlphaVec, 56
.set sgprSrdBias, 60

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */

/* shift vector components d0 */
v_mov_b32 v67, s[sgprWorkGroup0]
v_mul_i32_i24 v67, -0x80, v67                      // wg*MT
v_add_co_u32 v67, vcc, s[sgprSizesFree+0], v67     // wgMT = Size - wg*MT
v_mov_b32 v68, 0x80                                // MT
v_cmp_lt_u32 s[54:55], v67, v68                    // wgMT < MT
v_cndmask_b32 v67, v68, v67, s[54:55]              // wgMT = (wgMT < MT) ? wgMT : MT
v_lshrrev_b32 v69, 6, v[vgprSerial]                // v69 = v[vgprSerial] / 64
v_and_b32 v69, 1, v69                              // v69 = v[vgprSerial] / 64 % 2
v_lshrrev_b32 v70, 5, v67                          // v70 = v67 / 32
v_and_b32 v70, 1, v70                              // v70 = v70 % 2
v_cmp_eq_u32 s[54:55], v70, v69                    // wave_id == block_belong_to_wave?
v_cndmask_b32 v67, v68, v67, s[54:55]              // wgMT = (wgMT < MT) ? wgMT : MT

/* mbReg: which mb block need to shift, mb(matrixInstCoal(16) * VectorWidth(1)) */
//v_lshrrev_b32 v68, 4, v67                          // v68 = v67 / 16
//v_lshlrev_b32 v70, 0x2, v69                        // v70 = v[vgprSerial] / 64 % 2 * 4
//v_sub_u32 v68, v68, v70
v_and_b32 v68, 31,v67
v_lshrrev_b32 v68, 4, v68
v_lshrrev_b32 v70, 6, v67
v_lshlrev_b32 v70, 1, v70
v_add_u32 v68, v68, v70



/* gbReg: glvw block id */
//v_lshrrev_b32 v70, 2, v67                          // v70 = v67 / 4
//
///* tgbReg: glvw block id */
//v_lshrrev_b32 v71, 0, v[vgprSerial]                // v71 = v[vgprSerial] / 1
//v_and_b32 v71, 15, v71                             // v71 = v[vgprSerial] % 16
//                                                   // v71 = v71 * 1 (multiplier is 1, do nothing)
//v_lshrrev_b32 v71, 2, v71                          // v71 = v[vgprSerial] % 16 / 4
//v_lshlrev_b32 v69, 0x4, v69                        // v69 = v[vgprSerial] / 64 % 2 * 16
//v_add_co_u32 v71, vcc, v69, v71                    // tgbReg = (tid_coal * continOut) / GLVW
//v_sub_u32 v70, v70, v71

//v_and_b32 v70, 63, v67
//v_lshrrev_b32 v70, 3, v70
//
//v_lshrrev_b32 v71, 0, v[vgprSerial]
//v_and_b32 v72, 15, v71
//v_lshrrev_b32 v72, 3, v72
//
//v_lshrrev_b32 v71, 6, v71 
//v_and_b32 v71, 1, v71
//v_lshlrev_b32 v71, 2, v71
//
//v_and_b32 v69, 31, v67
//v_lshrrev_b32 v69, 3, v69
//v_lshlrev_b32 v69, 1, v69
//
//v_add_u32 v71, v71, v72
//v_add_u32 v71, v71, v69
//v_sub_u32 v70, v70, v71

v_and_b32 v69, 15, v67
v_lshrrev_b32 v69, 2, v69

v_and_b32 v70, 15, v[vgprSerial]
v_lshrrev_b32 v70, 2, v70

v_sub_u32 v70,v70,v69

/* vwReg: glvw in which vw block? */
v_and_b32 v69, 0, v67                              // permute register between threads
v_lshrrev_b32 v69, 2, v69                          // permute register between threads


/* rReg : reminder of M_size % GlobalReadVectorWidth */
v_and_b32 v71, 3, v67                              // v71 = v67 % 4
v_cmp_eq_u32 vcc, v71, 0x1                         // wgMT%VW == 1
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_0 // branch to shift d0 r=1
v_cmp_eq_u32 vcc, v71, 0x2                         // wgMT%VW == 2
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_0 // branch to shift d0 r=2
v_cmp_eq_u32 vcc, v71, 0x3                         // wgMT%VW == 3
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_0 // branch to shift d0 r=3
s_branch label_ShiftVectorComponents0_GLVW0_0      // no shifting

/******************************************/
/* shift d0 r=1                           */
/******************************************/
label_ShiftVectorComponents0_GLVW1_0:
v_cmp_eq_u32 vcc, v68, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM0_0 // branch to shift d0 r1 mb0
v_cmp_eq_u32 vcc, v68, 0x1
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM1_0 // branch to shift d0 r1 mb1
v_cmp_eq_u32 vcc, v68, 0x2
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM2_0 // branch to shift d0 r1 mb2
v_cmp_eq_u32 vcc, v68, 0x3
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM3_0 // branch to shift d0 r1 mb3

/******************************************/
/* shift d0 r=2                           */
/******************************************/
label_ShiftVectorComponents0_GLVW2_0:
v_cmp_eq_u32 vcc, v68, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM0_0 // branch to shift d0 r2 mb0
v_cmp_eq_u32 vcc, v68, 0x1
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM1_0 // branch to shift d0 r2 mb1
v_cmp_eq_u32 vcc, v68, 0x2
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM2_0 // branch to shift d0 r2 mb2
v_cmp_eq_u32 vcc, v68, 0x3
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM3_0 // branch to shift d0 r2 mb3

/******************************************/
/* shift d0 r=3                           */
/******************************************/
label_ShiftVectorComponents0_GLVW3_0:
v_cmp_eq_u32 vcc, v68, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM0_0 // branch to shift d0 r3 mb0
v_cmp_eq_u32 vcc, v68, 0x1
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM1_0 // branch to shift d0 r3 mb1
v_cmp_eq_u32 vcc, v68, 0x2
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM2_0 // branch to shift d0 r3 mb2
v_cmp_eq_u32 vcc, v68, 0x3
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM3_0 // branch to shift d0 r3 mb3

/******************************************/
/* shift d0 r=1 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM0_0:  /// r1 mb0
v_cmp_eq_u32 vcc, v69, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM0_VW0_0 // branch to shift d0 r1 mb0 vw0

/******************************************/
/* shift d0 r=1 mb=1                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM1_0:  /// r1 mb1
v_cmp_eq_u32 vcc, v69, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM1_VW0_0 // branch to shift d0 r1 mb1 vw0

/******************************************/
/* shift d0 r=1 mb=2                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM2_0:  /// r1 mb2
v_cmp_eq_u32 vcc, v69, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM2_VW0_0 // branch to shift d0 r1 mb2 vw0

/******************************************/
/* shift d0 r=1 mb=3                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM3_0:  /// r1 mb3
v_cmp_eq_u32 vcc, v69, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM3_VW0_0 // branch to shift d0 r1 mb3 vw0

/******************************************/
/* shift d0 r=2 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM0_0:  /// r2 mb0
v_cmp_eq_u32 vcc, v69, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM0_VW0_0 // branch to shift d0 r2 mb0 vw0

/******************************************/
/* shift d0 r=2 mb=1                      */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM1_0:  /// r2 mb1
v_cmp_eq_u32 vcc, v69, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM1_VW0_0 // branch to shift d0 r2 mb1 vw0

/******************************************/
/* shift d0 r=2 mb=2                      */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM2_0:  /// r2 mb2
v_cmp_eq_u32 vcc, v69, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM2_VW0_0 // branch to shift d0 r2 mb2 vw0

/******************************************/
/* shift d0 r=2 mb=3                      */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM3_0:  /// r2 mb3
v_cmp_eq_u32 vcc, v69, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM3_VW0_0 // branch to shift d0 r2 mb3 vw0

/******************************************/
/* shift d0 r=3 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM0_0:  /// r3 mb0
v_cmp_eq_u32 vcc, v69, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM0_VW0_0 // branch to shift d0 r3 mb0 vw0

/******************************************/
/* shift d0 r=3 mb=1                      */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM1_0:  /// r3 mb1
v_cmp_eq_u32 vcc, v69, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM1_VW0_0 // branch to shift d0 r3 mb1 vw0

/******************************************/
/* shift d0 r=3 mb=2                      */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM2_0:  /// r3 mb2
v_cmp_eq_u32 vcc, v69, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM2_VW0_0 // branch to shift d0 r3 mb2 vw0

/******************************************/
/* shift d0 r=3 mb=3                      */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM3_0:  /// r3 mb3
v_cmp_eq_u32 vcc, v69, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM3_VW0_0 // branch to shift d0 r3 mb3 vw0

/******************************************/
/* shift d0 r=1 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM0_VW0_0:  /// r1 mb0 vw0
s_mov_b32 s54, 0
v_cmpx_eq_u32 s[54:55], v70, s54                   // is thread in edge glvw region
v_and_b32 v64, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v64, 2, v64                          // permute register between threads
v_mov_b32 v71, v0                                  // glvw 1 mb 0 tt1 0 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v0, v71
v_mov_b32 v71, v1                                  // glvw 1 mb 0 tt1 1 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v1, v71
v_mov_b32 v71, v2                                  // glvw 1 mb 0 tt1 2 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v2, v71
v_mov_b32 v71, v3                                  // glvw 1 mb 0 tt1 3 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v3, v71
v_mov_b32 v71, v16                                 // glvw 1 mb 0 tt1 4 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v16, v71
v_mov_b32 v71, v17                                 // glvw 1 mb 0 tt1 5 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v17, v71
v_mov_b32 v71, v18                                 // glvw 1 mb 0 tt1 6 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v18, v71
v_mov_b32 v71, v19                                 // glvw 1 mb 0 tt1 7 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v19, v71
v_mov_b32 v71, v32                                 // glvw 1 mb 0 tt1 8 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v32, v71
v_mov_b32 v71, v33                                 // glvw 1 mb 0 tt1 9 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v33, v71
v_mov_b32 v71, v34                                 // glvw 1 mb 0 tt1 10 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v34, v71
v_mov_b32 v71, v35                                 // glvw 1 mb 0 tt1 11 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v35, v71
v_mov_b32 v71, v48                                 // glvw 1 mb 0 tt1 12 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v48, v71
v_mov_b32 v71, v49                                 // glvw 1 mb 0 tt1 13 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v49, v71
v_mov_b32 v71, v50                                 // glvw 1 mb 0 tt1 14 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v50, v71
v_mov_b32 v71, v51                                 // glvw 1 mb 0 tt1 15 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v51, v71
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=1 mb=1 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM1_VW0_0:  /// r1 mb1 vw0
s_mov_b32 s54, 0
v_cmpx_eq_u32 s[54:55], v70, s54                   // is thread in edge glvw region
v_and_b32 v64, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v64, 2, v64                          // permute register between threads
v_mov_b32 v71, v4                                  // glvw 1 mb 1 tt1 0 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v4, v71
v_mov_b32 v71, v5                                  // glvw 1 mb 1 tt1 1 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v5, v71
v_mov_b32 v71, v6                                  // glvw 1 mb 1 tt1 2 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v6, v71
v_mov_b32 v71, v7                                  // glvw 1 mb 1 tt1 3 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v7, v71
v_mov_b32 v71, v20                                 // glvw 1 mb 1 tt1 4 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v20, v71
v_mov_b32 v71, v21                                 // glvw 1 mb 1 tt1 5 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v21, v71
v_mov_b32 v71, v22                                 // glvw 1 mb 1 tt1 6 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v22, v71
v_mov_b32 v71, v23                                 // glvw 1 mb 1 tt1 7 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v23, v71
v_mov_b32 v71, v36                                 // glvw 1 mb 1 tt1 8 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v36, v71
v_mov_b32 v71, v37                                 // glvw 1 mb 1 tt1 9 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v37, v71
v_mov_b32 v71, v38                                 // glvw 1 mb 1 tt1 10 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v38, v71
v_mov_b32 v71, v39                                 // glvw 1 mb 1 tt1 11 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v39, v71
v_mov_b32 v71, v52                                 // glvw 1 mb 1 tt1 12 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v52, v71
v_mov_b32 v71, v53                                 // glvw 1 mb 1 tt1 13 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v53, v71
v_mov_b32 v71, v54                                 // glvw 1 mb 1 tt1 14 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v54, v71
v_mov_b32 v71, v55                                 // glvw 1 mb 1 tt1 15 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v55, v71
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=1 mb=2 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM2_VW0_0:  /// r1 mb2 vw0
s_mov_b32 s54, 0
v_cmpx_eq_u32 s[54:55], v70, s54                   // is thread in edge glvw region
v_and_b32 v64, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v64, 2, v64                          // permute register between threads
v_mov_b32 v71, v8                                  // glvw 1 mb 2 tt1 0 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v8, v71
v_mov_b32 v71, v9                                  // glvw 1 mb 2 tt1 1 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v9, v71
v_mov_b32 v71, v10                                 // glvw 1 mb 2 tt1 2 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v10, v71
v_mov_b32 v71, v11                                 // glvw 1 mb 2 tt1 3 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v11, v71
v_mov_b32 v71, v24                                 // glvw 1 mb 2 tt1 4 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v24, v71
v_mov_b32 v71, v25                                 // glvw 1 mb 2 tt1 5 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v25, v71
v_mov_b32 v71, v26                                 // glvw 1 mb 2 tt1 6 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v26, v71
v_mov_b32 v71, v27                                 // glvw 1 mb 2 tt1 7 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v27, v71
v_mov_b32 v71, v40                                 // glvw 1 mb 2 tt1 8 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v40, v71
v_mov_b32 v71, v41                                 // glvw 1 mb 2 tt1 9 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v41, v71
v_mov_b32 v71, v42                                 // glvw 1 mb 2 tt1 10 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v42, v71
v_mov_b32 v71, v43                                 // glvw 1 mb 2 tt1 11 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v43, v71
v_mov_b32 v71, v56                                 // glvw 1 mb 2 tt1 12 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v56, v71
v_mov_b32 v71, v57                                 // glvw 1 mb 2 tt1 13 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v57, v71
v_mov_b32 v71, v58                                 // glvw 1 mb 2 tt1 14 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v58, v71
v_mov_b32 v71, v59                                 // glvw 1 mb 2 tt1 15 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v59, v71
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=1 mb=3 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM3_VW0_0:  /// r1 mb3 vw0
s_mov_b32 s54, 0
v_cmpx_eq_u32 s[54:55], v70, s54                   // is thread in edge glvw region
v_and_b32 v64, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v64, 2, v64                          // permute register between threads
v_mov_b32 v71, v12                                 // glvw 1 mb 3 tt1 0 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v12, v71
v_mov_b32 v71, v13                                 // glvw 1 mb 3 tt1 1 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v13, v71
v_mov_b32 v71, v14                                 // glvw 1 mb 3 tt1 2 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v14, v71
v_mov_b32 v71, v15                                 // glvw 1 mb 3 tt1 3 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v15, v71
v_mov_b32 v71, v28                                 // glvw 1 mb 3 tt1 4 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v28, v71
v_mov_b32 v71, v29                                 // glvw 1 mb 3 tt1 5 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v29, v71
v_mov_b32 v71, v30                                 // glvw 1 mb 3 tt1 6 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v30, v71
v_mov_b32 v71, v31                                 // glvw 1 mb 3 tt1 7 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v31, v71
v_mov_b32 v71, v44                                 // glvw 1 mb 3 tt1 8 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v44, v71
v_mov_b32 v71, v45                                 // glvw 1 mb 3 tt1 9 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v45, v71
v_mov_b32 v71, v46                                 // glvw 1 mb 3 tt1 10 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v46, v71
v_mov_b32 v71, v47                                 // glvw 1 mb 3 tt1 11 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v47, v71
v_mov_b32 v71, v60                                 // glvw 1 mb 3 tt1 12 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v60, v71
v_mov_b32 v71, v61                                 // glvw 1 mb 3 tt1 13 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v61, v71
v_mov_b32 v71, v62                                 // glvw 1 mb 3 tt1 14 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v62, v71
v_mov_b32 v71, v63                                 // glvw 1 mb 3 tt1 15 r 0
ds_bpermute_b32 v71, v64, v71 offset:12            // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v63, v71
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=2 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM0_VW0_0:  /// r2 mb0 vw0
s_mov_b32 s54, 0
v_cmpx_eq_u32 s[54:55], v70, s54                   // is thread in edge glvw region
v_and_b32 v64, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v64, 2, v64                          // permute register between threads
v_mov_b32 v71, v0                                  // glvw 2 mb 0 tt1 0 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v0, v71
v_mov_b32 v71, v1                                  // glvw 2 mb 0 tt1 1 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v1, v71
v_mov_b32 v71, v2                                  // glvw 2 mb 0 tt1 2 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v2, v71
v_mov_b32 v71, v3                                  // glvw 2 mb 0 tt1 3 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v3, v71
v_mov_b32 v71, v16                                 // glvw 2 mb 0 tt1 4 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v16, v71
v_mov_b32 v71, v17                                 // glvw 2 mb 0 tt1 5 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v17, v71
v_mov_b32 v71, v18                                 // glvw 2 mb 0 tt1 6 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v18, v71
v_mov_b32 v71, v19                                 // glvw 2 mb 0 tt1 7 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v19, v71
v_mov_b32 v71, v32                                 // glvw 2 mb 0 tt1 8 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v32, v71
v_mov_b32 v71, v33                                 // glvw 2 mb 0 tt1 9 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v33, v71
v_mov_b32 v71, v34                                 // glvw 2 mb 0 tt1 10 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v34, v71
v_mov_b32 v71, v35                                 // glvw 2 mb 0 tt1 11 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v35, v71
v_mov_b32 v71, v48                                 // glvw 2 mb 0 tt1 12 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v48, v71
v_mov_b32 v71, v49                                 // glvw 2 mb 0 tt1 13 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v49, v71
v_mov_b32 v71, v50                                 // glvw 2 mb 0 tt1 14 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v50, v71
v_mov_b32 v71, v51                                 // glvw 2 mb 0 tt1 15 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v51, v71
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=2 mb=1 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM1_VW0_0:  /// r2 mb1 vw0
s_mov_b32 s54, 0
v_cmpx_eq_u32 s[54:55], v70, s54                   // is thread in edge glvw region
v_and_b32 v64, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v64, 2, v64                          // permute register between threads
v_mov_b32 v71, v4                                  // glvw 2 mb 1 tt1 0 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v4, v71
v_mov_b32 v71, v5                                  // glvw 2 mb 1 tt1 1 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v5, v71
v_mov_b32 v71, v6                                  // glvw 2 mb 1 tt1 2 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v6, v71
v_mov_b32 v71, v7                                  // glvw 2 mb 1 tt1 3 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v7, v71
v_mov_b32 v71, v20                                 // glvw 2 mb 1 tt1 4 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v20, v71
v_mov_b32 v71, v21                                 // glvw 2 mb 1 tt1 5 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v21, v71
v_mov_b32 v71, v22                                 // glvw 2 mb 1 tt1 6 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v22, v71
v_mov_b32 v71, v23                                 // glvw 2 mb 1 tt1 7 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v23, v71
v_mov_b32 v71, v36                                 // glvw 2 mb 1 tt1 8 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v36, v71
v_mov_b32 v71, v37                                 // glvw 2 mb 1 tt1 9 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v37, v71
v_mov_b32 v71, v38                                 // glvw 2 mb 1 tt1 10 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v38, v71
v_mov_b32 v71, v39                                 // glvw 2 mb 1 tt1 11 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v39, v71
v_mov_b32 v71, v52                                 // glvw 2 mb 1 tt1 12 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v52, v71
v_mov_b32 v71, v53                                 // glvw 2 mb 1 tt1 13 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v53, v71
v_mov_b32 v71, v54                                 // glvw 2 mb 1 tt1 14 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v54, v71
v_mov_b32 v71, v55                                 // glvw 2 mb 1 tt1 15 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v55, v71
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=2 mb=2 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM2_VW0_0:  /// r2 mb2 vw0
s_mov_b32 s54, 0
v_cmpx_eq_u32 s[54:55], v70, s54                   // is thread in edge glvw region
v_and_b32 v64, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v64, 2, v64                          // permute register between threads
v_mov_b32 v71, v8                                  // glvw 2 mb 2 tt1 0 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v8, v71
v_mov_b32 v71, v9                                  // glvw 2 mb 2 tt1 1 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v9, v71
v_mov_b32 v71, v10                                 // glvw 2 mb 2 tt1 2 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v10, v71
v_mov_b32 v71, v11                                 // glvw 2 mb 2 tt1 3 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v11, v71
v_mov_b32 v71, v24                                 // glvw 2 mb 2 tt1 4 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v24, v71
v_mov_b32 v71, v25                                 // glvw 2 mb 2 tt1 5 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v25, v71
v_mov_b32 v71, v26                                 // glvw 2 mb 2 tt1 6 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v26, v71
v_mov_b32 v71, v27                                 // glvw 2 mb 2 tt1 7 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v27, v71
v_mov_b32 v71, v40                                 // glvw 2 mb 2 tt1 8 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v40, v71
v_mov_b32 v71, v41                                 // glvw 2 mb 2 tt1 9 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v41, v71
v_mov_b32 v71, v42                                 // glvw 2 mb 2 tt1 10 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v42, v71
v_mov_b32 v71, v43                                 // glvw 2 mb 2 tt1 11 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v43, v71
v_mov_b32 v71, v56                                 // glvw 2 mb 2 tt1 12 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v56, v71
v_mov_b32 v71, v57                                 // glvw 2 mb 2 tt1 13 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v57, v71
v_mov_b32 v71, v58                                 // glvw 2 mb 2 tt1 14 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v58, v71
v_mov_b32 v71, v59                                 // glvw 2 mb 2 tt1 15 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v59, v71
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=2 mb=3 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM3_VW0_0:  /// r2 mb3 vw0
s_mov_b32 s54, 0
v_cmpx_eq_u32 s[54:55], v70, s54                   // is thread in edge glvw region
v_and_b32 v64, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v64, 2, v64                          // permute register between threads
v_mov_b32 v71, v12                                 // glvw 2 mb 3 tt1 0 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v12, v71
v_mov_b32 v71, v13                                 // glvw 2 mb 3 tt1 1 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v13, v71
v_mov_b32 v71, v14                                 // glvw 2 mb 3 tt1 2 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v14, v71
v_mov_b32 v71, v15                                 // glvw 2 mb 3 tt1 3 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v15, v71
v_mov_b32 v71, v28                                 // glvw 2 mb 3 tt1 4 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v28, v71
v_mov_b32 v71, v29                                 // glvw 2 mb 3 tt1 5 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v29, v71
v_mov_b32 v71, v30                                 // glvw 2 mb 3 tt1 6 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v30, v71
v_mov_b32 v71, v31                                 // glvw 2 mb 3 tt1 7 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v31, v71
v_mov_b32 v71, v44                                 // glvw 2 mb 3 tt1 8 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v44, v71
v_mov_b32 v71, v45                                 // glvw 2 mb 3 tt1 9 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v45, v71
v_mov_b32 v71, v46                                 // glvw 2 mb 3 tt1 10 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v46, v71
v_mov_b32 v71, v47                                 // glvw 2 mb 3 tt1 11 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v47, v71
v_mov_b32 v71, v60                                 // glvw 2 mb 3 tt1 12 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v60, v71
v_mov_b32 v71, v61                                 // glvw 2 mb 3 tt1 13 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v61, v71
v_mov_b32 v71, v62                                 // glvw 2 mb 3 tt1 14 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v62, v71
v_mov_b32 v71, v63                                 // glvw 2 mb 3 tt1 15 r 0
ds_bpermute_b32 v71, v64, v71 offset:8             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v63, v71
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=3 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM0_VW0_0:  /// r3 mb0 vw0
s_mov_b32 s54, 0
v_cmpx_eq_u32 s[54:55], v70, s54                   // is thread in edge glvw region
v_and_b32 v64, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v64, 2, v64                          // permute register between threads
v_mov_b32 v71, v0                                  // glvw 3 mb 0 tt1 0 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v0, v71
v_mov_b32 v71, v1                                  // glvw 3 mb 0 tt1 1 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v1, v71
v_mov_b32 v71, v2                                  // glvw 3 mb 0 tt1 2 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v2, v71
v_mov_b32 v71, v3                                  // glvw 3 mb 0 tt1 3 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v3, v71
v_mov_b32 v71, v16                                 // glvw 3 mb 0 tt1 4 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v16, v71
v_mov_b32 v71, v17                                 // glvw 3 mb 0 tt1 5 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v17, v71
v_mov_b32 v71, v18                                 // glvw 3 mb 0 tt1 6 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v18, v71
v_mov_b32 v71, v19                                 // glvw 3 mb 0 tt1 7 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v19, v71
v_mov_b32 v71, v32                                 // glvw 3 mb 0 tt1 8 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v32, v71
v_mov_b32 v71, v33                                 // glvw 3 mb 0 tt1 9 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v33, v71
v_mov_b32 v71, v34                                 // glvw 3 mb 0 tt1 10 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v34, v71
v_mov_b32 v71, v35                                 // glvw 3 mb 0 tt1 11 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v35, v71
v_mov_b32 v71, v48                                 // glvw 3 mb 0 tt1 12 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v48, v71
v_mov_b32 v71, v49                                 // glvw 3 mb 0 tt1 13 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v49, v71
v_mov_b32 v71, v50                                 // glvw 3 mb 0 tt1 14 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v50, v71
v_mov_b32 v71, v51                                 // glvw 3 mb 0 tt1 15 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v51, v71
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=3 mb=1 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM1_VW0_0:  /// r3 mb1 vw0
s_mov_b32 s54, 0
v_cmpx_eq_u32 s[54:55], v70, s54                   // is thread in edge glvw region
v_and_b32 v64, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v64, 2, v64                          // permute register between threads
v_mov_b32 v71, v4                                  // glvw 3 mb 1 tt1 0 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v4, v71
v_mov_b32 v71, v5                                  // glvw 3 mb 1 tt1 1 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v5, v71
v_mov_b32 v71, v6                                  // glvw 3 mb 1 tt1 2 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v6, v71
v_mov_b32 v71, v7                                  // glvw 3 mb 1 tt1 3 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v7, v71
v_mov_b32 v71, v20                                 // glvw 3 mb 1 tt1 4 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v20, v71
v_mov_b32 v71, v21                                 // glvw 3 mb 1 tt1 5 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v21, v71
v_mov_b32 v71, v22                                 // glvw 3 mb 1 tt1 6 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v22, v71
v_mov_b32 v71, v23                                 // glvw 3 mb 1 tt1 7 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v23, v71
v_mov_b32 v71, v36                                 // glvw 3 mb 1 tt1 8 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v36, v71
v_mov_b32 v71, v37                                 // glvw 3 mb 1 tt1 9 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v37, v71
v_mov_b32 v71, v38                                 // glvw 3 mb 1 tt1 10 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v38, v71
v_mov_b32 v71, v39                                 // glvw 3 mb 1 tt1 11 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v39, v71
v_mov_b32 v71, v52                                 // glvw 3 mb 1 tt1 12 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v52, v71
v_mov_b32 v71, v53                                 // glvw 3 mb 1 tt1 13 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v53, v71
v_mov_b32 v71, v54                                 // glvw 3 mb 1 tt1 14 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v54, v71
v_mov_b32 v71, v55                                 // glvw 3 mb 1 tt1 15 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v55, v71
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=3 mb=2 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM2_VW0_0:  /// r3 mb2 vw0
s_mov_b32 s54, 0
v_cmpx_eq_u32 s[54:55], v70, s54                   // is thread in edge glvw region
v_and_b32 v64, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v64, 2, v64                          // permute register between threads
v_mov_b32 v71, v8                                  // glvw 3 mb 2 tt1 0 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v8, v71
v_mov_b32 v71, v9                                  // glvw 3 mb 2 tt1 1 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v9, v71
v_mov_b32 v71, v10                                 // glvw 3 mb 2 tt1 2 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v10, v71
v_mov_b32 v71, v11                                 // glvw 3 mb 2 tt1 3 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v11, v71
v_mov_b32 v71, v24                                 // glvw 3 mb 2 tt1 4 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v24, v71
v_mov_b32 v71, v25                                 // glvw 3 mb 2 tt1 5 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v25, v71
v_mov_b32 v71, v26                                 // glvw 3 mb 2 tt1 6 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v26, v71
v_mov_b32 v71, v27                                 // glvw 3 mb 2 tt1 7 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v27, v71
v_mov_b32 v71, v40                                 // glvw 3 mb 2 tt1 8 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v40, v71
v_mov_b32 v71, v41                                 // glvw 3 mb 2 tt1 9 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v41, v71
v_mov_b32 v71, v42                                 // glvw 3 mb 2 tt1 10 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v42, v71
v_mov_b32 v71, v43                                 // glvw 3 mb 2 tt1 11 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v43, v71
v_mov_b32 v71, v56                                 // glvw 3 mb 2 tt1 12 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v56, v71
v_mov_b32 v71, v57                                 // glvw 3 mb 2 tt1 13 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v57, v71
v_mov_b32 v71, v58                                 // glvw 3 mb 2 tt1 14 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v58, v71
v_mov_b32 v71, v59                                 // glvw 3 mb 2 tt1 15 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v59, v71
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=3 mb=3 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM3_VW0_0:  /// r3 mb3 vw0
s_mov_b32 s54, 0
v_cmpx_eq_u32 s[54:55], v70, s54                   // is thread in edge glvw region
v_and_b32 v64, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v64, 2, v64                          // permute register between threads
v_mov_b32 v71, v12                                 // glvw 3 mb 3 tt1 0 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v12, v71
v_mov_b32 v71, v13                                 // glvw 3 mb 3 tt1 1 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v13, v71
v_mov_b32 v71, v14                                 // glvw 3 mb 3 tt1 2 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v14, v71
v_mov_b32 v71, v15                                 // glvw 3 mb 3 tt1 3 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v15, v71
v_mov_b32 v71, v28                                 // glvw 3 mb 3 tt1 4 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v28, v71
v_mov_b32 v71, v29                                 // glvw 3 mb 3 tt1 5 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v29, v71
v_mov_b32 v71, v30                                 // glvw 3 mb 3 tt1 6 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v30, v71
v_mov_b32 v71, v31                                 // glvw 3 mb 3 tt1 7 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v31, v71
v_mov_b32 v71, v44                                 // glvw 3 mb 3 tt1 8 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v44, v71
v_mov_b32 v71, v45                                 // glvw 3 mb 3 tt1 9 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v45, v71
v_mov_b32 v71, v46                                 // glvw 3 mb 3 tt1 10 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v46, v71
v_mov_b32 v71, v47                                 // glvw 3 mb 3 tt1 11 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v47, v71
v_mov_b32 v71, v60                                 // glvw 3 mb 3 tt1 12 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v60, v71
v_mov_b32 v71, v61                                 // glvw 3 mb 3 tt1 13 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v61, v71
v_mov_b32 v71, v62                                 // glvw 3 mb 3 tt1 14 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v62, v71
v_mov_b32 v71, v63                                 // glvw 3 mb 3 tt1 15 r 0
ds_bpermute_b32 v71, v64, v71 offset:4             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_mov_b32 v63, v71
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting

label_ShiftVectorComponents0_GLVW0_0:  /// end shift0

/* shift vector components d1 */

/* check which macro tile need to shift */
v_mov_b32 v67, s[sgprWorkGroup1]
v_mul_i32_i24 v67, -0x100, v67                     // wg*MT
v_add_co_u32 v67, vcc, s[sgprSizesFree+1], v67     // wgMT = Size - wg*MT
v_mov_b32 v68, 0x100                               // MT
v_cmp_lt_u32 s[54:55], v67, v68                    // wgMT < MT
v_cndmask_b32 v67, v68, v67, s[54:55]              // wgMT = (wgMT < MT) ? wgMT : MT


/* check which wave need to shift */
v_lshrrev_b32 v69, 7, v[vgprSerial]                // v69 = v[vgprSerial] / 128
v_and_b32 v69, 3, v69                              // v69 = v69 % 4
v_and_b32 v70, 127, v67
v_lshrrev_b32 v70, 5, v70                         // v70 = v67 %128 / 32
v_and_b32 v70, 3, v70                              // v70 = v70 % 4
v_cmp_eq_u32 s[54:55], v70, v69                    // wave_id == block_belong_to_wave?
v_cndmask_b32 v67, v68, v67, s[54:55]              // wgMT = (wgMT < MT) ? wgMT : MT


/* get id of which glvw block need to shift */
//v_mul_i32_i24 v71, -0x40, v69                      // wg * MIB  v[vgprSerial] / 128 % 4 x 64
//v_add_co_u32 v71, vcc, v71, v67                    // wgMT = Size - wg*MIB  v67 - v[vgprSerial] / 128 % 4 x 64
//v_lshrrev_b32 v71, 3, v71                          // glvw block id  (v67 - v[vgprSerial] / 128 % 4 x 64) / 8

v_lshrrev_b32 v69, 7, v67                // v69 = v67 / 128
v_lshlrev_b32 v69, 2, v69
v_and_b32 v71, 31, v67
v_lshrrev_b32 v71, 3, v71
v_add_u32 v71, v71, v69




/* dispatch to differet shift block for shift */
v_and_b32 v72, 7, v67                              // v72 = v67 % 8
v_cmp_eq_u32 vcc, v72, 0x1                         // wgMT%VW == 1
s_cbranch_vccnz label_ShiftVectorComponents1_shift1_0 // branch to shift d1 r=1
v_cmp_eq_u32 vcc, v72, 0x2                         // wgMT%VW == 2
s_cbranch_vccnz label_ShiftVectorComponents1_shift2_0 // branch to shift d1 r=2
v_cmp_eq_u32 vcc, v72, 0x3                         // wgMT%VW == 3
s_cbranch_vccnz label_ShiftVectorComponents1_shift3_0 // branch to shift d1 r=3
v_cmp_eq_u32 vcc, v72, 0x4                         // wgMT%VW == 4
s_cbranch_vccnz label_ShiftVectorComponents1_shift4_0 // branch to shift d1 r=4
v_cmp_eq_u32 vcc, v72, 0x5                         // wgMT%VW == 5
s_cbranch_vccnz label_ShiftVectorComponents1_shift5_0 // branch to shift d1 r=5
v_cmp_eq_u32 vcc, v72, 0x6                         // wgMT%VW == 6
s_cbranch_vccnz label_ShiftVectorComponents1_shift6_0 // branch to shift d1 r=6
v_cmp_eq_u32 vcc, v72, 0x7                         // wgMT%VW == 7
s_cbranch_vccnz label_ShiftVectorComponents1_shift7_0 // branch to shift d1 r=7
s_branch label_ShiftVectorComponents1_shift0_0     // no shifting

/******************************************/
/* shift d1 shift=1                       */
/******************************************/
label_ShiftVectorComponents1_shift1_0:
v_cmp_eq_u32 vcc, v71, 0x0
s_cbranch_vccnz label_ShiftVectorComponents1_shift1_glvwblk0_0 // branch to shift d1 shift1 glvwblk0
v_cmp_eq_u32 vcc, v71, 0x1
s_cbranch_vccnz label_ShiftVectorComponents1_shift1_glvwblk1_0 // branch to shift d1 shift1 glvwblk1
v_cmp_eq_u32 vcc, v71, 0x2
s_cbranch_vccnz label_ShiftVectorComponents1_shift1_glvwblk2_0 // branch to shift d1 shift1 glvwblk2
v_cmp_eq_u32 vcc, v71, 0x3
s_cbranch_vccnz label_ShiftVectorComponents1_shift1_glvwblk3_0 // branch to shift d1 shift1 glvwblk3
v_cmp_eq_u32 vcc, v71, 0x4
s_cbranch_vccnz label_ShiftVectorComponents1_shift1_glvwblk4_0 // branch to shift d1 shift1 glvwblk4
v_cmp_eq_u32 vcc, v71, 0x5
s_cbranch_vccnz label_ShiftVectorComponents1_shift1_glvwblk5_0 // branch to shift d1 shift1 glvwblk5
v_cmp_eq_u32 vcc, v71, 0x6
s_cbranch_vccnz label_ShiftVectorComponents1_shift1_glvwblk6_0 // branch to shift d1 shift1 glvwblk6
v_cmp_eq_u32 vcc, v71, 0x7
s_cbranch_vccnz label_ShiftVectorComponents1_shift1_glvwblk7_0 // branch to shift d1 shift1 glvwblk7

/******************************************/
/* shift d1 shift=2                       */
/******************************************/
label_ShiftVectorComponents1_shift2_0:
v_cmp_eq_u32 vcc, v71, 0x0
s_cbranch_vccnz label_ShiftVectorComponents1_shift2_glvwblk0_0 // branch to shift d1 shift2 glvwblk0
v_cmp_eq_u32 vcc, v71, 0x1
s_cbranch_vccnz label_ShiftVectorComponents1_shift2_glvwblk1_0 // branch to shift d1 shift2 glvwblk1
v_cmp_eq_u32 vcc, v71, 0x2
s_cbranch_vccnz label_ShiftVectorComponents1_shift2_glvwblk2_0 // branch to shift d1 shift2 glvwblk2
v_cmp_eq_u32 vcc, v71, 0x3
s_cbranch_vccnz label_ShiftVectorComponents1_shift2_glvwblk3_0 // branch to shift d1 shift2 glvwblk3
v_cmp_eq_u32 vcc, v71, 0x4
s_cbranch_vccnz label_ShiftVectorComponents1_shift2_glvwblk4_0 // branch to shift d1 shift2 glvwblk4
v_cmp_eq_u32 vcc, v71, 0x5
s_cbranch_vccnz label_ShiftVectorComponents1_shift2_glvwblk5_0 // branch to shift d1 shift2 glvwblk5
v_cmp_eq_u32 vcc, v71, 0x6
s_cbranch_vccnz label_ShiftVectorComponents1_shift2_glvwblk6_0 // branch to shift d1 shift2 glvwblk6
v_cmp_eq_u32 vcc, v71, 0x7
s_cbranch_vccnz label_ShiftVectorComponents1_shift2_glvwblk7_0 // branch to shift d1 shift2 glvwblk7

/******************************************/
/* shift d1 shift=3                       */
/******************************************/
label_ShiftVectorComponents1_shift3_0:
v_cmp_eq_u32 vcc, v71, 0x0
s_cbranch_vccnz label_ShiftVectorComponents1_shift3_glvwblk0_0 // branch to shift d1 shift3 glvwblk0
v_cmp_eq_u32 vcc, v71, 0x1
s_cbranch_vccnz label_ShiftVectorComponents1_shift3_glvwblk1_0 // branch to shift d1 shift3 glvwblk1
v_cmp_eq_u32 vcc, v71, 0x2
s_cbranch_vccnz label_ShiftVectorComponents1_shift3_glvwblk2_0 // branch to shift d1 shift3 glvwblk2
v_cmp_eq_u32 vcc, v71, 0x3
s_cbranch_vccnz label_ShiftVectorComponents1_shift3_glvwblk3_0 // branch to shift d1 shift3 glvwblk3
v_cmp_eq_u32 vcc, v71, 0x4
s_cbranch_vccnz label_ShiftVectorComponents1_shift3_glvwblk4_0 // branch to shift d1 shift3 glvwblk4
v_cmp_eq_u32 vcc, v71, 0x5
s_cbranch_vccnz label_ShiftVectorComponents1_shift3_glvwblk5_0 // branch to shift d1 shift3 glvwblk5
v_cmp_eq_u32 vcc, v71, 0x6
s_cbranch_vccnz label_ShiftVectorComponents1_shift3_glvwblk6_0 // branch to shift d1 shift3 glvwblk6
v_cmp_eq_u32 vcc, v71, 0x7
s_cbranch_vccnz label_ShiftVectorComponents1_shift3_glvwblk7_0 // branch to shift d1 shift3 glvwblk7

/******************************************/
/* shift d1 shift=4                       */
/******************************************/
label_ShiftVectorComponents1_shift4_0:
v_cmp_eq_u32 vcc, v71, 0x0
s_cbranch_vccnz label_ShiftVectorComponents1_shift4_glvwblk0_0 // branch to shift d1 shift4 glvwblk0
v_cmp_eq_u32 vcc, v71, 0x1
s_cbranch_vccnz label_ShiftVectorComponents1_shift4_glvwblk1_0 // branch to shift d1 shift4 glvwblk1
v_cmp_eq_u32 vcc, v71, 0x2
s_cbranch_vccnz label_ShiftVectorComponents1_shift4_glvwblk2_0 // branch to shift d1 shift4 glvwblk2
v_cmp_eq_u32 vcc, v71, 0x3
s_cbranch_vccnz label_ShiftVectorComponents1_shift4_glvwblk3_0 // branch to shift d1 shift4 glvwblk3
v_cmp_eq_u32 vcc, v71, 0x4
s_cbranch_vccnz label_ShiftVectorComponents1_shift4_glvwblk4_0 // branch to shift d1 shift4 glvwblk4
v_cmp_eq_u32 vcc, v71, 0x5
s_cbranch_vccnz label_ShiftVectorComponents1_shift4_glvwblk5_0 // branch to shift d1 shift4 glvwblk5
v_cmp_eq_u32 vcc, v71, 0x6
s_cbranch_vccnz label_ShiftVectorComponents1_shift4_glvwblk6_0 // branch to shift d1 shift4 glvwblk6
v_cmp_eq_u32 vcc, v71, 0x7
s_cbranch_vccnz label_ShiftVectorComponents1_shift4_glvwblk7_0 // branch to shift d1 shift4 glvwblk7

/******************************************/
/* shift d1 shift=5                       */
/******************************************/
label_ShiftVectorComponents1_shift5_0:
v_cmp_eq_u32 vcc, v71, 0x0
s_cbranch_vccnz label_ShiftVectorComponents1_shift5_glvwblk0_0 // branch to shift d1 shift5 glvwblk0
v_cmp_eq_u32 vcc, v71, 0x1
s_cbranch_vccnz label_ShiftVectorComponents1_shift5_glvwblk1_0 // branch to shift d1 shift5 glvwblk1
v_cmp_eq_u32 vcc, v71, 0x2
s_cbranch_vccnz label_ShiftVectorComponents1_shift5_glvwblk2_0 // branch to shift d1 shift5 glvwblk2
v_cmp_eq_u32 vcc, v71, 0x3
s_cbranch_vccnz label_ShiftVectorComponents1_shift5_glvwblk3_0 // branch to shift d1 shift5 glvwblk3
v_cmp_eq_u32 vcc, v71, 0x4
s_cbranch_vccnz label_ShiftVectorComponents1_shift5_glvwblk4_0 // branch to shift d1 shift5 glvwblk4
v_cmp_eq_u32 vcc, v71, 0x5
s_cbranch_vccnz label_ShiftVectorComponents1_shift5_glvwblk5_0 // branch to shift d1 shift5 glvwblk5
v_cmp_eq_u32 vcc, v71, 0x6
s_cbranch_vccnz label_ShiftVectorComponents1_shift5_glvwblk6_0 // branch to shift d1 shift5 glvwblk6
v_cmp_eq_u32 vcc, v71, 0x7
s_cbranch_vccnz label_ShiftVectorComponents1_shift5_glvwblk7_0 // branch to shift d1 shift5 glvwblk7

/******************************************/
/* shift d1 shift=6                       */
/******************************************/
label_ShiftVectorComponents1_shift6_0:
v_cmp_eq_u32 vcc, v71, 0x0
s_cbranch_vccnz label_ShiftVectorComponents1_shift6_glvwblk0_0 // branch to shift d1 shift6 glvwblk0
v_cmp_eq_u32 vcc, v71, 0x1
s_cbranch_vccnz label_ShiftVectorComponents1_shift6_glvwblk1_0 // branch to shift d1 shift6 glvwblk1
v_cmp_eq_u32 vcc, v71, 0x2
s_cbranch_vccnz label_ShiftVectorComponents1_shift6_glvwblk2_0 // branch to shift d1 shift6 glvwblk2
v_cmp_eq_u32 vcc, v71, 0x3
s_cbranch_vccnz label_ShiftVectorComponents1_shift6_glvwblk3_0 // branch to shift d1 shift6 glvwblk3
v_cmp_eq_u32 vcc, v71, 0x4
s_cbranch_vccnz label_ShiftVectorComponents1_shift6_glvwblk4_0 // branch to shift d1 shift6 glvwblk4
v_cmp_eq_u32 vcc, v71, 0x5
s_cbranch_vccnz label_ShiftVectorComponents1_shift6_glvwblk5_0 // branch to shift d1 shift6 glvwblk5
v_cmp_eq_u32 vcc, v71, 0x6
s_cbranch_vccnz label_ShiftVectorComponents1_shift6_glvwblk6_0 // branch to shift d1 shift6 glvwblk6
v_cmp_eq_u32 vcc, v71, 0x7
s_cbranch_vccnz label_ShiftVectorComponents1_shift6_glvwblk7_0 // branch to shift d1 shift6 glvwblk7

/******************************************/
/* shift d1 shift=7                       */
/******************************************/
label_ShiftVectorComponents1_shift7_0:
v_cmp_eq_u32 vcc, v71, 0x0
s_cbranch_vccnz label_ShiftVectorComponents1_shift7_glvwblk0_0 // branch to shift d1 shift7 glvwblk0
v_cmp_eq_u32 vcc, v71, 0x1
s_cbranch_vccnz label_ShiftVectorComponents1_shift7_glvwblk1_0 // branch to shift d1 shift7 glvwblk1
v_cmp_eq_u32 vcc, v71, 0x2
s_cbranch_vccnz label_ShiftVectorComponents1_shift7_glvwblk2_0 // branch to shift d1 shift7 glvwblk2
v_cmp_eq_u32 vcc, v71, 0x3
s_cbranch_vccnz label_ShiftVectorComponents1_shift7_glvwblk3_0 // branch to shift d1 shift7 glvwblk3
v_cmp_eq_u32 vcc, v71, 0x4
s_cbranch_vccnz label_ShiftVectorComponents1_shift7_glvwblk4_0 // branch to shift d1 shift7 glvwblk4
v_cmp_eq_u32 vcc, v71, 0x5
s_cbranch_vccnz label_ShiftVectorComponents1_shift7_glvwblk5_0 // branch to shift d1 shift7 glvwblk5
v_cmp_eq_u32 vcc, v71, 0x6
s_cbranch_vccnz label_ShiftVectorComponents1_shift7_glvwblk6_0 // branch to shift d1 shift7 glvwblk6
v_cmp_eq_u32 vcc, v71, 0x7
s_cbranch_vccnz label_ShiftVectorComponents1_shift7_glvwblk7_0 // branch to shift d1 shift7 glvwblk7

/******************************************/
/* Reg 74-89                              */
/******************************************/

/******************************************/
/* shift d1 shift=1 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift1_glvwblk0_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=1 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift1_glvwblk1_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=1 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift1_glvwblk2_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=1 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift1_glvwblk3_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=1 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift1_glvwblk4_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=1 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift1_glvwblk5_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=1 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift1_glvwblk6_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=1 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift1_glvwblk7_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=2 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift2_glvwblk0_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=2 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift2_glvwblk1_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=2 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift2_glvwblk2_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=2 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift2_glvwblk3_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=2 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift2_glvwblk4_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=2 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift2_glvwblk5_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=2 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift2_glvwblk6_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=2 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift2_glvwblk7_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=3 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift3_glvwblk0_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=3 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift3_glvwblk1_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=3 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift3_glvwblk2_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=3 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift3_glvwblk3_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=3 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift3_glvwblk4_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=3 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift3_glvwblk5_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=3 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift3_glvwblk6_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=3 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift3_glvwblk7_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=4 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift4_glvwblk0_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=4 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift4_glvwblk1_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=4 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift4_glvwblk2_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=4 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift4_glvwblk3_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=4 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift4_glvwblk4_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=4 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift4_glvwblk5_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=4 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift4_glvwblk6_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=4 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift4_glvwblk7_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=5 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift5_glvwblk0_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v0
v_mov_b32 v75, v4
v_mov_b32 v76, v8
v_mov_b32 v77, v12
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v1, v74
v_mov_b32 v5, v75
v_mov_b32 v9, v76
v_mov_b32 v13, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=5 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift5_glvwblk1_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v2
v_mov_b32 v75, v6
v_mov_b32 v76, v10
v_mov_b32 v77, v14
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v3, v74
v_mov_b32 v7, v75
v_mov_b32 v11, v76
v_mov_b32 v15, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=5 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift5_glvwblk2_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v16
v_mov_b32 v75, v20
v_mov_b32 v76, v24
v_mov_b32 v77, v28
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v17, v74
v_mov_b32 v21, v75
v_mov_b32 v25, v76
v_mov_b32 v29, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=5 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift5_glvwblk3_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v18
v_mov_b32 v75, v22
v_mov_b32 v76, v26
v_mov_b32 v77, v30
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v19, v74
v_mov_b32 v23, v75
v_mov_b32 v27, v76
v_mov_b32 v31, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=5 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift5_glvwblk4_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v32
v_mov_b32 v75, v36
v_mov_b32 v76, v40
v_mov_b32 v77, v44
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v33, v74
v_mov_b32 v37, v75
v_mov_b32 v41, v76
v_mov_b32 v45, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=5 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift5_glvwblk5_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v34
v_mov_b32 v75, v38
v_mov_b32 v76, v42
v_mov_b32 v77, v46
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v35, v74
v_mov_b32 v39, v75
v_mov_b32 v43, v76
v_mov_b32 v47, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=5 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift5_glvwblk6_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v48
v_mov_b32 v75, v52
v_mov_b32 v76, v56
v_mov_b32 v77, v60
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v49, v74
v_mov_b32 v53, v75
v_mov_b32 v57, v76
v_mov_b32 v61, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=5 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift5_glvwblk7_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v50
v_mov_b32 v75, v54
v_mov_b32 v76, v58
v_mov_b32 v77, v62
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:192           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:192           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:192           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:192           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v51, v74
v_mov_b32 v55, v75
v_mov_b32 v59, v76
v_mov_b32 v63, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=6 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift6_glvwblk0_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v0
v_mov_b32 v75, v4
v_mov_b32 v76, v8
v_mov_b32 v77, v12
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v0
v_mov_b32 v75, v4
v_mov_b32 v76, v8
v_mov_b32 v77, v12
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v1, v74
v_mov_b32 v5, v75
v_mov_b32 v9, v76
v_mov_b32 v13, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v1, v74
v_mov_b32 v5, v75
v_mov_b32 v9, v76
v_mov_b32 v13, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=6 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift6_glvwblk1_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v2
v_mov_b32 v75, v6
v_mov_b32 v76, v10
v_mov_b32 v77, v14
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v2
v_mov_b32 v75, v6
v_mov_b32 v76, v10
v_mov_b32 v77, v14
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v3, v74
v_mov_b32 v7, v75
v_mov_b32 v11, v76
v_mov_b32 v15, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v3, v74
v_mov_b32 v7, v75
v_mov_b32 v11, v76
v_mov_b32 v15, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=6 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift6_glvwblk2_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v16
v_mov_b32 v75, v20
v_mov_b32 v76, v24
v_mov_b32 v77, v28
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v16
v_mov_b32 v75, v20
v_mov_b32 v76, v24
v_mov_b32 v77, v28
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v17, v74
v_mov_b32 v21, v75
v_mov_b32 v25, v76
v_mov_b32 v29, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v17, v74
v_mov_b32 v21, v75
v_mov_b32 v25, v76
v_mov_b32 v29, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=6 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift6_glvwblk3_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v18
v_mov_b32 v75, v22
v_mov_b32 v76, v26
v_mov_b32 v77, v30
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v18
v_mov_b32 v75, v22
v_mov_b32 v76, v26
v_mov_b32 v77, v30
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v19, v74
v_mov_b32 v23, v75
v_mov_b32 v27, v76
v_mov_b32 v31, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v19, v74
v_mov_b32 v23, v75
v_mov_b32 v27, v76
v_mov_b32 v31, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=6 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift6_glvwblk4_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v32
v_mov_b32 v75, v36
v_mov_b32 v76, v40
v_mov_b32 v77, v44
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v32
v_mov_b32 v75, v36
v_mov_b32 v76, v40
v_mov_b32 v77, v44
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v33, v74
v_mov_b32 v37, v75
v_mov_b32 v41, v76
v_mov_b32 v45, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v33, v74
v_mov_b32 v37, v75
v_mov_b32 v41, v76
v_mov_b32 v45, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=6 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift6_glvwblk5_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v34
v_mov_b32 v75, v38
v_mov_b32 v76, v42
v_mov_b32 v77, v46
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v34
v_mov_b32 v75, v38
v_mov_b32 v76, v42
v_mov_b32 v77, v46
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v35, v74
v_mov_b32 v39, v75
v_mov_b32 v43, v76
v_mov_b32 v47, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v35, v74
v_mov_b32 v39, v75
v_mov_b32 v43, v76
v_mov_b32 v47, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=6 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift6_glvwblk6_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v48
v_mov_b32 v75, v52
v_mov_b32 v76, v56
v_mov_b32 v77, v60
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v48
v_mov_b32 v75, v52
v_mov_b32 v76, v56
v_mov_b32 v77, v60
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v49, v74
v_mov_b32 v53, v75
v_mov_b32 v57, v76
v_mov_b32 v61, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v49, v74
v_mov_b32 v53, v75
v_mov_b32 v57, v76
v_mov_b32 v61, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=6 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift6_glvwblk7_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v50
v_mov_b32 v75, v54
v_mov_b32 v76, v58
v_mov_b32 v77, v62
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v50
v_mov_b32 v75, v54
v_mov_b32 v76, v58
v_mov_b32 v77, v62
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v51, v74
v_mov_b32 v55, v75
v_mov_b32 v59, v76
v_mov_b32 v63, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:128           // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:128           // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:128           // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:128           // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v51, v74
v_mov_b32 v55, v75
v_mov_b32 v59, v76
v_mov_b32 v63, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=7 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift7_glvwblk0_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v0
v_mov_b32 v75, v4
v_mov_b32 v76, v8
v_mov_b32 v77, v12
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v0
v_mov_b32 v75, v4
v_mov_b32 v76, v8
v_mov_b32 v77, v12
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v0
v_mov_b32 v75, v4
v_mov_b32 v76, v8
v_mov_b32 v77, v12
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v0, v74
v_mov_b32 v4, v75
v_mov_b32 v8, v76
v_mov_b32 v12, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v1, v74
v_mov_b32 v5, v75
v_mov_b32 v9, v76
v_mov_b32 v13, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v1, v74
v_mov_b32 v5, v75
v_mov_b32 v9, v76
v_mov_b32 v13, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v1
v_mov_b32 v75, v5
v_mov_b32 v76, v9
v_mov_b32 v77, v13
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v1, v74
v_mov_b32 v5, v75
v_mov_b32 v9, v76
v_mov_b32 v13, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=7 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift7_glvwblk1_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v2
v_mov_b32 v75, v6
v_mov_b32 v76, v10
v_mov_b32 v77, v14
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v2
v_mov_b32 v75, v6
v_mov_b32 v76, v10
v_mov_b32 v77, v14
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v2
v_mov_b32 v75, v6
v_mov_b32 v76, v10
v_mov_b32 v77, v14
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v2, v74
v_mov_b32 v6, v75
v_mov_b32 v10, v76
v_mov_b32 v14, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v3, v74
v_mov_b32 v7, v75
v_mov_b32 v11, v76
v_mov_b32 v15, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v3, v74
v_mov_b32 v7, v75
v_mov_b32 v11, v76
v_mov_b32 v15, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v3
v_mov_b32 v75, v7
v_mov_b32 v76, v11
v_mov_b32 v77, v15
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v3, v74
v_mov_b32 v7, v75
v_mov_b32 v11, v76
v_mov_b32 v15, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=7 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift7_glvwblk2_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v16
v_mov_b32 v75, v20
v_mov_b32 v76, v24
v_mov_b32 v77, v28
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v16
v_mov_b32 v75, v20
v_mov_b32 v76, v24
v_mov_b32 v77, v28
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v16
v_mov_b32 v75, v20
v_mov_b32 v76, v24
v_mov_b32 v77, v28
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v16, v74
v_mov_b32 v20, v75
v_mov_b32 v24, v76
v_mov_b32 v28, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v17, v74
v_mov_b32 v21, v75
v_mov_b32 v25, v76
v_mov_b32 v29, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v17, v74
v_mov_b32 v21, v75
v_mov_b32 v25, v76
v_mov_b32 v29, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v17
v_mov_b32 v75, v21
v_mov_b32 v76, v25
v_mov_b32 v77, v29
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v17, v74
v_mov_b32 v21, v75
v_mov_b32 v25, v76
v_mov_b32 v29, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=7 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift7_glvwblk3_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v18
v_mov_b32 v75, v22
v_mov_b32 v76, v26
v_mov_b32 v77, v30
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v18
v_mov_b32 v75, v22
v_mov_b32 v76, v26
v_mov_b32 v77, v30
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v18
v_mov_b32 v75, v22
v_mov_b32 v76, v26
v_mov_b32 v77, v30
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v18, v74
v_mov_b32 v22, v75
v_mov_b32 v26, v76
v_mov_b32 v30, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v19, v74
v_mov_b32 v23, v75
v_mov_b32 v27, v76
v_mov_b32 v31, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v19, v74
v_mov_b32 v23, v75
v_mov_b32 v27, v76
v_mov_b32 v31, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v19
v_mov_b32 v75, v23
v_mov_b32 v76, v27
v_mov_b32 v77, v31
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v19, v74
v_mov_b32 v23, v75
v_mov_b32 v27, v76
v_mov_b32 v31, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=7 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift7_glvwblk4_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v32
v_mov_b32 v75, v36
v_mov_b32 v76, v40
v_mov_b32 v77, v44
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v32
v_mov_b32 v75, v36
v_mov_b32 v76, v40
v_mov_b32 v77, v44
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v32
v_mov_b32 v75, v36
v_mov_b32 v76, v40
v_mov_b32 v77, v44
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v32, v74
v_mov_b32 v36, v75
v_mov_b32 v40, v76
v_mov_b32 v44, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v33, v74
v_mov_b32 v37, v75
v_mov_b32 v41, v76
v_mov_b32 v45, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v33, v74
v_mov_b32 v37, v75
v_mov_b32 v41, v76
v_mov_b32 v45, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v33
v_mov_b32 v75, v37
v_mov_b32 v76, v41
v_mov_b32 v77, v45
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v33, v74
v_mov_b32 v37, v75
v_mov_b32 v41, v76
v_mov_b32 v45, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=7 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift7_glvwblk5_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v34
v_mov_b32 v75, v38
v_mov_b32 v76, v42
v_mov_b32 v77, v46
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v34
v_mov_b32 v75, v38
v_mov_b32 v76, v42
v_mov_b32 v77, v46
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v34
v_mov_b32 v75, v38
v_mov_b32 v76, v42
v_mov_b32 v77, v46
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v34, v74
v_mov_b32 v38, v75
v_mov_b32 v42, v76
v_mov_b32 v46, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v35, v74
v_mov_b32 v39, v75
v_mov_b32 v43, v76
v_mov_b32 v47, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v35, v74
v_mov_b32 v39, v75
v_mov_b32 v43, v76
v_mov_b32 v47, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v35
v_mov_b32 v75, v39
v_mov_b32 v76, v43
v_mov_b32 v77, v47
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v35, v74
v_mov_b32 v39, v75
v_mov_b32 v43, v76
v_mov_b32 v47, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=7 glvwblk=0             */
/******************************************/
label_ShiftVectorComponents1_shift7_glvwblk6_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v48
v_mov_b32 v75, v52
v_mov_b32 v76, v56
v_mov_b32 v77, v60
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v48
v_mov_b32 v75, v52
v_mov_b32 v76, v56
v_mov_b32 v77, v60
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v48
v_mov_b32 v75, v52
v_mov_b32 v76, v56
v_mov_b32 v77, v60
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v48, v74
v_mov_b32 v52, v75
v_mov_b32 v56, v76
v_mov_b32 v60, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v49, v74
v_mov_b32 v53, v75
v_mov_b32 v57, v76
v_mov_b32 v61, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v49, v74
v_mov_b32 v53, v75
v_mov_b32 v57, v76
v_mov_b32 v61, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v49
v_mov_b32 v75, v53
v_mov_b32 v76, v57
v_mov_b32 v77, v61
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v49, v74
v_mov_b32 v53, v75
v_mov_b32 v57, v76
v_mov_b32 v61, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done

/******************************************/
/* shift d1 shift=7 glvwblk=1             */
/******************************************/
label_ShiftVectorComponents1_shift7_glvwblk7_0:
v_and_b32 v72, 63, v[vgprSerial]                   // permute register between threads
v_lshlrev_b32 v72, 0x2, v72                        // permute register between threads
v_lshrrev_b32 v73, 4, v[vgprSerial]                // v73 = v[vgprSerial] / 16
v_and_b32 v73, 3, v73                              // v73 = v73 % 4
v_mov_b32 v74, v50
v_mov_b32 v75, v54
v_mov_b32 v76, v58
v_mov_b32 v77, v62
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v50
v_mov_b32 v75, v54
v_mov_b32 v76, v58
v_mov_b32 v77, v62
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v50
v_mov_b32 v75, v54
v_mov_b32 v76, v58
v_mov_b32 v77, v62
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 3                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v50, v74
v_mov_b32 v54, v75
v_mov_b32 v58, v76
v_mov_b32 v62, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 0                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v51, v74
v_mov_b32 v55, v75
v_mov_b32 v59, v76
v_mov_b32 v63, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 1                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v51, v74
v_mov_b32 v55, v75
v_mov_b32 v59, v76
v_mov_b32 v63, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
v_mov_b32 v74, v51
v_mov_b32 v75, v55
v_mov_b32 v76, v59
v_mov_b32 v77, v63
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v74, v72, v74 offset:64            // permute edge values
ds_bpermute_b32 v75, v72, v75 offset:64            // permute edge values
ds_bpermute_b32 v76, v72, v76 offset:64            // permute edge values
ds_bpermute_b32 v77, v72, v77 offset:64            // permute edge values
s_waitcnt lgkmcnt(0)                               // wait for swizzle operation
s_mov_b32 s54, 2                                   // which thread need to shfit in this block
v_cmpx_eq_u32 s[54:55], v73, s54                   // is thread in edge glvw region
s_nop 3                                            // wait for exec mask
v_mov_b32 v51, v74
v_mov_b32 v55, v75
v_mov_b32 v59, v76
v_mov_b32 v63, v77
s_mov_b64 s[54:55], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[54:55]                    // all threads active
s_nop 3                                            // wait for exec mask
s_branch label_ShiftVectorComponents1_shift0_0     // done
label_ShiftVectorComponents1_shift0_0:

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v68, 6, v[vgprSerial]                // v68 = v[vgprSerial] / 64
v_lshrrev_b32 v69, 1, v68                          // v69 = v68 / 2
s_mov_b32 s43, 0x20                                // tempStore tmpSgpr
v_mul_lo_u32 v69, s43, v69                         // wave coordination offset 1
v_and_b32 v65, 63, v[vgprSerial]                   // v65 = v[vgprSerial] % 64
v_lshrrev_b32 v65, 4, v65                          // v65 = v65 / 16
v_add_lshl_u32 v65, v69, v65, 0                    // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v66, v65, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v67, v65, s[sgprStrideD1J]            //  offset 1
v_and_b32 v64, 1, v68                              // v64 = v68 % 2
s_mov_b32 s43, 0x20                                // tempStore tmpSgpr
v_mul_lo_u32 v64, s43, v64                         // wave coordination offset 0
v_and_b32 v69, 15, v[vgprSerial]                   // v69 = v[vgprSerial] % 16
v_add_lshl_u32 v64, v69, v64, 0                    // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s43, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_add_u32 v64, s43, v64                            // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s43, 256, s[sgprWorkGroup1]              // wgp1 * MT1
v_add_u32 v65, s43, v65                            // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_waitcnt lgkmcnt(0)                               // wait for 36 bytes of kern args.
s_mov_b32 s[sgprSrdScaleAlphaVec+0], s[sgprAddressScaleAlphaVec+0] // init SRD base address (lower)
s_mov_b32 s[sgprSrdScaleAlphaVec+1], s[sgprAddressScaleAlphaVec+1] // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdScaleAlphaVec+3], Srd127_96     // Set bits 127_96 in post-loop SRD
s_cmp_eq_u64 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], 0 // s[AddressScaleAlphaVec] == 0 ?
s_cbranch_scc0 label_ScaleAlphaVec_1AddrValid      // branch if s[AddressScaleAlphaVec] != 0
s_mov_b32 s[sgprSrdScaleAlphaVec+2], 0
s_branch label_ScaleAlphaVec_1AddrValid_End
label_ScaleAlphaVec_1AddrValid:
s_mov_b32 s[sgprSrdScaleAlphaVec+2], s[sgprSizeI]
label_ScaleAlphaVec_1AddrValid_End:

s_mul_i32 s[sgprSrdScaleAlphaVec+2], 0x4, s[sgprSrdScaleAlphaVec+2] // ScaleAlphaVec scaled by BPE
s_add_u32 s43, s[sgprWorkGroup2], 0x1
s_mul_i32 s43, s[sgprBiasStride], s43              // stride * (wg+1)
s_cmp_eq_u32 s43, 0x0                              // bias stride = 0?
s_cselect_b32 s43, s[sgprSizeI], s43
s_mov_b32 s[sgprSrdBias+0], s[sgprAddressBias+0]   // init SRD base address (lower)
s_mov_b32 s[sgprSrdBias+1], s[sgprAddressBias+1]   // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdBias+3], Srd127_96              // Set bits 127_96 in post-loop SRD
s_cmp_eq_u64 s[sgprAddressBias:sgprAddressBias+1], 0 // s[AddressBias] == 0 ?
s_cbranch_scc0 label_Bias_1AddrValid               // branch if s[AddressBias] != 0
s_mov_b32 s[sgprSrdBias+2], 0
s_branch label_Bias_1AddrValid_End
label_Bias_1AddrValid:
s_mov_b32 s[sgprSrdBias+2], s43
label_Bias_1AddrValid_End:

label_Load_Biasf32_1:
s_cmpk_lg_u32 s[sgprBiasType], 0                   // BiasType != 0
s_cbranch_scc1 label_Load_Biasbf16_1               // Branch if true

/******************************************/
/* Read Bias to LDS                       */
/******************************************/
s_mul_i32 s[sgprSrdBias+2], 0x4, s[sgprSrdBias+2]  // scaled by BPE
s_mul_i32 s43, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_add_u32 v72, s43, v[vgprSerial]                  // coord 0 = wgp0 * MT0 + thread offset
s_mul_i32 s43, s[sgprBiasStride], s[sgprWorkGroup2] // Stride * WG
v_add_u32 v72, s43, v72                            // coord 0 = wgp0 * MT0 + thread offset + Stride * WG
v_lshlrev_b32 v72, 0x2, v72                        // Global bias address scaled by BPE
buffer_load_dword v68, v72, s[sgprSrdBias:sgprSrdBias+3], 0 offen offset:0 // load bias
v_lshlrev_b32 v72, 0x2, v[vgprSerial]              // Local bias address scaled by BPE
s_waitcnt vmcnt(0)                                 // wait for bias load
s_barrier                                          // Wait for all wavefronts
ds_write_b32 v72, v68 offset:0                     // store bias
s_branch label_Load_Bias_End_1                     // Branch to load bias end
label_Load_Biasbf16_1:
s_cmpk_lg_u32 s[sgprBiasType], 7                   // BiasType != 7
s_cbranch_scc1 label_Load_Bias_End_1               // Branch if true

/******************************************/
/* Read Bias to LDS                       */
/******************************************/
s_mul_i32 s[sgprSrdBias+2], 0x2, s[sgprSrdBias+2]  // scaled by BPE
s_mul_i32 s43, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_add_u32 v72, s43, v[vgprSerial]                  // coord 0 = wgp0 * MT0 + thread offset
s_mul_i32 s43, s[sgprBiasStride], s[sgprWorkGroup2] // Stride * WG
v_add_u32 v72, s43, v72                            // coord 0 = wgp0 * MT0 + thread offset + Stride * WG
v_lshlrev_b32 v72, 0x1, v72                        // Global bias address scaled by BPE
buffer_load_short_d16 v68, v72, s[sgprSrdBias:sgprSrdBias+3], 0 offen offset:0 // load bias
v_lshlrev_b32 v72, 0x2, v[vgprSerial]              // Local bias address scaled by BPE
s_waitcnt vmcnt(0)                                 // wait for bias load
s_barrier                                          // Wait for all wavefronts
v_lshlrev_b32 v68, 16, v68                         // convert bf16 to fp32
ds_write_b32 v72, v68 offset:0                     // store bias
s_branch label_Load_Bias_End_1                     // Branch to load bias end
label_Load_Bias_End_1:

s_waitcnt lgkmcnt(0)                               // Wait for Bias LDS write
s_barrier                                          // Bias LDS write barrier

/******************************************/
/* Read ScaleAlphaVec to LDS              */
/******************************************/
s_mul_i32 s43, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_add_u32 v72, s43, v[vgprSerial]                  // coord 0 = wgp0 * MT0 + thread offset
//s_mul_i32 s43, s[sgprBiasStride], s[sgprWorkGroup2] // Stride * WG
//v_add_u32 v72, s43, v72                            // coord 0 = wgp0 * MT0 + thread offset + Stride * WG
v_lshlrev_b32 v72, 0x2, v72                        // Global ScaleAlphaVec address scaled by BPE
buffer_load_dword v68, v72, s[sgprSrdScaleAlphaVec:sgprSrdScaleAlphaVec+3], 0 offen offset:0 // load ScaleAlphaVec
v_lshlrev_b32 v72, 0x2, v[vgprSerial]              // Local ScaleAlphaVec address scaled by BPE
s_waitcnt vmcnt(0)                                 // wait for ScaleAlphaVec load
s_barrier                                          // Wait for all wavefronts
ds_write_b32 v72, v68 offset:0x1000                // store ScaleAlphaVec
s_branch label_Load_ScaleAlphaVec_End_1            // Branch to load ScaleAlphaVec end
label_Load_ScaleAlphaVec_End_1:

s_waitcnt lgkmcnt(0)                               // Wait for ScaleAlphaVec LDS write
s_barrier                                          // ScaleAlphaVec LDS write barrier

/******************************************/
/* Read Bias & ScaleAlphaVec from LDS     */
/******************************************/
s_mul_i32 s66, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v76, v64, s66
v_lshlrev_b32 v76, 0x2, v76                        // Bias & ScaleAlphaVec address scaled by BPE

ds_read_b32 v[vgprValuBias_M+0], v76 offset:0                      // load bias
ds_read_b32 v[vgprValuBias_M+1], v76 offset:64                     // load bias
ds_read_b32 v[vgprValuBias_M+2], v76 offset:256                    // load bias
ds_read_b32 v[vgprValuBias_M+3], v76 offset:320                    // load bias

s_mov_b32 s66, 4096
v_add_co_u32 v76, vcc, v76, s66

ds_read_b32 v[vgprValuScalAlpha_M+0], v76 offset:0                 // load ScaleAlphaVec
ds_read_b32 v[vgprValuScalAlpha_M+1], v76 offset:64                // load ScaleAlphaVec
ds_read_b32 v[vgprValuScalAlpha_M+2], v76 offset:256               // load ScaleAlphaVec
ds_read_b32 v[vgprValuScalAlpha_M+3], v76 offset:320               // load ScaleAlphaVec

s_waitcnt lgkmcnt(0)                               // Wait for Bias LDS write
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v[vgprValuScalAlpha_M+0], 1.0, v[vgprValuScalAlpha_M+0], s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v[vgprValuScalAlpha_M+1], 1.0, v[vgprValuScalAlpha_M+1], s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v[vgprValuScalAlpha_M+2], 1.0, v[vgprValuScalAlpha_M+2], s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v[vgprValuScalAlpha_M+3], 1.0, v[vgprValuScalAlpha_M+3], s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0



s_cmpk_eq_u32 s[sgprActivationType], 1             // activationType == 1
s_cbranch_scc1 label_To_Activation_Abs_VW1_1       // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 2             // activationType == 2
s_cbranch_scc1 label_To_Activation_Clippedrelu_VW1_1 // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 3             // activationType == 3
s_cbranch_scc1 label_To_Activation_Gelu_VW1_1      // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 4             // activationType == 4
s_cbranch_scc1 label_To_Activation_Leakyrelu_VW1_1 // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 5             // activationType == 5
s_cbranch_scc1 label_To_Activation_Relu_VW1_1      // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 6             // activationType == 6
s_cbranch_scc1 label_To_Activation_Sigmoid_VW1_1   // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 7             // activationType == 7
s_cbranch_scc1 label_To_Activation_Tanh_VW1_1      // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 9             // activationType == 9
s_cbranch_scc1 label_To_Activation_Geluscaling_VW1_1 // Branch if true
label_To_Activation_None_VW1_1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_None_VW1, 0x4      // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd_1
label_To_Activation_Abs_VW1_1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Abs_VW1, 0x4       // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd_1
label_To_Activation_Clippedrelu_VW1_1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Clippedrelu_VW1, 0x4 // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd_1
label_To_Activation_Gelu_VW1_1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Gelu_VW1, 0x4      // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd_1
label_To_Activation_Leakyrelu_VW1_1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Leakyrelu_VW1, 0x4 // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd_1
label_To_Activation_Relu_VW1_1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Relu_VW1, 0x4      // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd_1
label_To_Activation_Sigmoid_VW1_1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Sigmoid_VW1, 0x4   // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd_1
label_To_Activation_Tanh_VW1_1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Tanh_VW1, 0x4      // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd_1
label_To_Activation_Geluscaling_VW1_1:
s_getpc_b64 s[54:55]                               // addr of next instr
s_add_i32 s43, label_Activation_Geluscaling_VW1, 0x4 // target branch offset
s_add_u32 s54, s54, s43                            // add target branch offset
s_addc_u32 s55, s55, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd_1
label_ActivationSetPCAddrEnd_1:
s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 label_GW_Beta_3                     // Branch if Beta is not zero

s_and_b32 s66, 127, s[sgprSizeI]                   // s66 = s[sgprSizeI] % 128
s_add_u32 s67, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s67                // wg0 >= nwg0-1 ?
s_cselect_b32 s66, s66, 0                          // set rMT0
s_cmpk_gt_u32 s66, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_GW_B0_E1                      // jump if edges required
s_and_b32 s66, 255, s[sgprSizeJ]                   // s66 = s[sgprSizeJ] % 256
s_add_u32 s67, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s67                // wg1 >= nwg1-1
s_cselect_b32 s66, s66, 0                          // set rMT1
s_cmpk_gt_u32 s66, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_GW_B0_E1                      // jump if edges required
label_GW_B0_E0_1:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
v_add_lshl_u32 v74, v67, v64, 0x1                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16


v_mul_f32 v[vgprValuC+0], v[vgprValuScalAlpha_M+0], v[vgprValuC+0]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+0]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v0, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+0], v[vgprValuC+0] // check Nan
v_bfe_u32 v70, v[vgprValuC+0], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+0], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+0], v70, v72, s[66:67]
v_lshrrev_b32 v0, 16, v[vgprValuC+0]               // convert C to bf16
buffer_store_short v0, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D


v_mul_f32 v[vgprValuC+4], v[vgprValuScalAlpha_M+1], v[vgprValuC+4]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+4]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v4, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+4], v[vgprValuC+4] // check Nan
v_bfe_u32 v70, v[vgprValuC+4], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+4], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+4], v70, v72, s[66:67]
v_lshrrev_b32 v4, 16, v[vgprValuC+4]               // convert C to bf16
buffer_store_short v4, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D


v_mul_f32 v[vgprValuC+8], v[vgprValuScalAlpha_M+2], v[vgprValuC+8]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+8]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v8, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+8], v[vgprValuC+8] // check Nan
v_bfe_u32 v70, v[vgprValuC+8], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+8], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+8], v70, v72, s[66:67]
v_lshrrev_b32 v8, 16, v[vgprValuC+8]               // convert C to bf16
buffer_store_short v8, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D


v_mul_f32 v[vgprValuC+12], v[vgprValuScalAlpha_M+3], v[vgprValuC+12]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+12]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v12, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+12], v[vgprValuC+12] // check Nan
v_bfe_u32 v70, v[vgprValuC+12], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+12], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+12], v70, v72, s[66:67]
v_lshrrev_b32 v12, 16, v[vgprValuC+12]             // convert C to bf16
buffer_store_short v12, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+1], v[vgprValuScalAlpha_M+0], v[vgprValuC+1]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+1]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v1, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+1], v[vgprValuC+1] // check Nan
v_bfe_u32 v70, v[vgprValuC+1], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+1], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+1], v70, v72, s[66:67]
v_lshrrev_b32 v1, 16, v[vgprValuC+1]               // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v1, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+5], v[vgprValuScalAlpha_M+1], v[vgprValuC+5]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+5]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v5, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+5], v[vgprValuC+5] // check Nan
v_bfe_u32 v70, v[vgprValuC+5], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+5], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+5], v70, v72, s[66:67]
v_lshrrev_b32 v5, 16, v[vgprValuC+5]               // convert C to bf16
buffer_store_short v5, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+9], v[vgprValuScalAlpha_M+2], v[vgprValuC+9]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+9]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v9, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+9], v[vgprValuC+9] // check Nan
v_bfe_u32 v70, v[vgprValuC+9], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+9], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+9], v70, v72, s[66:67]
v_lshrrev_b32 v9, 16, v[vgprValuC+9]               // convert C to bf16
buffer_store_short v9, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+13], v[vgprValuScalAlpha_M+3], v[vgprValuC+13]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+13]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v13, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+13], v[vgprValuC+13] // check Nan
v_bfe_u32 v70, v[vgprValuC+13], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+13], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+13], v70, v72, s[66:67]
v_lshrrev_b32 v13, 16, v[vgprValuC+13]             // convert C to bf16
buffer_store_short v13, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+2], v[vgprValuScalAlpha_M+0], v[vgprValuC+2]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+2]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v2, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+2], v[vgprValuC+2] // check Nan
v_bfe_u32 v70, v[vgprValuC+2], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+2], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+2], v70, v72, s[66:67]
v_lshrrev_b32 v2, 16, v[vgprValuC+2]               // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v2, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+6], v[vgprValuScalAlpha_M+1], v[vgprValuC+6]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+6]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v6, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+6], v[vgprValuC+6] // check Nan
v_bfe_u32 v70, v[vgprValuC+6], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+6], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+6], v70, v72, s[66:67]
v_lshrrev_b32 v6, 16, v[vgprValuC+6]               // convert C to bf16
buffer_store_short v6, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+10], v[vgprValuScalAlpha_M+2], v[vgprValuC+10]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+10]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v10, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+10], v[vgprValuC+10] // check Nan
v_bfe_u32 v70, v[vgprValuC+10], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+10], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+10], v70, v72, s[66:67]
v_lshrrev_b32 v10, 16, v[vgprValuC+10]             // convert C to bf16
buffer_store_short v10, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+14], v[vgprValuScalAlpha_M+3], v[vgprValuC+14]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+14]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v14, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+14], v[vgprValuC+14] // check Nan
v_bfe_u32 v70, v[vgprValuC+14], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+14], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+14], v70, v72, s[66:67]
v_lshrrev_b32 v14, 16, v[vgprValuC+14]             // convert C to bf16
buffer_store_short v14, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+3], v[vgprValuScalAlpha_M+0], v[vgprValuC+3]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+3]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v3, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+3], v[vgprValuC+3] // check Nan
v_bfe_u32 v70, v[vgprValuC+3], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+3], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+3], v70, v72, s[66:67]
v_lshrrev_b32 v3, 16, v[vgprValuC+3]               // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v3, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+7], v[vgprValuScalAlpha_M+1], v[vgprValuC+7]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+7]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v7, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+7], v[vgprValuC+7] // check Nan
v_bfe_u32 v70, v[vgprValuC+7], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+7], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+7], v70, v72, s[66:67]
v_lshrrev_b32 v7, 16, v[vgprValuC+7]               // convert C to bf16
buffer_store_short v7, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+11], v[vgprValuScalAlpha_M+2], v[vgprValuC+11]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+11]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v11, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+11], v[vgprValuC+11] // check Nan
v_bfe_u32 v70, v[vgprValuC+11], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+11], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+11], v70, v72, s[66:67]
v_lshrrev_b32 v11, 16, v[vgprValuC+11]             // convert C to bf16
buffer_store_short v11, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+15], v[vgprValuScalAlpha_M+3], v[vgprValuC+15]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+15]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v15, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+15], v[vgprValuC+15] // check Nan
v_bfe_u32 v70, v[vgprValuC+15], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+15], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+15], v70, v72, s[66:67]
v_lshrrev_b32 v15, 16, v[vgprValuC+15]             // convert C to bf16
buffer_store_short v15, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #1 (d1,d0,vc1,vc0) = */
/*    (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */

/* rC *= alpha batchElements=[(4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16


v_mul_f32 v[vgprValuC+16], v[vgprValuScalAlpha_M+0], v[vgprValuC+16]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+16]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v16, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+16], v[vgprValuC+16] // check Nan
v_bfe_u32 v70, v[vgprValuC+16], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+16], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+16], v70, v72, s[66:67]
v_lshrrev_b32 v16, 16, v[vgprValuC+16]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v16, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D


v_mul_f32 v[vgprValuC+20], v[vgprValuScalAlpha_M+1], v[vgprValuC+20]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+20]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v20, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+20], v[vgprValuC+20] // check Nan
v_bfe_u32 v70, v[vgprValuC+20], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+20], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+20], v70, v72, s[66:67]
v_lshrrev_b32 v20, 16, v[vgprValuC+20]             // convert C to bf16
buffer_store_short v20, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D


v_mul_f32 v[vgprValuC+24], v[vgprValuScalAlpha_M+2], v[vgprValuC+24]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+24]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v24, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+24], v[vgprValuC+24] // check Nan
v_bfe_u32 v70, v[vgprValuC+24], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+24], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+24], v70, v72, s[66:67]
v_lshrrev_b32 v24, 16, v[vgprValuC+24]             // convert C to bf16
buffer_store_short v24, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D


v_mul_f32 v[vgprValuC+28], v[vgprValuScalAlpha_M+3], v[vgprValuC+28]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+28]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v28, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+28], v[vgprValuC+28] // check Nan
v_bfe_u32 v70, v[vgprValuC+28], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+28], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+28], v70, v72, s[66:67]
v_lshrrev_b32 v28, 16, v[vgprValuC+28]             // convert C to bf16
buffer_store_short v28, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+17], v[vgprValuScalAlpha_M+0], v[vgprValuC+17]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+17]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v17, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+17], v[vgprValuC+17] // check Nan
v_bfe_u32 v70, v[vgprValuC+17], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+17], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+17], v70, v72, s[66:67]
v_lshrrev_b32 v17, 16, v[vgprValuC+17]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v17, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+21], v[vgprValuScalAlpha_M+1], v[vgprValuC+21]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+21]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v21, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+21], v[vgprValuC+21] // check Nan
v_bfe_u32 v70, v[vgprValuC+21], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+21], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+21], v70, v72, s[66:67]
v_lshrrev_b32 v21, 16, v[vgprValuC+21]             // convert C to bf16
buffer_store_short v21, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+25], v[vgprValuScalAlpha_M+2], v[vgprValuC+25]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+25]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v25, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+25], v[vgprValuC+25] // check Nan
v_bfe_u32 v70, v[vgprValuC+25], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+25], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+25], v70, v72, s[66:67]
v_lshrrev_b32 v25, 16, v[vgprValuC+25]             // convert C to bf16
buffer_store_short v25, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+29], v[vgprValuScalAlpha_M+3], v[vgprValuC+29]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+29]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v29, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+29], v[vgprValuC+29] // check Nan
v_bfe_u32 v70, v[vgprValuC+29], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+29], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+29], v70, v72, s[66:67]
v_lshrrev_b32 v29, 16, v[vgprValuC+29]             // convert C to bf16
buffer_store_short v29, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+18], v[vgprValuScalAlpha_M+0], v[vgprValuC+18]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+18]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v18, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+18], v[vgprValuC+18] // check Nan
v_bfe_u32 v70, v[vgprValuC+18], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+18], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+18], v70, v72, s[66:67]
v_lshrrev_b32 v18, 16, v[vgprValuC+18]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v18, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+22], v[vgprValuScalAlpha_M+1], v[vgprValuC+22]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+22]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v22, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+22], v[vgprValuC+22] // check Nan
v_bfe_u32 v70, v[vgprValuC+22], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+22], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+22], v70, v72, s[66:67]
v_lshrrev_b32 v22, 16, v[vgprValuC+22]             // convert C to bf16
buffer_store_short v22, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+26], v[vgprValuScalAlpha_M+2], v[vgprValuC+26]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+26]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v26, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+26], v[vgprValuC+26] // check Nan
v_bfe_u32 v70, v[vgprValuC+26], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+26], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+26], v70, v72, s[66:67]
v_lshrrev_b32 v26, 16, v[vgprValuC+26]             // convert C to bf16
buffer_store_short v26, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+30], v[vgprValuScalAlpha_M+3], v[vgprValuC+30]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+30]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v30, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+30], v[vgprValuC+30] // check Nan
v_bfe_u32 v70, v[vgprValuC+30], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+30], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+30], v70, v72, s[66:67]
v_lshrrev_b32 v30, 16, v[vgprValuC+30]             // convert C to bf16
buffer_store_short v30, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+19], v[vgprValuScalAlpha_M+0], v[vgprValuC+19]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+19]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v19, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+19], v[vgprValuC+19] // check Nan
v_bfe_u32 v70, v[vgprValuC+19], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+19], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+19], v70, v72, s[66:67]
v_lshrrev_b32 v19, 16, v[vgprValuC+19]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v19, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+23], v[vgprValuScalAlpha_M+1], v[vgprValuC+23]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+23]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v23, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+23], v[vgprValuC+23] // check Nan
v_bfe_u32 v70, v[vgprValuC+23], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+23], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+23], v70, v72, s[66:67]
v_lshrrev_b32 v23, 16, v[vgprValuC+23]             // convert C to bf16
buffer_store_short v23, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+27], v[vgprValuScalAlpha_M+2], v[vgprValuC+27]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+27]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v27, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+27], v[vgprValuC+27] // check Nan
v_bfe_u32 v70, v[vgprValuC+27], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+27], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+27], v70, v72, s[66:67]
v_lshrrev_b32 v27, 16, v[vgprValuC+27]             // convert C to bf16
buffer_store_short v27, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+31], v[vgprValuScalAlpha_M+3], v[vgprValuC+31]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+31]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v31, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+31], v[vgprValuC+31] // check Nan
v_bfe_u32 v70, v[vgprValuC+31], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+31], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+31], v70, v72, s[66:67]
v_lshrrev_b32 v31, 16, v[vgprValuC+31]             // convert C to bf16
buffer_store_short v31, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #2 (d1,d0,vc1,vc0) = */
/*    (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (8,3,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (9,3,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (10,3,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (11,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(8,0,0,0) */

/* rC *= alpha batchElements=[(8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (8, 3, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (9, 3, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (10, 3, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (11, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16


v_mul_f32 v[vgprValuC+32], v[vgprValuScalAlpha_M+0], v[vgprValuC+32]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+32]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v32, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+32], v[vgprValuC+32] // check Nan
v_bfe_u32 v70, v[vgprValuC+32], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+32], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+32], v70, v72, s[66:67]
v_lshrrev_b32 v32, 16, v[vgprValuC+32]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 200                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v32, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D


v_mul_f32 v[vgprValuC+36], v[vgprValuScalAlpha_M+1], v[vgprValuC+36]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+36]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v36, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+36], v[vgprValuC+36] // check Nan
v_bfe_u32 v70, v[vgprValuC+36], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+36], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+36], v70, v72, s[66:67]
v_lshrrev_b32 v36, 16, v[vgprValuC+36]             // convert C to bf16
buffer_store_short v36, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D


v_mul_f32 v[vgprValuC+40], v[vgprValuScalAlpha_M+2], v[vgprValuC+40]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+40]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v40, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+40], v[vgprValuC+40] // check Nan
v_bfe_u32 v70, v[vgprValuC+40], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+40], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+40], v70, v72, s[66:67]
v_lshrrev_b32 v40, 16, v[vgprValuC+40]             // convert C to bf16
buffer_store_short v40, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D


v_mul_f32 v[vgprValuC+44], v[vgprValuScalAlpha_M+3], v[vgprValuC+44]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+44]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v44, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+44], v[vgprValuC+44] // check Nan
v_bfe_u32 v70, v[vgprValuC+44], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+44], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+44], v70, v72, s[66:67]
v_lshrrev_b32 v44, 16, v[vgprValuC+44]             // convert C to bf16
buffer_store_short v44, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+33], v[vgprValuScalAlpha_M+0], v[vgprValuC+33]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+33]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v33, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+33], v[vgprValuC+33] // check Nan
v_bfe_u32 v70, v[vgprValuC+33], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+33], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+33], v70, v72, s[66:67]
v_lshrrev_b32 v33, 16, v[vgprValuC+33]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v33, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+37], v[vgprValuScalAlpha_M+1], v[vgprValuC+37]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+37]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v37, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+37], v[vgprValuC+37] // check Nan
v_bfe_u32 v70, v[vgprValuC+37], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+37], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+37], v70, v72, s[66:67]
v_lshrrev_b32 v37, 16, v[vgprValuC+37]             // convert C to bf16
buffer_store_short v37, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+41], v[vgprValuScalAlpha_M+2], v[vgprValuC+41]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+41]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v41, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+41], v[vgprValuC+41] // check Nan
v_bfe_u32 v70, v[vgprValuC+41], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+41], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+41], v70, v72, s[66:67]
v_lshrrev_b32 v41, 16, v[vgprValuC+41]             // convert C to bf16
buffer_store_short v41, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+45], v[vgprValuScalAlpha_M+3], v[vgprValuC+45]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+45]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v45, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+45], v[vgprValuC+45] // check Nan
v_bfe_u32 v70, v[vgprValuC+45], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+45], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+45], v70, v72, s[66:67]
v_lshrrev_b32 v45, 16, v[vgprValuC+45]             // convert C to bf16
buffer_store_short v45, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+34], v[vgprValuScalAlpha_M+0], v[vgprValuC+34]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+34]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v34, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+34], v[vgprValuC+34] // check Nan
v_bfe_u32 v70, v[vgprValuC+34], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+34], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+34], v70, v72, s[66:67]
v_lshrrev_b32 v34, 16, v[vgprValuC+34]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v34, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+38], v[vgprValuScalAlpha_M+1], v[vgprValuC+38]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+38]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v38, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+38], v[vgprValuC+38] // check Nan
v_bfe_u32 v70, v[vgprValuC+38], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+38], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+38], v70, v72, s[66:67]
v_lshrrev_b32 v38, 16, v[vgprValuC+38]             // convert C to bf16
buffer_store_short v38, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+42], v[vgprValuScalAlpha_M+2], v[vgprValuC+42]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+42]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v42, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+42], v[vgprValuC+42] // check Nan
v_bfe_u32 v70, v[vgprValuC+42], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+42], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+42], v70, v72, s[66:67]
v_lshrrev_b32 v42, 16, v[vgprValuC+42]             // convert C to bf16
buffer_store_short v42, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+46], v[vgprValuScalAlpha_M+3], v[vgprValuC+46]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+46]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v46, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+46], v[vgprValuC+46] // check Nan
v_bfe_u32 v70, v[vgprValuC+46], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+46], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+46], v70, v72, s[66:67]
v_lshrrev_b32 v46, 16, v[vgprValuC+46]             // convert C to bf16
buffer_store_short v46, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+35], v[vgprValuScalAlpha_M+0], v[vgprValuC+35]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+35]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v35, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+35], v[vgprValuC+35] // check Nan
v_bfe_u32 v70, v[vgprValuC+35], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+35], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+35], v70, v72, s[66:67]
v_lshrrev_b32 v35, 16, v[vgprValuC+35]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v35, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+39], v[vgprValuScalAlpha_M+1], v[vgprValuC+39]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+39]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v39, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+39], v[vgprValuC+39] // check Nan
v_bfe_u32 v70, v[vgprValuC+39], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+39], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+39], v70, v72, s[66:67]
v_lshrrev_b32 v39, 16, v[vgprValuC+39]             // convert C to bf16
buffer_store_short v39, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+43], v[vgprValuScalAlpha_M+2], v[vgprValuC+43]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+43]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v43, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+43], v[vgprValuC+43] // check Nan
v_bfe_u32 v70, v[vgprValuC+43], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+43], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+43], v70, v72, s[66:67]
v_lshrrev_b32 v43, 16, v[vgprValuC+43]             // convert C to bf16
buffer_store_short v43, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+47], v[vgprValuScalAlpha_M+3], v[vgprValuC+47]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+47]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v47, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+47], v[vgprValuC+47] // check Nan
v_bfe_u32 v70, v[vgprValuC+47], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+47], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+47], v70, v72, s[66:67]
v_lshrrev_b32 v47, 16, v[vgprValuC+47]             // convert C to bf16
buffer_store_short v47, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #3 (d1,d0,vc1,vc0) = */
/*    (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (12,3,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (13,3,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (14,3,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (15,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */

/* rC *= alpha batchElements=[(12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (12, 3, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (13, 3, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (14, 3, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (15, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16


v_mul_f32 v[vgprValuC+48], v[vgprValuScalAlpha_M+0], v[vgprValuC+48]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+48]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v48, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+48], v[vgprValuC+48] // check Nan
v_bfe_u32 v70, v[vgprValuC+48], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+48], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+48], v70, v72, s[66:67]
v_lshrrev_b32 v48, 16, v[vgprValuC+48]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v48, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D


v_mul_f32 v[vgprValuC+52], v[vgprValuScalAlpha_M+1], v[vgprValuC+52]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+52]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v52, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+52], v[vgprValuC+52] // check Nan
v_bfe_u32 v70, v[vgprValuC+52], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+52], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+52], v70, v72, s[66:67]
v_lshrrev_b32 v52, 16, v[vgprValuC+52]             // convert C to bf16
buffer_store_short v52, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D


v_mul_f32 v[vgprValuC+56], v[vgprValuScalAlpha_M+2], v[vgprValuC+56]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+56]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v56, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+56], v[vgprValuC+56] // check Nan
v_bfe_u32 v70, v[vgprValuC+56], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+56], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+56], v70, v72, s[66:67]
v_lshrrev_b32 v56, 16, v[vgprValuC+56]             // convert C to bf16
buffer_store_short v56, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D


v_mul_f32 v[vgprValuC+60], v[vgprValuScalAlpha_M+3], v[vgprValuC+60]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+60]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v60, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+60], v[vgprValuC+60] // check Nan
v_bfe_u32 v70, v[vgprValuC+60], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+60], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+60], v70, v72, s[66:67]
v_lshrrev_b32 v60, 16, v[vgprValuC+60]             // convert C to bf16
buffer_store_short v60, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+49], v[vgprValuScalAlpha_M+0], v[vgprValuC+49]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+49]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v49, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+49], v[vgprValuC+49] // check Nan
v_bfe_u32 v70, v[vgprValuC+49], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+49], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+49], v70, v72, s[66:67]
v_lshrrev_b32 v49, 16, v[vgprValuC+49]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v49, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+53], v[vgprValuScalAlpha_M+1], v[vgprValuC+53]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+53]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v53, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+53], v[vgprValuC+53] // check Nan
v_bfe_u32 v70, v[vgprValuC+53], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+53], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+53], v70, v72, s[66:67]
v_lshrrev_b32 v53, 16, v[vgprValuC+53]             // convert C to bf16
buffer_store_short v53, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+57], v[vgprValuScalAlpha_M+2], v[vgprValuC+57]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+57]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v57, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+57], v[vgprValuC+57] // check Nan
v_bfe_u32 v70, v[vgprValuC+57], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+57], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+57], v70, v72, s[66:67]
v_lshrrev_b32 v57, 16, v[vgprValuC+57]             // convert C to bf16
buffer_store_short v57, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+61], v[vgprValuScalAlpha_M+3], v[vgprValuC+61]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+61]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v61, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+61], v[vgprValuC+61] // check Nan
v_bfe_u32 v70, v[vgprValuC+61], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+61], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+61], v70, v72, s[66:67]
v_lshrrev_b32 v61, 16, v[vgprValuC+61]             // convert C to bf16
buffer_store_short v61, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+50], v[vgprValuScalAlpha_M+0], v[vgprValuC+50]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+50]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v50, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+50], v[vgprValuC+50] // check Nan
v_bfe_u32 v70, v[vgprValuC+50], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+50], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+50], v70, v72, s[66:67]
v_lshrrev_b32 v50, 16, v[vgprValuC+50]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v50, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+54], v[vgprValuScalAlpha_M+1], v[vgprValuC+54]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+54]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v54, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+54], v[vgprValuC+54] // check Nan
v_bfe_u32 v70, v[vgprValuC+54], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+54], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+54], v70, v72, s[66:67]
v_lshrrev_b32 v54, 16, v[vgprValuC+54]             // convert C to bf16
buffer_store_short v54, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+58], v[vgprValuScalAlpha_M+2], v[vgprValuC+58]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+58]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v58, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+58], v[vgprValuC+58] // check Nan
v_bfe_u32 v70, v[vgprValuC+58], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+58], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+58], v70, v72, s[66:67]
v_lshrrev_b32 v58, 16, v[vgprValuC+58]             // convert C to bf16
buffer_store_short v58, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+62], v[vgprValuScalAlpha_M+3], v[vgprValuC+62]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+62]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v62, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+62], v[vgprValuC+62] // check Nan
v_bfe_u32 v70, v[vgprValuC+62], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+62], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+62], v70, v72, s[66:67]
v_lshrrev_b32 v62, 16, v[vgprValuC+62]             // convert C to bf16
buffer_store_short v62, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

v_mul_f32 v[vgprValuC+51], v[vgprValuScalAlpha_M+0], v[vgprValuC+51]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+51]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v51, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+51], v[vgprValuC+51] // check Nan
v_bfe_u32 v70, v[vgprValuC+51], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+51], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+51], v70, v72, s[66:67]
v_lshrrev_b32 v51, 16, v[vgprValuC+51]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v51, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+55], v[vgprValuScalAlpha_M+1], v[vgprValuC+55]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+55]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v55, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+55], v[vgprValuC+55] // check Nan
v_bfe_u32 v70, v[vgprValuC+55], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+55], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+55], v70, v72, s[66:67]
v_lshrrev_b32 v55, 16, v[vgprValuC+55]             // convert C to bf16
buffer_store_short v55, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

v_mul_f32 v[vgprValuC+59], v[vgprValuScalAlpha_M+2], v[vgprValuC+59]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+59]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v59, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+59], v[vgprValuC+59] // check Nan
v_bfe_u32 v70, v[vgprValuC+59], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+59], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+59], v70, v72, s[66:67]
v_lshrrev_b32 v59, 16, v[vgprValuC+59]             // convert C to bf16
buffer_store_short v59, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

v_mul_f32 v[vgprValuC+63], v[vgprValuScalAlpha_M+3], v[vgprValuC+63]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+63]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v63, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+63], v[vgprValuC+63] // check Nan
v_bfe_u32 v70, v[vgprValuC+63], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+63], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+63], v70, v72, s[66:67]
v_lshrrev_b32 v63, 16, v[vgprValuC+63]             // convert C to bf16
buffer_store_short v63, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_E1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=8 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v74, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v79, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v79, v107, v79, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
s_mov_b32 s66, 64                                 // coordOffset0 d0=7 vc0=0
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v84, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v84, v107, v84, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
s_mov_b32 s66, 80                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v89, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v107, v89, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v95, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v95, v107, v95, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v98, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v98, v107, v98, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v101, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v101, v107, v101, s[70:71]           // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v104, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v104, v107, v104, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+0], v[vgprValuScalAlpha_M+0], v[vgprValuC+0]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+0]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v0, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+0], v[vgprValuC+0] // check Nan
v_bfe_u32 v70, v[vgprValuC+0], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+0], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+0], v70, v72, s[66:67]
v_lshrrev_b32 v0, 16, v[vgprValuC+0]               // convert C to bf16
buffer_store_short v0, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+4], v[vgprValuScalAlpha_M+1], v[vgprValuC+4]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+4]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v4, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+4], v[vgprValuC+4] // check Nan
v_bfe_u32 v70, v[vgprValuC+4], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+4], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+4], v70, v72, s[66:67]
v_lshrrev_b32 v4, 16, v[vgprValuC+4]               // convert C to bf16
buffer_store_short v4, v79, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+8], v[vgprValuScalAlpha_M+2], v[vgprValuC+8]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+8]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v8, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+8], v[vgprValuC+8] // check Nan
v_bfe_u32 v70, v[vgprValuC+8], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+8], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+8], v70, v72, s[66:67]
v_lshrrev_b32 v8, 16, v[vgprValuC+8]               // convert C to bf16
buffer_store_short v8, v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+12], v[vgprValuScalAlpha_M+3], v[vgprValuC+12]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+12]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v12, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+12], v[vgprValuC+12] // check Nan
v_bfe_u32 v70, v[vgprValuC+12], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+12], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+12], v70, v72, s[66:67]
v_lshrrev_b32 v12, 16, v[vgprValuC+12]             // convert C to bf16
buffer_store_short v12, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+1], v[vgprValuScalAlpha_M+0], v[vgprValuC+1]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+1]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v1, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+1], v[vgprValuC+1] // check Nan
v_bfe_u32 v70, v[vgprValuC+1], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+1], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+1], v70, v72, s[66:67]
v_lshrrev_b32 v1, 16, v[vgprValuC+1]               // convert C to bf16
buffer_store_short v1, v95, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+5], v[vgprValuScalAlpha_M+1], v[vgprValuC+5]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+5]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v5, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+5], v[vgprValuC+5] // check Nan
v_bfe_u32 v70, v[vgprValuC+5], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+5], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+5], v70, v72, s[66:67]
v_lshrrev_b32 v5, 16, v[vgprValuC+5]               // convert C to bf16
buffer_store_short v5, v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+9], v[vgprValuScalAlpha_M+2], v[vgprValuC+9]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+9]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v9, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+9], v[vgprValuC+9] // check Nan
v_bfe_u32 v70, v[vgprValuC+9], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+9], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+9], v70, v72, s[66:67]
v_lshrrev_b32 v9, 16, v[vgprValuC+9]               // convert C to bf16
buffer_store_short v9, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+13], v[vgprValuScalAlpha_M+3], v[vgprValuC+13]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+13]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v13, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+13], v[vgprValuC+13] // check Nan
v_bfe_u32 v70, v[vgprValuC+13], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+13], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+13], v70, v72, s[66:67]
v_lshrrev_b32 v13, 16, v[vgprValuC+13]             // convert C to bf16
buffer_store_short v13, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v74, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v79, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v79, v107, v79, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v84, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v84, v107, v84, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v89, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v107, v89, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v95, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v95, v107, v95, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v98, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v98, v107, v98, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v101, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v101, v107, v101, s[70:71]           // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v104, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v104, v107, v104, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+2], v[vgprValuScalAlpha_M+0], v[vgprValuC+2]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+2]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v2, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+2], v[vgprValuC+2] // check Nan
v_bfe_u32 v70, v[vgprValuC+2], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+2], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+2], v70, v72, s[66:67]
v_lshrrev_b32 v2, 16, v[vgprValuC+2]               // convert C to bf16
buffer_store_short v2, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+6], v[vgprValuScalAlpha_M+1], v[vgprValuC+6]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+6]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v6, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+6], v[vgprValuC+6] // check Nan
v_bfe_u32 v70, v[vgprValuC+6], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+6], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+6], v70, v72, s[66:67]
v_lshrrev_b32 v6, 16, v[vgprValuC+6]               // convert C to bf16
buffer_store_short v6, v79, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+10], v[vgprValuScalAlpha_M+2], v[vgprValuC+10]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+10]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v10, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+10], v[vgprValuC+10] // check Nan
v_bfe_u32 v70, v[vgprValuC+10], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+10], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+10], v70, v72, s[66:67]
v_lshrrev_b32 v10, 16, v[vgprValuC+10]             // convert C to bf16
buffer_store_short v10, v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+14], v[vgprValuScalAlpha_M+3], v[vgprValuC+14]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+14]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v14, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+14], v[vgprValuC+14] // check Nan
v_bfe_u32 v70, v[vgprValuC+14], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+14], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+14], v70, v72, s[66:67]
v_lshrrev_b32 v14, 16, v[vgprValuC+14]             // convert C to bf16
buffer_store_short v14, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+3], v[vgprValuScalAlpha_M+0], v[vgprValuC+3]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+3]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v3, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+3], v[vgprValuC+3] // check Nan
v_bfe_u32 v70, v[vgprValuC+3], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+3], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+3], v70, v72, s[66:67]
v_lshrrev_b32 v3, 16, v[vgprValuC+3]               // convert C to bf16
buffer_store_short v3, v95, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+7], v[vgprValuScalAlpha_M+1], v[vgprValuC+7]      // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+7]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v7, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+7], v[vgprValuC+7] // check Nan
v_bfe_u32 v70, v[vgprValuC+7], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+7], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+7], v70, v72, s[66:67]
v_lshrrev_b32 v7, 16, v[vgprValuC+7]               // convert C to bf16
buffer_store_short v7, v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+11], v[vgprValuScalAlpha_M+2], v[vgprValuC+11]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+11]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v11, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+11], v[vgprValuC+11] // check Nan
v_bfe_u32 v70, v[vgprValuC+11], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+11], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+11], v70, v72, s[66:67]
v_lshrrev_b32 v11, 16, v[vgprValuC+11]             // convert C to bf16
buffer_store_short v11, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+15], v[vgprValuScalAlpha_M+3], v[vgprValuC+15]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+15]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v15, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+15], v[vgprValuC+15] // check Nan
v_bfe_u32 v70, v[vgprValuC+15], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+15], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+15], v70, v72, s[66:67]
v_lshrrev_b32 v15, 16, v[vgprValuC+15]             // convert C to bf16
buffer_store_short v15, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v74, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v79, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v79, v107, v79, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v84, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v84, v107, v84, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v89, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v107, v89, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v95, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v95, v107, v95, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v98, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v98, v107, v98, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v101, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v101, v107, v101, s[70:71]           // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v104, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v104, v107, v104, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+16], v[vgprValuScalAlpha_M+0], v[vgprValuC+16]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+16]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v16, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+16], v[vgprValuC+16] // check Nan
v_bfe_u32 v70, v[vgprValuC+16], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+16], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+16], v70, v72, s[66:67]
v_lshrrev_b32 v16, 16, v[vgprValuC+16]             // convert C to bf16
buffer_store_short v16, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+20], v[vgprValuScalAlpha_M+1], v[vgprValuC+20]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+20]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v20, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+20], v[vgprValuC+20] // check Nan
v_bfe_u32 v70, v[vgprValuC+20], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+20], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+20], v70, v72, s[66:67]
v_lshrrev_b32 v20, 16, v[vgprValuC+20]             // convert C to bf16
buffer_store_short v20, v79, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+24], v[vgprValuScalAlpha_M+2], v[vgprValuC+24]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+24]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v24, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+24], v[vgprValuC+24] // check Nan
v_bfe_u32 v70, v[vgprValuC+24], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+24], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+24], v70, v72, s[66:67]
v_lshrrev_b32 v24, 16, v[vgprValuC+24]             // convert C to bf16
buffer_store_short v24, v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+28], v[vgprValuScalAlpha_M+3], v[vgprValuC+28]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+28]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v28, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+28], v[vgprValuC+28] // check Nan
v_bfe_u32 v70, v[vgprValuC+28], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+28], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+28], v70, v72, s[66:67]
v_lshrrev_b32 v28, 16, v[vgprValuC+28]             // convert C to bf16
buffer_store_short v28, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+17], v[vgprValuScalAlpha_M+0], v[vgprValuC+17]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+17]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v17, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+17], v[vgprValuC+17] // check Nan
v_bfe_u32 v70, v[vgprValuC+17], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+17], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+17], v70, v72, s[66:67]
v_lshrrev_b32 v17, 16, v[vgprValuC+17]             // convert C to bf16
buffer_store_short v17, v95, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+21], v[vgprValuScalAlpha_M+1], v[vgprValuC+21]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+21]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v21, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+21], v[vgprValuC+21] // check Nan
v_bfe_u32 v70, v[vgprValuC+21], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+21], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+21], v70, v72, s[66:67]
v_lshrrev_b32 v21, 16, v[vgprValuC+21]             // convert C to bf16
buffer_store_short v21, v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+25], v[vgprValuScalAlpha_M+2], v[vgprValuC+25]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+25]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v25, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+25], v[vgprValuC+25] // check Nan
v_bfe_u32 v70, v[vgprValuC+25], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+25], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+25], v70, v72, s[66:67]
v_lshrrev_b32 v25, 16, v[vgprValuC+25]             // convert C to bf16
buffer_store_short v25, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+29], v[vgprValuScalAlpha_M+3], v[vgprValuC+29]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+29]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v29, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+29], v[vgprValuC+29] // check Nan
v_bfe_u32 v70, v[vgprValuC+29], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+29], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+29], v70, v72, s[66:67]
v_lshrrev_b32 v29, 16, v[vgprValuC+29]             // convert C to bf16
buffer_store_short v29, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v74, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v79, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v79, v107, v79, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v84, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v84, v107, v84, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
s_mov_b32 s66, 80                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v89, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v107, v89, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v95, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v95, v107, v95, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v98, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v98, v107, v98, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v101, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v101, v107, v101, s[70:71]           // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
s_mov_b32 s66, 80                                 
v_add_co_u32 v68, vcc, v64, s66                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v104, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v104, v107, v104, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+18], v[vgprValuScalAlpha_M+0], v[vgprValuC+18]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+18]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v18, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+18], v[vgprValuC+18] // check Nan
v_bfe_u32 v70, v[vgprValuC+18], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+18], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+18], v70, v72, s[66:67]
v_lshrrev_b32 v18, 16, v[vgprValuC+18]             // convert C to bf16
buffer_store_short v18, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+22], v[vgprValuScalAlpha_M+1], v[vgprValuC+22]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+22]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v22, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+22], v[vgprValuC+22] // check Nan
v_bfe_u32 v70, v[vgprValuC+22], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+22], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+22], v70, v72, s[66:67]
v_lshrrev_b32 v22, 16, v[vgprValuC+22]             // convert C to bf16
buffer_store_short v22, v79, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+26], v[vgprValuScalAlpha_M+2], v[vgprValuC+26]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+26]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v26, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+26], v[vgprValuC+26] // check Nan
v_bfe_u32 v70, v[vgprValuC+26], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+26], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+26], v70, v72, s[66:67]
v_lshrrev_b32 v26, 16, v[vgprValuC+26]             // convert C to bf16
buffer_store_short v26, v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+30], v[vgprValuScalAlpha_M+3], v[vgprValuC+30]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+30]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v30, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+30], v[vgprValuC+30] // check Nan
v_bfe_u32 v70, v[vgprValuC+30], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+30], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+30], v70, v72, s[66:67]
v_lshrrev_b32 v30, 16, v[vgprValuC+30]             // convert C to bf16
buffer_store_short v30, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+19], v[vgprValuScalAlpha_M+0], v[vgprValuC+19]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+19]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v19, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+19], v[vgprValuC+19] // check Nan
v_bfe_u32 v70, v[vgprValuC+19], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+19], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+19], v70, v72, s[66:67]
v_lshrrev_b32 v19, 16, v[vgprValuC+19]             // convert C to bf16
buffer_store_short v19, v95, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+23], v[vgprValuScalAlpha_M+1], v[vgprValuC+23]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+23]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v23, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+23], v[vgprValuC+23] // check Nan
v_bfe_u32 v70, v[vgprValuC+23], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+23], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+23], v70, v72, s[66:67]
v_lshrrev_b32 v23, 16, v[vgprValuC+23]             // convert C to bf16
buffer_store_short v23, v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+27], v[vgprValuScalAlpha_M+2], v[vgprValuC+27]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+27]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v27, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+27], v[vgprValuC+27] // check Nan
v_bfe_u32 v70, v[vgprValuC+27], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+27], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+27], v70, v72, s[66:67]
v_lshrrev_b32 v27, 16, v[vgprValuC+27]             // convert C to bf16
buffer_store_short v27, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+31], v[vgprValuScalAlpha_M+3], v[vgprValuC+31]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+31]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v31, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+31], v[vgprValuC+31] // check Nan
v_bfe_u32 v70, v[vgprValuC+31], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+31], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+31], v70, v72, s[66:67]
v_lshrrev_b32 v31, 16, v[vgprValuC+31]             // convert C to bf16
buffer_store_short v31, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #4 (d1,d0,vc1,vc0) = */
/*    (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (8,3,0,0:vw1); (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (9,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
s_mov_b32 s66, 100
v_add_co_u32 v65, vcc, v65, s66                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 100                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 100                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v74, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v79, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v79, v107, v79, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v84, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v84, v107, v84, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,3,0) */
s_mov_b32 s66, 80                               
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v89, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v107, v89, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v95, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v95, v107, v95, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v98, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v98, v107, v98, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v101, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v101, v107, v101, s[70:71]           // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,3,0) */
s_mov_b32 s66, 80                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v104, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v104, v107, v104, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (8, 3, 0, 0), (9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (9, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+32], v[vgprValuScalAlpha_M+0], v[vgprValuC+32]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+32]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v32, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+32], v[vgprValuC+32] // check Nan
v_bfe_u32 v70, v[vgprValuC+32], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+32], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+32], v70, v72, s[66:67]
v_lshrrev_b32 v32, 16, v[vgprValuC+32]             // convert C to bf16
buffer_store_short v32, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+36], v[vgprValuScalAlpha_M+1], v[vgprValuC+36]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+36]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v36, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+36], v[vgprValuC+36] // check Nan
v_bfe_u32 v70, v[vgprValuC+36], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+36], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+36], v70, v72, s[66:67]
v_lshrrev_b32 v36, 16, v[vgprValuC+36]             // convert C to bf16
buffer_store_short v36, v79, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+40], v[vgprValuScalAlpha_M+2], v[vgprValuC+40]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+40]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v40, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+40], v[vgprValuC+40] // check Nan
v_bfe_u32 v70, v[vgprValuC+40], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+40], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+40], v70, v72, s[66:67]
v_lshrrev_b32 v40, 16, v[vgprValuC+40]             // convert C to bf16
buffer_store_short v40, v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+44], v[vgprValuScalAlpha_M+3], v[vgprValuC+44]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+44]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v44, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+44], v[vgprValuC+44] // check Nan
v_bfe_u32 v70, v[vgprValuC+44], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+44], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+44], v70, v72, s[66:67]
v_lshrrev_b32 v44, 16, v[vgprValuC+44]             // convert C to bf16
buffer_store_short v44, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+33], v[vgprValuScalAlpha_M+0], v[vgprValuC+33]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+33]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v33, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+33], v[vgprValuC+33] // check Nan
v_bfe_u32 v70, v[vgprValuC+33], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+33], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+33], v70, v72, s[66:67]
v_lshrrev_b32 v33, 16, v[vgprValuC+33]             // convert C to bf16
buffer_store_short v33, v95, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+37], v[vgprValuScalAlpha_M+1], v[vgprValuC+37]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+37]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v37, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+37], v[vgprValuC+37] // check Nan
v_bfe_u32 v70, v[vgprValuC+37], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+37], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+37], v70, v72, s[66:67]
v_lshrrev_b32 v37, 16, v[vgprValuC+37]             // convert C to bf16
buffer_store_short v37, v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+41], v[vgprValuScalAlpha_M+2], v[vgprValuC+41]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+41]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v41, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+41], v[vgprValuC+41] // check Nan
v_bfe_u32 v70, v[vgprValuC+41], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+41], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+41], v70, v72, s[66:67]
v_lshrrev_b32 v41, 16, v[vgprValuC+41]             // convert C to bf16
buffer_store_short v41, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+45], v[vgprValuScalAlpha_M+3], v[vgprValuC+45]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+45]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v45, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+45], v[vgprValuC+45] // check Nan
v_bfe_u32 v70, v[vgprValuC+45], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+45], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+45], v70, v72, s[66:67]
v_lshrrev_b32 v45, 16, v[vgprValuC+45]             // convert C to bf16
buffer_store_short v45, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #5 (d1,d0,vc1,vc0) = */
/*    (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (10,3,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (11,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v74, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v79, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v79, v107, v79, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v84, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v84, v107, v84, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,3,0) */
s_mov_b32 s66, 80                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v89, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v107, v89, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v95, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v95, v107, v95, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v98, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v98, v107, v98, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v101, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v101, v107, v101, s[70:71]           // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,3,0) */
s_mov_b32 s66,80                               
v_add_co_u32 v68, vcc, v64, s66                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v104, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v104, v107, v104, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (10, 3, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (11, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+34], v[vgprValuScalAlpha_M+0], v[vgprValuC+34]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+34]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v34, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+34], v[vgprValuC+34] // check Nan
v_bfe_u32 v70, v[vgprValuC+34], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+34], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+34], v70, v72, s[66:67]
v_lshrrev_b32 v34, 16, v[vgprValuC+34]             // convert C to bf16
buffer_store_short v34, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+38], v[vgprValuScalAlpha_M+1], v[vgprValuC+38]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+38]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v38, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+38], v[vgprValuC+38] // check Nan
v_bfe_u32 v70, v[vgprValuC+38], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+38], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+38], v70, v72, s[66:67]
v_lshrrev_b32 v38, 16, v[vgprValuC+38]             // convert C to bf16
buffer_store_short v38, v79, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+42], v[vgprValuScalAlpha_M+2], v[vgprValuC+42]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+42]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v42, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+42], v[vgprValuC+42] // check Nan
v_bfe_u32 v70, v[vgprValuC+42], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+42], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+42], v70, v72, s[66:67]
v_lshrrev_b32 v42, 16, v[vgprValuC+42]             // convert C to bf16
buffer_store_short v42, v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+46], v[vgprValuScalAlpha_M+3], v[vgprValuC+46]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+46]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v46, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+46], v[vgprValuC+46] // check Nan
v_bfe_u32 v70, v[vgprValuC+46], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+46], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+46], v70, v72, s[66:67]
v_lshrrev_b32 v46, 16, v[vgprValuC+46]             // convert C to bf16
buffer_store_short v46, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+35], v[vgprValuScalAlpha_M+0], v[vgprValuC+35]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+35]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v35, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+35], v[vgprValuC+35] // check Nan
v_bfe_u32 v70, v[vgprValuC+35], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+35], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+35], v70, v72, s[66:67]
v_lshrrev_b32 v35, 16, v[vgprValuC+35]             // convert C to bf16
buffer_store_short v35, v95, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+39], v[vgprValuScalAlpha_M+1], v[vgprValuC+39]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+39]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v39, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+39], v[vgprValuC+39] // check Nan
v_bfe_u32 v70, v[vgprValuC+39], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+39], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+39], v70, v72, s[66:67]
v_lshrrev_b32 v39, 16, v[vgprValuC+39]             // convert C to bf16
buffer_store_short v39, v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+43], v[vgprValuScalAlpha_M+2], v[vgprValuC+43]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+43]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v43, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+43], v[vgprValuC+43] // check Nan
v_bfe_u32 v70, v[vgprValuC+43], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+43], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+43], v70, v72, s[66:67]
v_lshrrev_b32 v43, 16, v[vgprValuC+43]             // convert C to bf16
buffer_store_short v43, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+47], v[vgprValuScalAlpha_M+3], v[vgprValuC+47]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+47]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v47, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+47], v[vgprValuC+47] // check Nan
v_bfe_u32 v70, v[vgprValuC+47], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+47], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+47], v70, v72, s[66:67]
v_lshrrev_b32 v47, 16, v[vgprValuC+47]             // convert C to bf16
buffer_store_short v47, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #6 (d1,d0,vc1,vc0) = */
/*    (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (12,3,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (13,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v74, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v79, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v79, v107, v79, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v84, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v84, v107, v84, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v89, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v107, v89, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v95, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v95, v107, v95, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v98, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v98, v107, v98, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v101, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v101, v107, v101, s[70:71]           // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v104, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v104, v107, v104, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (12, 3, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (13, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+48], v[vgprValuScalAlpha_M+0], v[vgprValuC+48]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+48]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v48, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+48], v[vgprValuC+48] // check Nan
v_bfe_u32 v70, v[vgprValuC+48], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+48], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+48], v70, v72, s[66:67]
v_lshrrev_b32 v48, 16, v[vgprValuC+48]             // convert C to bf16
buffer_store_short v48, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+52], v[vgprValuScalAlpha_M+1], v[vgprValuC+52]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+52]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v52, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+52], v[vgprValuC+52] // check Nan
v_bfe_u32 v70, v[vgprValuC+52], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+52], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+52], v70, v72, s[66:67]
v_lshrrev_b32 v52, 16, v[vgprValuC+52]             // convert C to bf16
buffer_store_short v52, v79, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+56], v[vgprValuScalAlpha_M+2], v[vgprValuC+56]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+56]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v56, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+56], v[vgprValuC+56] // check Nan
v_bfe_u32 v70, v[vgprValuC+56], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+56], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+56], v70, v72, s[66:67]
v_lshrrev_b32 v56, 16, v[vgprValuC+56]             // convert C to bf16
buffer_store_short v56, v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+60], v[vgprValuScalAlpha_M+3], v[vgprValuC+60]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+60]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v60, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+60], v[vgprValuC+60] // check Nan
v_bfe_u32 v70, v[vgprValuC+60], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+60], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+60], v70, v72, s[66:67]
v_lshrrev_b32 v60, 16, v[vgprValuC+60]             // convert C to bf16
buffer_store_short v60, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+49], v[vgprValuScalAlpha_M+0], v[vgprValuC+49]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+49]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v49, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+49], v[vgprValuC+49] // check Nan
v_bfe_u32 v70, v[vgprValuC+49], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+49], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+49], v70, v72, s[66:67]
v_lshrrev_b32 v49, 16, v[vgprValuC+49]             // convert C to bf16
buffer_store_short v49, v95, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+53], v[vgprValuScalAlpha_M+1], v[vgprValuC+53]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+53]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v53, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+53], v[vgprValuC+53] // check Nan
v_bfe_u32 v70, v[vgprValuC+53], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+53], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+53], v70, v72, s[66:67]
v_lshrrev_b32 v53, 16, v[vgprValuC+53]             // convert C to bf16
buffer_store_short v53, v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+57], v[vgprValuScalAlpha_M+2], v[vgprValuC+57]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+57]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v57, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+57], v[vgprValuC+57] // check Nan
v_bfe_u32 v70, v[vgprValuC+57], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+57], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+57], v70, v72, s[66:67]
v_lshrrev_b32 v57, 16, v[vgprValuC+57]             // convert C to bf16
buffer_store_short v57, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+61], v[vgprValuScalAlpha_M+3], v[vgprValuC+61]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+61]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v61, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+61], v[vgprValuC+61] // check Nan
v_bfe_u32 v70, v[vgprValuC+61], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+61], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+61], v70, v72, s[66:67]
v_lshrrev_b32 v61, 16, v[vgprValuC+61]             // convert C to bf16
buffer_store_short v61, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #7 (d1,d0,vc1,vc0) = */
/*    (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (14,3,0,0:vw1); (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (15,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v74, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v79, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v79, v107, v79, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v84, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v84, v107, v84, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,3,0) */
s_mov_b32 s66, 80                               
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v89, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v89, v107, v89, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v95, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v95, v107, v95, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v98, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v98, v107, v98, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v101, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v101, v107, v101, s[70:71]           // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,3,0) */
s_mov_b32 s66, 80                               
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1

v_add_lshl_u32 v104, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v104, v107, v104, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (14, 3, 0, 0), (15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (15, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+50], v[vgprValuScalAlpha_M+0], v[vgprValuC+50]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+50]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v50, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+50], v[vgprValuC+50] // check Nan
v_bfe_u32 v70, v[vgprValuC+50], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+50], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+50], v70, v72, s[66:67]
v_lshrrev_b32 v50, 16, v[vgprValuC+50]             // convert C to bf16
buffer_store_short v50, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+54], v[vgprValuScalAlpha_M+1], v[vgprValuC+54]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+54]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v54, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+54], v[vgprValuC+54] // check Nan
v_bfe_u32 v70, v[vgprValuC+54], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+54], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+54], v70, v72, s[66:67]
v_lshrrev_b32 v54, 16, v[vgprValuC+54]             // convert C to bf16
buffer_store_short v54, v79, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+58], v[vgprValuScalAlpha_M+2], v[vgprValuC+58]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+58]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v58, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+58], v[vgprValuC+58] // check Nan
v_bfe_u32 v70, v[vgprValuC+58], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+58], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+58], v70, v72, s[66:67]
v_lshrrev_b32 v58, 16, v[vgprValuC+58]             // convert C to bf16
buffer_store_short v58, v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+62], v[vgprValuScalAlpha_M+3], v[vgprValuC+62]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+62]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v62, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+62], v[vgprValuC+62] // check Nan
v_bfe_u32 v70, v[vgprValuC+62], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+62], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+62], v70, v72, s[66:67]
v_lshrrev_b32 v62, 16, v[vgprValuC+62]             // convert C to bf16
buffer_store_short v62, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+51], v[vgprValuScalAlpha_M+0], v[vgprValuC+51]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+51]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v51, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+51], v[vgprValuC+51] // check Nan
v_bfe_u32 v70, v[vgprValuC+51], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+51], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+51], v70, v72, s[66:67]
v_lshrrev_b32 v51, 16, v[vgprValuC+51]             // convert C to bf16
buffer_store_short v51, v95, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+55], v[vgprValuScalAlpha_M+1], v[vgprValuC+55]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+55]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v55, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+55], v[vgprValuC+55] // check Nan
v_bfe_u32 v70, v[vgprValuC+55], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+55], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+55], v70, v72, s[66:67]
v_lshrrev_b32 v55, 16, v[vgprValuC+55]             // convert C to bf16
buffer_store_short v55, v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+59], v[vgprValuScalAlpha_M+2], v[vgprValuC+59]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+59]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v59, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+59], v[vgprValuC+59] // check Nan
v_bfe_u32 v70, v[vgprValuC+59], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+59], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+59], v70, v72, s[66:67]
v_lshrrev_b32 v59, 16, v[vgprValuC+59]             // convert C to bf16
buffer_store_short v59, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+63], v[vgprValuScalAlpha_M+3], v[vgprValuC+63]    // *= scaleAlphaVecVMul
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+63]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v63, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+63], v[vgprValuC+63] // check Nan
v_bfe_u32 v70, v[vgprValuC+63], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+63], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+63], v70, v72, s[66:67]
v_lshrrev_b32 v63, 16, v[vgprValuC+63]             // convert C to bf16
buffer_store_short v63, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_Beta_3:
s_and_b32 s66, 127, s[sgprSizeI]                   // s66 = s[sgprSizeI] % 128
s_add_u32 s67, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s67                // wg0 >= nwg0-1 ?
s_cselect_b32 s66, s66, 0                          // set rMT0
s_cmpk_gt_u32 s66, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_GW_B1_E1                      // jump if edges required
s_and_b32 s66, 255, s[sgprSizeJ]                   // s66 = s[sgprSizeJ] % 256
s_add_u32 s67, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s67                // wg1 >= nwg1-1
s_cselect_b32 s66, s66, 0                          // set rMT1
s_cmpk_gt_u32 s66, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_GW_B1_E1                      // jump if edges required
label_GW_B1_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=12 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v75, v66, v64, 0x1                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64
buffer_load_short_d16 v78, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_short_d16 v81, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
buffer_load_short_d16 v84, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
buffer_load_short_d16 v87, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s66, s[sgprStrideC1J], 8                 // scale StrideC *= numRows(4) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v90, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
buffer_load_short_d16 v91, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
buffer_load_short_d16 v92, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
buffer_load_short_d16 v93, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s66, s[sgprStrideC1J], 8                 // scale StrideC *= numRows(4) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v95, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
buffer_load_short_d16 v96, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
buffer_load_short_d16 v97, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
buffer_load_short_d16 v98, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C
v_add_lshl_u32 v74, v67, v64, 0x1                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

s_waitcnt  vmcnt(11)                    // vmcnt(14) = 16 - 1 (beta) - 1 (scaleAlphaVec) lgkmcnt(3) = 4 - 1 (bias) (interleaved)

v_mul_f32 v[vgprValuC+0], v[vgprValuScalAlpha_M+0], v[vgprValuC+0]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v78                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+0], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+0]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v0, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+0], v[vgprValuC+0] // check Nan
v_bfe_u32 v70, v[vgprValuC+0], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+0], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+0], v70, v72, s[66:67]
v_lshrrev_b32 v0, 16, v[vgprValuC+0]               // convert C to bf16
buffer_store_short v0, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt  vmcnt(10)                    // vmcnt(12) = 16 - 2 (beta) - 2 (scaleAlphaVec) lgkmcnt(2) = 4 - 2 (bias) (interleaved)

v_mul_f32 v[vgprValuC+4], v[vgprValuScalAlpha_M+1], v[vgprValuC+4]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v81                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+4], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+4]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v4, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+4], v[vgprValuC+4] // check Nan
v_bfe_u32 v70, v[vgprValuC+4], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+4], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+4], v70, v72, s[66:67]
v_lshrrev_b32 v4, 16, v[vgprValuC+4]               // convert C to bf16
buffer_store_short v4, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt  vmcnt(9)                    // vmcnt(10) = 16 - 3 (beta) - 3 (scaleAlphaVec) lgkmcnt(1) = 4 - 3 (bias) (interleaved)

v_mul_f32 v[vgprValuC+8], v[vgprValuScalAlpha_M+2], v[vgprValuC+8]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v84                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+8], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+8]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v8, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+8], v[vgprValuC+8] // check Nan
v_bfe_u32 v70, v[vgprValuC+8], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+8], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+8], v70, v72, s[66:67]
v_lshrrev_b32 v8, 16, v[vgprValuC+8]               // convert C to bf16
buffer_store_short v8, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt  vmcnt(8)                    // vmcnt(8) = 16 - 4 (beta) - 4 (scaleAlphaVec) lgkmcnt(0) = 4 - 4 (bias) (interleaved)

v_mul_f32 v[vgprValuC+12], v[vgprValuScalAlpha_M+3], v[vgprValuC+12]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v87                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+12], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+12]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v12, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+12], v[vgprValuC+12] // check Nan
v_bfe_u32 v70, v[vgprValuC+12], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+12], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+12], v70, v72, s[66:67]
v_lshrrev_b32 v12, 16, v[vgprValuC+12]             // convert C to bf16
buffer_store_short v12, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

s_waitcnt vmcnt(7)                                // vmcnt(7) = 16 - 5 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+1], v[vgprValuScalAlpha_M+0], v[vgprValuC+1]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v90                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+1], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+1]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v1, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+1], v[vgprValuC+1] // check Nan
v_bfe_u32 v70, v[vgprValuC+1], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+1], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+1], v70, v72, s[66:67]
v_lshrrev_b32 v1, 16, v[vgprValuC+1]               // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v1, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                // vmcnt(6) = 16 - 6 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+5], v[vgprValuScalAlpha_M+1], v[vgprValuC+5]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v91                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+5], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+5]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v5, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+5], v[vgprValuC+5] // check Nan
v_bfe_u32 v70, v[vgprValuC+5], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+5], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+5], v70, v72, s[66:67]
v_lshrrev_b32 v5, 16, v[vgprValuC+5]               // convert C to bf16
buffer_store_short v5, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(5)                                // vmcnt(5) = 16 - 7 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+9], v[vgprValuScalAlpha_M+2], v[vgprValuC+9]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v92                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+9], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+9]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v9, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+9], v[vgprValuC+9] // check Nan
v_bfe_u32 v70, v[vgprValuC+9], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+9], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+9], v70, v72, s[66:67]
v_lshrrev_b32 v9, 16, v[vgprValuC+9]               // convert C to bf16
buffer_store_short v9, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(4)                                // vmcnt(4) = 16 - 8 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+13], v[vgprValuScalAlpha_M+3], v[vgprValuC+13]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v93                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+13], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+13]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v13, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+13], v[vgprValuC+13] // check Nan
v_bfe_u32 v70, v[vgprValuC+13], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+13], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+13], v70, v72, s[66:67]
v_lshrrev_b32 v13, 16, v[vgprValuC+13]             // convert C to bf16
buffer_store_short v13, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

s_waitcnt vmcnt(3)                                // vmcnt(3) = 16 - 9 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+2], v[vgprValuScalAlpha_M+0], v[vgprValuC+2]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v95                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+2], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+2]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v2, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+2], v[vgprValuC+2] // check Nan
v_bfe_u32 v70, v[vgprValuC+2], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+2], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+2], v70, v72, s[66:67]
v_lshrrev_b32 v2, 16, v[vgprValuC+2]               // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v2, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                // vmcnt(2) = 16 - 10 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+6], v[vgprValuScalAlpha_M+1], v[vgprValuC+6]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+6], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+6]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v6, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+6], v[vgprValuC+6] // check Nan
v_bfe_u32 v70, v[vgprValuC+6], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+6], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+6], v70, v72, s[66:67]
v_lshrrev_b32 v6, 16, v[vgprValuC+6]               // convert C to bf16
buffer_store_short v6, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(1)                                // vmcnt(1) = 16 - 11 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+10], v[vgprValuScalAlpha_M+2], v[vgprValuC+10]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v97                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+10], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+10]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v10, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+10], v[vgprValuC+10] // check Nan
v_bfe_u32 v70, v[vgprValuC+10], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+10], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+10], v70, v72, s[66:67]
v_lshrrev_b32 v10, 16, v[vgprValuC+10]             // convert C to bf16
buffer_store_short v10, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(0)                                // vmcnt(0) = 16 - 12 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+14], v[vgprValuScalAlpha_M+3], v[vgprValuC+14]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v98                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+14], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+14]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v14, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+14], v[vgprValuC+14] // check Nan
v_bfe_u32 v70, v[vgprValuC+14], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+14], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+14], v70, v72, s[66:67]
v_lshrrev_b32 v14, 16, v[vgprValuC+14]             // convert C to bf16
buffer_store_short v14, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Beta Batch #1 (d1,d0,vc1,vc0) = */
/*    (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s66, s[sgprStrideC1J], 8                 // scale StrideC *= numRows(4) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v78, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
buffer_load_short_d16 v81, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
buffer_load_short_d16 v84, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
buffer_load_short_d16 v87, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_mul_i32 s66, s[sgprStrideC1J], 8                 // scale StrideC *= numRows(4) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v90, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
buffer_load_short_d16 v91, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
buffer_load_short_d16 v92, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
buffer_load_short_d16 v93, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
s_mul_i32 s66, s[sgprStrideC1J], 8                 // scale StrideC *= numRows(4) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v95, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
buffer_load_short_d16 v96, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
buffer_load_short_d16 v97, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
buffer_load_short_d16 v98, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C

/* rC *= alpha batchElements=[(3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

s_waitcnt  vmcnt(11)                    // vmcnt(14) = 16 - 1 (beta) - 1 (scaleAlphaVec) lgkmcnt(3) = 4 - 1 (bias) (interleaved)

v_mul_f32 v[vgprValuC+3], v[vgprValuScalAlpha_M+0], v[vgprValuC+3]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v78                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+3], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+3]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v3, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+3], v[vgprValuC+3] // check Nan
v_bfe_u32 v70, v[vgprValuC+3], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+3], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+3], v70, v72, s[66:67]
v_lshrrev_b32 v3, 16, v[vgprValuC+3]               // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v3, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt  vmcnt(10)                    // vmcnt(12) = 16 - 2 (beta) - 2 (scaleAlphaVec) lgkmcnt(2) = 4 - 2 (bias) (interleaved)

v_mul_f32 v[vgprValuC+7], v[vgprValuScalAlpha_M+1], v[vgprValuC+7]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v81                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+7], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+7]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v7, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+7], v[vgprValuC+7] // check Nan
v_bfe_u32 v70, v[vgprValuC+7], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+7], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+7], v70, v72, s[66:67]
v_lshrrev_b32 v7, 16, v[vgprValuC+7]               // convert C to bf16
buffer_store_short v7, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt  vmcnt(9)                    // vmcnt(10) = 16 - 3 (beta) - 3 (scaleAlphaVec) lgkmcnt(1) = 4 - 3 (bias) (interleaved)

v_mul_f32 v[vgprValuC+11], v[vgprValuScalAlpha_M+2], v[vgprValuC+11]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v84                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+11], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+11]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v11, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+11], v[vgprValuC+11] // check Nan
v_bfe_u32 v70, v[vgprValuC+11], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+11], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+11], v70, v72, s[66:67]
v_lshrrev_b32 v11, 16, v[vgprValuC+11]             // convert C to bf16
buffer_store_short v11, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt  vmcnt(8)                    // vmcnt(8) = 16 - 4 (beta) - 4 (scaleAlphaVec) lgkmcnt(0) = 4 - 4 (bias) (interleaved)

v_mul_f32 v[vgprValuC+15], v[vgprValuScalAlpha_M+3], v[vgprValuC+15]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v87                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+15], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+15]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v15, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+15], v[vgprValuC+15] // check Nan
v_bfe_u32 v70, v[vgprValuC+15], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+15], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+15], v70, v72, s[66:67]
v_lshrrev_b32 v15, 16, v[vgprValuC+15]             // convert C to bf16
buffer_store_short v15, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

s_waitcnt vmcnt(7)                                // vmcnt(7) = 16 - 5 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+16], v[vgprValuScalAlpha_M+0], v[vgprValuC+16]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v90                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+16], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+16]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v16, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+16], v[vgprValuC+16] // check Nan
v_bfe_u32 v70, v[vgprValuC+16], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+16], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+16], v70, v72, s[66:67]
v_lshrrev_b32 v16, 16, v[vgprValuC+16]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v16, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                // vmcnt(5) = 16 - 7 (beta) - 4 (scaleAlphaVec) (interleaved)
v_mul_f32 v[vgprValuC+20], v[vgprValuScalAlpha_M+1], v[vgprValuC+20]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v91                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+20], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+20]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v20, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+20], v[vgprValuC+20] // check Nan
v_bfe_u32 v70, v[vgprValuC+20], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+20], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+20], v70, v72, s[66:67]
v_lshrrev_b32 v20, 16, v[vgprValuC+20]             // convert C to bf16
buffer_store_short v20, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(5)                                // vmcnt(5) = 16 - 7 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+24], v[vgprValuScalAlpha_M+2], v[vgprValuC+24]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v92                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+24], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+24]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v24, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+24], v[vgprValuC+24] // check Nan
v_bfe_u32 v70, v[vgprValuC+24], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+24], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+24], v70, v72, s[66:67]
v_lshrrev_b32 v24, 16, v[vgprValuC+24]             // convert C to bf16
buffer_store_short v24, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(4)                                // vmcnt(4) = 16 - 8 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+28], v[vgprValuScalAlpha_M+3], v[vgprValuC+28]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v93                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+28], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+28]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v28, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+28], v[vgprValuC+28] // check Nan
v_bfe_u32 v70, v[vgprValuC+28], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+28], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+28], v70, v72, s[66:67]
v_lshrrev_b32 v28, 16, v[vgprValuC+28]             // convert C to bf16
buffer_store_short v28, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

s_waitcnt vmcnt(3)                                // vmcnt(3) = 16 - 9 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+17], v[vgprValuScalAlpha_M+0], v[vgprValuC+17]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v95                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+17], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+17]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v17, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+17], v[vgprValuC+17] // check Nan
v_bfe_u32 v70, v[vgprValuC+17], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+17], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+17], v70, v72, s[66:67]
v_lshrrev_b32 v17, 16, v[vgprValuC+17]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v17, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                // vmcnt(2) = 16 - 10 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+21], v[vgprValuScalAlpha_M+1], v[vgprValuC+21]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+21], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+21]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v21, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+21], v[vgprValuC+21] // check Nan
v_bfe_u32 v70, v[vgprValuC+21], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+21], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+21], v70, v72, s[66:67]
v_lshrrev_b32 v21, 16, v[vgprValuC+21]             // convert C to bf16
buffer_store_short v21, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(1)                                // vmcnt(1) = 16 - 11 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+25], v[vgprValuScalAlpha_M+2], v[vgprValuC+25]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v97                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+25], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+25]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v25, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+25], v[vgprValuC+25] // check Nan
v_bfe_u32 v70, v[vgprValuC+25], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+25], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+25], v70, v72, s[66:67]
v_lshrrev_b32 v25, 16, v[vgprValuC+25]             // convert C to bf16
buffer_store_short v25, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(0)                                // vmcnt(0) = 16 - 12 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+29], v[vgprValuScalAlpha_M+3], v[vgprValuC+29]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v98                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+29], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+29]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v29, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+29], v[vgprValuC+29] // check Nan
v_bfe_u32 v70, v[vgprValuC+29], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+29], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+29], v70, v72, s[66:67]
v_lshrrev_b32 v29, 16, v[vgprValuC+29]             // convert C to bf16
buffer_store_short v29, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Beta Batch #2 (d1,d0,vc1,vc0) = */
/*    (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (8,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
s_mul_i32 s66, s[sgprStrideC1J], 8                 // scale StrideC *= numRows(4) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v78, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
buffer_load_short_d16 v81, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
buffer_load_short_d16 v84, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
buffer_load_short_d16 v87, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
s_mul_i32 s66, s[sgprStrideC1J], 8                 // scale StrideC *= numRows(4) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v90, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
buffer_load_short_d16 v91, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
buffer_load_short_d16 v92, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
buffer_load_short_d16 v93, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
s_mul_i32 s66, s[sgprStrideC1J], 200                 // scale StrideC *= numRows(4) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v95, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
buffer_load_short_d16 v96, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
buffer_load_short_d16 v97, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128// load C
/* (d1,vc1,d0,vc0)=(8,0,3,0) */
buffer_load_short_d16 v98, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C

/* rC *= alpha batchElements=[(6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (8, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

s_waitcnt  vmcnt(11)                    // vmcnt(14) = 16 - 1 (beta) - 1 (scaleAlphaVec) lgkmcnt(3) = 4 - 1 (bias) (interleaved)

v_mul_f32 v[vgprValuC+18], v[vgprValuScalAlpha_M+0], v[vgprValuC+18]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v78                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+18], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+18]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v18, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+18], v[vgprValuC+18] // check Nan
v_bfe_u32 v70, v[vgprValuC+18], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+18], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+18], v70, v72, s[66:67]
v_lshrrev_b32 v18, 16, v[vgprValuC+18]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v18, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt  vmcnt(10)                    // vmcnt(12) = 16 - 2 (beta) - 2 (scaleAlphaVec) lgkmcnt(2) = 4 - 2 (bias) (interleaved)

v_mul_f32 v[vgprValuC+22], v[vgprValuScalAlpha_M+1], v[vgprValuC+22]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v81                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+22], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+22]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v22, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+22], v[vgprValuC+22] // check Nan
v_bfe_u32 v70, v[vgprValuC+22], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+22], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+22], v70, v72, s[66:67]
v_lshrrev_b32 v22, 16, v[vgprValuC+22]             // convert C to bf16
buffer_store_short v22, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt  vmcnt(9)                    // vmcnt(10) = 16 - 3 (beta) - 3 (scaleAlphaVec) lgkmcnt(1) = 4 - 3 (bias) (interleaved)

v_mul_f32 v[vgprValuC+26], v[vgprValuScalAlpha_M+2], v[vgprValuC+26]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v84                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+26], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+26]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v26, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+26], v[vgprValuC+26] // check Nan
v_bfe_u32 v70, v[vgprValuC+26], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+26], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+26], v70, v72, s[66:67]
v_lshrrev_b32 v26, 16, v[vgprValuC+26]             // convert C to bf16
buffer_store_short v26, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt  vmcnt(8)                    // vmcnt(8) = 16 - 4 (beta) - 4 (scaleAlphaVec) lgkmcnt(0) = 4 - 4 (bias) (interleaved)

v_mul_f32 v[vgprValuC+30], v[vgprValuScalAlpha_M+3], v[vgprValuC+30]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v87                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+30], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+30]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v30, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+30], v[vgprValuC+30] // check Nan
v_bfe_u32 v70, v[vgprValuC+30], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+30], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+30], v70, v72, s[66:67]
v_lshrrev_b32 v30, 16, v[vgprValuC+30]             // convert C to bf16
buffer_store_short v30, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

s_waitcnt vmcnt(7)                                // vmcnt(7) = 16 - 5 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+19], v[vgprValuScalAlpha_M+0], v[vgprValuC+19]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v90                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+19], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+19]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v19, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+19], v[vgprValuC+19] // check Nan
v_bfe_u32 v70, v[vgprValuC+19], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+19], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+19], v70, v72, s[66:67]
v_lshrrev_b32 v19, 16, v[vgprValuC+19]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v19, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                // vmcnt(6) = 16 - 6 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+23], v[vgprValuScalAlpha_M+1], v[vgprValuC+23]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v91                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+23], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+23]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v23, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+23], v[vgprValuC+23] // check Nan
v_bfe_u32 v70, v[vgprValuC+23], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+23], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+23], v70, v72, s[66:67]
v_lshrrev_b32 v23, 16, v[vgprValuC+23]             // convert C to bf16
buffer_store_short v23, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(5)                                // vmcnt(5) = 16 - 7 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+27], v[vgprValuScalAlpha_M+2], v[vgprValuC+27]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v92                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+27], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+27]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v27, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+27], v[vgprValuC+27] // check Nan
v_bfe_u32 v70, v[vgprValuC+27], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+27], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+27], v70, v72, s[66:67]
v_lshrrev_b32 v27, 16, v[vgprValuC+27]             // convert C to bf16
buffer_store_short v27, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(4)                                // vmcnt(4) = 16 - 8 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+31], v[vgprValuScalAlpha_M+3], v[vgprValuC+31]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v93                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+31], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+31]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v31, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+31], v[vgprValuC+31] // check Nan
v_bfe_u32 v70, v[vgprValuC+31], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+31], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+31], v70, v72, s[66:67]
v_lshrrev_b32 v31, 16, v[vgprValuC+31]             // convert C to bf16
buffer_store_short v31, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

s_waitcnt vmcnt(3)                                // vmcnt(3) = 16 - 9 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+32], v[vgprValuScalAlpha_M+0], v[vgprValuC+32]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v95                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+32], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+32]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v32, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+32], v[vgprValuC+32] // check Nan
v_bfe_u32 v70, v[vgprValuC+32], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+32], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+32], v70, v72, s[66:67]
v_lshrrev_b32 v32, 16, v[vgprValuC+32]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 200                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v32, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                // vmcnt(2) = 16 - 10 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+36], v[vgprValuScalAlpha_M+1], v[vgprValuC+36]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+36], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+36]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v36, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+36], v[vgprValuC+36] // check Nan
v_bfe_u32 v70, v[vgprValuC+36], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+36], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+36], v70, v72, s[66:67]
v_lshrrev_b32 v36, 16, v[vgprValuC+36]             // convert C to bf16
buffer_store_short v36, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(1)                                // vmcnt(1) = 16 - 11 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+40], v[vgprValuScalAlpha_M+2], v[vgprValuC+40]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v97                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+40], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+40]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v40, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+40], v[vgprValuC+40] // check Nan
v_bfe_u32 v70, v[vgprValuC+40], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+40], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+40], v70, v72, s[66:67]
v_lshrrev_b32 v40, 16, v[vgprValuC+40]             // convert C to bf16
buffer_store_short v40, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(0)                                // vmcnt(0) = 16 - 12 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+44], v[vgprValuScalAlpha_M+3], v[vgprValuC+44]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v98                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+44], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+44]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v44, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+44], v[vgprValuC+44] // check Nan
v_bfe_u32 v70, v[vgprValuC+44], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+44], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+44], v70, v72, s[66:67]
v_lshrrev_b32 v44, 16, v[vgprValuC+44]             // convert C to bf16
buffer_store_short v44, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Beta Batch #3 (d1,d0,vc1,vc0) = */
/*    (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (9,3,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1); (10,2,0,0:vw1); (10,3,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (11,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
s_mul_i32 s66, s[sgprStrideC1J], 8                 // scale StrideC *= numRows(4) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v78, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
buffer_load_short_d16 v81, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
buffer_load_short_d16 v84, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(9,0,3,0) */
buffer_load_short_d16 v87, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
s_mul_i32 s66, s[sgprStrideC1J], 8                 // scale StrideC *= numRows(4) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v90, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
buffer_load_short_d16 v91, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
buffer_load_short_d16 v92, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(10,0,3,0) */
buffer_load_short_d16 v93, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
s_mul_i32 s66, s[sgprStrideC1J], 8                 // scale StrideC *= numRows(4) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v95, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
buffer_load_short_d16 v96, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
buffer_load_short_d16 v97, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(11,0,3,0) */
buffer_load_short_d16 v98, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C

/* rC *= alpha batchElements=[(9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (9, 3, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0), (10, 2, 0, 0), (10, 3, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (11, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

s_waitcnt  vmcnt(11)                    // vmcnt(14) = 16 - 1 (beta) - 1 (scaleAlphaVec) lgkmcnt(3) = 4 - 1 (bias) (interleaved)

v_mul_f32 v[vgprValuC+33], v[vgprValuScalAlpha_M+0], v[vgprValuC+33]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v78                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+33], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+33]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v33, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+33], v[vgprValuC+33] // check Nan
v_bfe_u32 v70, v[vgprValuC+33], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+33], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+33], v70, v72, s[66:67]
v_lshrrev_b32 v33, 16, v[vgprValuC+33]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v33, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt  vmcnt(10)                    // vmcnt(12) = 16 - 2 (beta) - 2 (scaleAlphaVec) lgkmcnt(2) = 4 - 2 (bias) (interleaved)

v_mul_f32 v[vgprValuC+37], v[vgprValuScalAlpha_M+1], v[vgprValuC+37]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v81                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+37], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+37]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v37, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+37], v[vgprValuC+37] // check Nan
v_bfe_u32 v70, v[vgprValuC+37], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+37], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+37], v70, v72, s[66:67]
v_lshrrev_b32 v37, 16, v[vgprValuC+37]             // convert C to bf16
buffer_store_short v37, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt  vmcnt(9)                    // vmcnt(10) = 16 - 3 (beta) - 3 (scaleAlphaVec) lgkmcnt(1) = 4 - 3 (bias) (interleaved)

v_mul_f32 v[vgprValuC+41], v[vgprValuScalAlpha_M+2], v[vgprValuC+41]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v84                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+41], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+41]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v41, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+41], v[vgprValuC+41] // check Nan
v_bfe_u32 v70, v[vgprValuC+41], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+41], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+41], v70, v72, s[66:67]
v_lshrrev_b32 v41, 16, v[vgprValuC+41]             // convert C to bf16
buffer_store_short v41, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt  vmcnt(8)                    // vmcnt(8) = 16 - 4 (beta) - 4 (scaleAlphaVec) lgkmcnt(0) = 4 - 4 (bias) (interleaved)

v_mul_f32 v[vgprValuC+45], v[vgprValuScalAlpha_M+3], v[vgprValuC+45]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v87                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+45], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+45]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v45, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+45], v[vgprValuC+45] // check Nan
v_bfe_u32 v70, v[vgprValuC+45], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+45], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+45], v70, v72, s[66:67]
v_lshrrev_b32 v45, 16, v[vgprValuC+45]             // convert C to bf16
buffer_store_short v45, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

s_waitcnt vmcnt(7)                                // vmcnt(7) = 16 - 5 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+34], v[vgprValuScalAlpha_M+0], v[vgprValuC+34]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v90                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+34], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+34]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v34, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+34], v[vgprValuC+34] // check Nan
v_bfe_u32 v70, v[vgprValuC+34], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+34], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+34], v70, v72, s[66:67]
v_lshrrev_b32 v34, 16, v[vgprValuC+34]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v34, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                // vmcnt(6) = 16 - 6 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+38], v[vgprValuScalAlpha_M+1], v[vgprValuC+38]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v91                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+38], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+38]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v38, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+38], v[vgprValuC+38] // check Nan
v_bfe_u32 v70, v[vgprValuC+38], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+38], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+38], v70, v72, s[66:67]
v_lshrrev_b32 v38, 16, v[vgprValuC+38]             // convert C to bf16
buffer_store_short v38, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(5)                                // vmcnt(5) = 16 - 7 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+42], v[vgprValuScalAlpha_M+2], v[vgprValuC+42]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v92                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+42], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+42]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v42, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+42], v[vgprValuC+42] // check Nan
v_bfe_u32 v70, v[vgprValuC+42], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+42], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+42], v70, v72, s[66:67]
v_lshrrev_b32 v42, 16, v[vgprValuC+42]             // convert C to bf16
buffer_store_short v42, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(4)                                // vmcnt(4) = 16 - 8 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+46], v[vgprValuScalAlpha_M+3], v[vgprValuC+46]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v93                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+46], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+46]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v46, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+46], v[vgprValuC+46] // check Nan
v_bfe_u32 v70, v[vgprValuC+46], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+46], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+46], v70, v72, s[66:67]
v_lshrrev_b32 v46, 16, v[vgprValuC+46]             // convert C to bf16
buffer_store_short v46, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

s_waitcnt vmcnt(3)                                // vmcnt(3) = 16 - 9 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+35], v[vgprValuScalAlpha_M+0], v[vgprValuC+35]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v95                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+35], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+35]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v35, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+35], v[vgprValuC+35] // check Nan
v_bfe_u32 v70, v[vgprValuC+35], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+35], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+35], v70, v72, s[66:67]
v_lshrrev_b32 v35, 16, v[vgprValuC+35]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v35, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                // vmcnt(2) = 16 - 10 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+39], v[vgprValuScalAlpha_M+1], v[vgprValuC+39]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+39], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+39]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v39, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+39], v[vgprValuC+39] // check Nan
v_bfe_u32 v70, v[vgprValuC+39], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+39], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+39], v70, v72, s[66:67]
v_lshrrev_b32 v39, 16, v[vgprValuC+39]             // convert C to bf16
buffer_store_short v39, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(1)                                // vmcnt(1) = 16 - 11 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+43], v[vgprValuScalAlpha_M+2], v[vgprValuC+43]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v97                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+43], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+43]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v43, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+43], v[vgprValuC+43] // check Nan
v_bfe_u32 v70, v[vgprValuC+43], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+43], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+43], v70, v72, s[66:67]
v_lshrrev_b32 v43, 16, v[vgprValuC+43]             // convert C to bf16
buffer_store_short v43, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(0)                                // vmcnt(0) = 16 - 12 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+47], v[vgprValuScalAlpha_M+3], v[vgprValuC+47]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v98                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+47], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+47]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v47, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+47], v[vgprValuC+47] // check Nan
v_bfe_u32 v70, v[vgprValuC+47], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+47], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+47], v70, v72, s[66:67]
v_lshrrev_b32 v47, 16, v[vgprValuC+47]             // convert C to bf16
buffer_store_short v47, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Beta Batch #4 (d1,d0,vc1,vc0) = */
/*    (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (12,3,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1); (13,2,0,0:vw1); (13,3,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (14,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
s_mul_i32 s66, s[sgprStrideC1J], 8                 // scale StrideC *= numRows(4) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v78, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
buffer_load_short_d16 v81, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
buffer_load_short_d16 v84, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(12,0,3,) */
buffer_load_short_d16 v87, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
s_mul_i32 s66, s[sgprStrideC1J], 8                 // scale StrideC *= numRows(4) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v90, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
buffer_load_short_d16 v91, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
buffer_load_short_d16 v92, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(13,0,3,0) */
buffer_load_short_d16 v93, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
s_mul_i32 s66, s[sgprStrideC1J], 8                 // scale StrideC *= numRows(4) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v95, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
buffer_load_short_d16 v96, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
buffer_load_short_d16 v97, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(14,0,3,0) */
buffer_load_short_d16 v98, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C

/* rC *= alpha batchElements=[(12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (12, 3, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0), (13, 2, 0, 0), (13, 3, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (14, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

s_waitcnt  vmcnt(11)                    // vmcnt(14) = 16 - 1 (beta) - 1 (scaleAlphaVec) lgkmcnt(3) = 4 - 1 (bias) (interleaved)

v_mul_f32 v[vgprValuC+48], v[vgprValuScalAlpha_M+0], v[vgprValuC+48]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v78                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+48], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+48]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v48, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+48], v[vgprValuC+48] // check Nan
v_bfe_u32 v70, v[vgprValuC+48], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+48], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+48], v70, v72, s[66:67]
v_lshrrev_b32 v48, 16, v[vgprValuC+48]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v48, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt  vmcnt(10)                    // vmcnt(12) = 16 - 2 (beta) - 2 (scaleAlphaVec) lgkmcnt(2) = 4 - 2 (bias) (interleaved)

v_mul_f32 v[vgprValuC+52], v[vgprValuScalAlpha_M+1], v[vgprValuC+52]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v81                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+52], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+52]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v52, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+52], v[vgprValuC+52] // check Nan
v_bfe_u32 v70, v[vgprValuC+52], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+52], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+52], v70, v72, s[66:67]
v_lshrrev_b32 v52, 16, v[vgprValuC+52]             // convert C to bf16
buffer_store_short v52, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt  vmcnt(9)                    // vmcnt(10) = 16 - 3 (beta) - 3 (scaleAlphaVec) lgkmcnt(1) = 4 - 3 (bias) (interleaved)

v_mul_f32 v[vgprValuC+56], v[vgprValuScalAlpha_M+2], v[vgprValuC+56]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v84                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+56], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+56]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v56, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+56], v[vgprValuC+56] // check Nan
v_bfe_u32 v70, v[vgprValuC+56], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+56], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+56], v70, v72, s[66:67]
v_lshrrev_b32 v56, 16, v[vgprValuC+56]             // convert C to bf16
buffer_store_short v56, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt  vmcnt(8)                    // vmcnt(8) = 16 - 4 (beta) - 4 (scaleAlphaVec) lgkmcnt(0) = 4 - 4 (bias) (interleaved)

v_mul_f32 v[vgprValuC+60], v[vgprValuScalAlpha_M+3], v[vgprValuC+60]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v87                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+60], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+60]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v60, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+60], v[vgprValuC+60] // check Nan
v_bfe_u32 v70, v[vgprValuC+60], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+60], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+60], v70, v72, s[66:67]
v_lshrrev_b32 v60, 16, v[vgprValuC+60]             // convert C to bf16
buffer_store_short v60, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

s_waitcnt vmcnt(7)                                // vmcnt(7) = 16 - 5 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+49], v[vgprValuScalAlpha_M+0], v[vgprValuC+49]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v90                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+49], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+49]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v49, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+49], v[vgprValuC+49] // check Nan
v_bfe_u32 v70, v[vgprValuC+49], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+49], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+49], v70, v72, s[66:67]
v_lshrrev_b32 v49, 16, v[vgprValuC+49]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v49, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                // vmcnt(6) = 16 - 6 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+53], v[vgprValuScalAlpha_M+1], v[vgprValuC+53]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v91                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+53], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+53]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v53, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+53], v[vgprValuC+53] // check Nan
v_bfe_u32 v70, v[vgprValuC+53], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+53], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+53], v70, v72, s[66:67]
v_lshrrev_b32 v53, 16, v[vgprValuC+53]             // convert C to bf16
buffer_store_short v53, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(5)                                // vmcnt(5) = 16 - 7 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+57], v[vgprValuScalAlpha_M+2], v[vgprValuC+57]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v92                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+57], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+57]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v57, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+57], v[vgprValuC+57] // check Nan
v_bfe_u32 v70, v[vgprValuC+57], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+57], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+57], v70, v72, s[66:67]
v_lshrrev_b32 v57, 16, v[vgprValuC+57]             // convert C to bf16
buffer_store_short v57, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(4)                                // vmcnt(4) = 16 - 8 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+61], v[vgprValuScalAlpha_M+3], v[vgprValuC+61]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v93                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+61], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+61]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v61, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+61], v[vgprValuC+61] // check Nan
v_bfe_u32 v70, v[vgprValuC+61], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+61], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+61], v70, v72, s[66:67]
v_lshrrev_b32 v61, 16, v[vgprValuC+61]             // convert C to bf16
buffer_store_short v61, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D

s_waitcnt vmcnt(3)                                // vmcnt(3) = 16 - 9 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+50], v[vgprValuScalAlpha_M+0], v[vgprValuC+50]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v95                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+50], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+50]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v50, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+50], v[vgprValuC+50] // check Nan
v_bfe_u32 v70, v[vgprValuC+50], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+50], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+50], v70, v72, s[66:67]
v_lshrrev_b32 v50, 16, v[vgprValuC+50]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v50, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                // vmcnt(2) = 16 - 10 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+54], v[vgprValuScalAlpha_M+1], v[vgprValuC+54]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+54], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+54]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v54, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+54], v[vgprValuC+54] // check Nan
v_bfe_u32 v70, v[vgprValuC+54], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+54], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+54], v70, v72, s[66:67]
v_lshrrev_b32 v54, 16, v[vgprValuC+54]             // convert C to bf16
buffer_store_short v54, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(1)                                // vmcnt(1) = 16 - 11 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+58], v[vgprValuScalAlpha_M+2], v[vgprValuC+58]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v97                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+58], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+58]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v58, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+58], v[vgprValuC+58] // check Nan
v_bfe_u32 v70, v[vgprValuC+58], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+58], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+58], v70, v72, s[66:67]
v_lshrrev_b32 v58, 16, v[vgprValuC+58]             // convert C to bf16
buffer_store_short v58, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt vmcnt(0)                                // vmcnt(0) = 16 - 12 (beta) - 4 (scaleAlphaVec) (interleaved)

v_mul_f32 v[vgprValuC+62], v[vgprValuScalAlpha_M+3], v[vgprValuC+62]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v98                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+62], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+62]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v62, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+62], v[vgprValuC+62] // check Nan
v_bfe_u32 v70, v[vgprValuC+62], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+62], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+62], v70, v72, s[66:67]
v_lshrrev_b32 v62, 16, v[vgprValuC+62]             // convert C to bf16
buffer_store_short v62, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Beta Batch #5 (d1,d0,vc1,vc0) = */
/*    (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (15,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
s_mul_i32 s66, s[sgprStrideC1J], 8                 // scale StrideC *= numRows(4) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v78, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
buffer_load_short_d16 v81, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
buffer_load_short_d16 v84, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:128 // load C
/* (d1,vc1,d0,vc0)=(15,0,3,) */
buffer_load_short_d16 v87, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:160 // load C

/* rC *= alpha batchElements=[(15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (15, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

s_waitcnt  vmcnt(3)                     // vmcnt(6) = 8 - 1 (beta) - 1 (scaleAlphaVec) lgkmcnt(3) = 4 - 1 (bias) (interleaved)

v_mul_f32 v[vgprValuC+51], v[vgprValuScalAlpha_M+0], v[vgprValuC+51]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v78                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+51], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+51]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v51, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+51], v[vgprValuC+51] // check Nan
v_bfe_u32 v70, v[vgprValuC+51], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+51], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+51], v70, v72, s[66:67]
v_lshrrev_b32 v51, 16, v[vgprValuC+51]             // convert C to bf16
s_mul_i32 s66, s[sgprStrideD1J], 8                 // scale StrideD *= numRows(4) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_short v51, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt  vmcnt(2)                     // vmcnt(4) = 8 - 2 (beta) - 2 (scaleAlphaVec) lgkmcnt(2) = 4 - 2 (bias) (interleaved)

v_mul_f32 v[vgprValuC+55], v[vgprValuScalAlpha_M+1], v[vgprValuC+55]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v81                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+55], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+55]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v55, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+55], v[vgprValuC+55] // check Nan
v_bfe_u32 v70, v[vgprValuC+55], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+55], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+55], v70, v72, s[66:67]
v_lshrrev_b32 v55, 16, v[vgprValuC+55]             // convert C to bf16
buffer_store_short v55, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt  vmcnt(1)                     // vmcnt(2) = 8 - 3 (beta) - 3 (scaleAlphaVec) lgkmcnt(1) = 4 - 3 (bias) (interleaved)

v_mul_f32 v[vgprValuC+59], v[vgprValuScalAlpha_M+2], v[vgprValuC+59]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v84                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+59], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+59]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v59, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+59], v[vgprValuC+59] // check Nan
v_bfe_u32 v70, v[vgprValuC+59], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+59], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+59], v70, v72, s[66:67]
v_lshrrev_b32 v59, 16, v[vgprValuC+59]             // convert C to bf16
buffer_store_short v59, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D

s_waitcnt  vmcnt(0)                     // vmcnt(0) = 8 - 4 (beta) - 4 (scaleAlphaVec) lgkmcnt(0) = 4 - 4 (bias) (interleaved)

v_mul_f32 v[vgprValuC+63], v[vgprValuScalAlpha_M+3], v[vgprValuC+63]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v87                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+63], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+63]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v63, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+63], v[vgprValuC+63] // check Nan
v_bfe_u32 v70, v[vgprValuC+63], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+63], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+63], v70, v72, s[66:67]
v_lshrrev_b32 v63, 16, v[vgprValuC+63]             // convert C to bf16
buffer_store_short v63, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_E1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=6 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v74, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v77, v74, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v74, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v80, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v83, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v80, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v86, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v89, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v86, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v92, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v96, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v92, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v99, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v102, v99, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v99, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v103, v66, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDC clip if OOB. offset
buffer_load_short_d16 v106, v103, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v103, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for Beta, ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+0], v[vgprValuScalAlpha_M+0], v[vgprValuC+0]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v77                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+0], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+0]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v0, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+0], v[vgprValuC+0] // check Nan
v_bfe_u32 v70, v[vgprValuC+0], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+0], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+0], v70, v72, s[66:67]
v_lshrrev_b32 v0, 16, v[vgprValuC+0]               // convert C to bf16
buffer_store_short v0, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+4], v[vgprValuScalAlpha_M+1], v[vgprValuC+4]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v83                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+4], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+4]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v4, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+4], v[vgprValuC+4] // check Nan
v_bfe_u32 v70, v[vgprValuC+4], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+4], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+4], v70, v72, s[66:67]
v_lshrrev_b32 v4, 16, v[vgprValuC+4]               // convert C to bf16
buffer_store_short v4, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+8], v[vgprValuScalAlpha_M+2], v[vgprValuC+8]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v89                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+8], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+8]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v8, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+8], v[vgprValuC+8] // check Nan
v_bfe_u32 v70, v[vgprValuC+8], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+8], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+8], v70, v72, s[66:67]
v_lshrrev_b32 v8, 16, v[vgprValuC+8]               // convert C to bf16
buffer_store_short v8, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+12], v[vgprValuScalAlpha_M+3], v[vgprValuC+12]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+12], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+12]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v12, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+12], v[vgprValuC+12] // check Nan
v_bfe_u32 v70, v[vgprValuC+12], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+12], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+12], v70, v72, s[66:67]
v_lshrrev_b32 v12, 16, v[vgprValuC+12]             // convert C to bf16
buffer_store_short v12, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+1], v[vgprValuScalAlpha_M+0], v[vgprValuC+1]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v102                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+1], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+1]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v1, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+1], v[vgprValuC+1] // check Nan
v_bfe_u32 v70, v[vgprValuC+1], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+1], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+1], v70, v72, s[66:67]
v_lshrrev_b32 v1, 16, v[vgprValuC+1]               // convert C to bf16
buffer_store_short v1, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+5], v[vgprValuScalAlpha_M+1], v[vgprValuC+5]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v106                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+5], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+5]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v5, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+5], v[vgprValuC+5] // check Nan
v_bfe_u32 v70, v[vgprValuC+5], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+5], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+5], v70, v72, s[66:67]
v_lshrrev_b32 v5, 16, v[vgprValuC+5]               // convert C to bf16
buffer_store_short v5, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v74, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v77, v74, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v74, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
s_mov_b32 s66, 80                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v80, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v83, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v80, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v86, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v89, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v86, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v92, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v96, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v92, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v99, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v102, v99, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v99, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v103, v66, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDC clip if OOB. offset
buffer_load_short_d16 v106, v103, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v103, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for Beta, ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+9], v[vgprValuScalAlpha_M+2], v[vgprValuC+9]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v77                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+9], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+9]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v9, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+9], v[vgprValuC+9] // check Nan
v_bfe_u32 v70, v[vgprValuC+9], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+9], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+9], v70, v72, s[66:67]
v_lshrrev_b32 v9, 16, v[vgprValuC+9]               // convert C to bf16
buffer_store_short v9, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+13], v[vgprValuScalAlpha_M+3], v[vgprValuC+13]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v83                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+13], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+13]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v13, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+13], v[vgprValuC+13] // check Nan
v_bfe_u32 v70, v[vgprValuC+13], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+13], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+13], v70, v72, s[66:67]
v_lshrrev_b32 v13, 16, v[vgprValuC+13]             // convert C to bf16
buffer_store_short v13, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+2], v[vgprValuScalAlpha_M+0], v[vgprValuC+2]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v89                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+2], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+2]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v2, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+2], v[vgprValuC+2] // check Nan
v_bfe_u32 v70, v[vgprValuC+2], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+2], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+2], v70, v72, s[66:67]
v_lshrrev_b32 v2, 16, v[vgprValuC+2]               // convert C to bf16
buffer_store_short v2, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+6], v[vgprValuScalAlpha_M+1], v[vgprValuC+6]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+6], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+6]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v6, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+6], v[vgprValuC+6] // check Nan
v_bfe_u32 v70, v[vgprValuC+6], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+6], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+6], v70, v72, s[66:67]
v_lshrrev_b32 v6, 16, v[vgprValuC+6]               // convert C to bf16
buffer_store_short v6, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+10], v[vgprValuScalAlpha_M+2], v[vgprValuC+10]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v102                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+10], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+10]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v10, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+10], v[vgprValuC+10] // check Nan
v_bfe_u32 v70, v[vgprValuC+10], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+10], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+10], v70, v72, s[66:67]
v_lshrrev_b32 v10, 16, v[vgprValuC+10]             // convert C to bf16
buffer_store_short v10, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+14], v[vgprValuScalAlpha_M+3], v[vgprValuC+14]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v106                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+14], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+14]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v14, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+14], v[vgprValuC+14] // check Nan
v_bfe_u32 v70, v[vgprValuC+14], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+14], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+14], v70, v72, s[66:67]
v_lshrrev_b32 v14, 16, v[vgprValuC+14]             // convert C to bf16
buffer_store_short v14, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v74, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v77, v74, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v74, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v80, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v83, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v80, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v86, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v89, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v86, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v92, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v96, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v92, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v99, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v102, v99, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v99, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v103, v66, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDC clip if OOB. offset
buffer_load_short_d16 v106, v103, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v103, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for Beta, ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+3], v[vgprValuScalAlpha_M+0], v[vgprValuC+3]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v77                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+3], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+3]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v3, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+3], v[vgprValuC+3] // check Nan
v_bfe_u32 v70, v[vgprValuC+3], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+3], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+3], v70, v72, s[66:67]
v_lshrrev_b32 v3, 16, v[vgprValuC+3]               // convert C to bf16
buffer_store_short v3, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+7], v[vgprValuScalAlpha_M+1], v[vgprValuC+7]      // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v83                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+7], v68, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+7]                 // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v7, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+7], v[vgprValuC+7] // check Nan
v_bfe_u32 v70, v[vgprValuC+7], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+7], v70, v73           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+7], v70, v72, s[66:67]
v_lshrrev_b32 v7, 16, v[vgprValuC+7]               // convert C to bf16
buffer_store_short v7, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+11], v[vgprValuScalAlpha_M+2], v[vgprValuC+11]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v89                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+11], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+11]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v11, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+11], v[vgprValuC+11] // check Nan
v_bfe_u32 v70, v[vgprValuC+11], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+11], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+11], v70, v72, s[66:67]
v_lshrrev_b32 v11, 16, v[vgprValuC+11]             // convert C to bf16
buffer_store_short v11, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+15], v[vgprValuScalAlpha_M+3], v[vgprValuC+15]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+15], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+15]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v15, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+15], v[vgprValuC+15] // check Nan
v_bfe_u32 v70, v[vgprValuC+15], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+15], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+15], v70, v72, s[66:67]
v_lshrrev_b32 v15, 16, v[vgprValuC+15]             // convert C to bf16
buffer_store_short v15, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+16], v[vgprValuScalAlpha_M+0], v[vgprValuC+16]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v102                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+16], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+16]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v16, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+16], v[vgprValuC+16] // check Nan
v_bfe_u32 v70, v[vgprValuC+16], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+16], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+16], v70, v72, s[66:67]
v_lshrrev_b32 v16, 16, v[vgprValuC+16]             // convert C to bf16
buffer_store_short v16, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+20], v[vgprValuScalAlpha_M+1], v[vgprValuC+20]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v106                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+20], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+20]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v20, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+20], v[vgprValuC+20] // check Nan
v_bfe_u32 v70, v[vgprValuC+20], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+20], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+20], v70, v72, s[66:67]
v_lshrrev_b32 v20, 16, v[vgprValuC+20]             // convert C to bf16
buffer_store_short v20, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (4,2,0,0:vw1); (4,3,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v74, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v77, v74, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v74, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
s_mov_b32 s66,80                                
v_add_co_u32 v68, vcc, v64, s66                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v80, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v83, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v80, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v86, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v89, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v86, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v92, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v96, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v92, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v99, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v102, v99, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v99, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v103, v66, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDC clip if OOB. offset
buffer_load_short_d16 v106, v103, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v103, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(4, 2, 0, 0), (4, 3, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for Beta, ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+24], v[vgprValuScalAlpha_M+2], v[vgprValuC+24]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v77                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+24], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+24]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v24, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+24], v[vgprValuC+24] // check Nan
v_bfe_u32 v70, v[vgprValuC+24], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+24], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+24], v70, v72, s[66:67]
v_lshrrev_b32 v24, 16, v[vgprValuC+24]             // convert C to bf16
buffer_store_short v24, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+28], v[vgprValuScalAlpha_M+3], v[vgprValuC+28]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v83                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+28], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+28]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v28, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+28], v[vgprValuC+28] // check Nan
v_bfe_u32 v70, v[vgprValuC+28], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+28], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+28], v70, v72, s[66:67]
v_lshrrev_b32 v28, 16, v[vgprValuC+28]             // convert C to bf16
buffer_store_short v28, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+17], v[vgprValuScalAlpha_M+0], v[vgprValuC+17]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v89                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+17], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+17]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v17, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+17], v[vgprValuC+17] // check Nan
v_bfe_u32 v70, v[vgprValuC+17], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+17], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+17], v70, v72, s[66:67]
v_lshrrev_b32 v17, 16, v[vgprValuC+17]             // convert C to bf16
buffer_store_short v17, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+21], v[vgprValuScalAlpha_M+1], v[vgprValuC+21]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+21], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+21]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v21, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+21], v[vgprValuC+21] // check Nan
v_bfe_u32 v70, v[vgprValuC+21], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+21], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+21], v70, v72, s[66:67]
v_lshrrev_b32 v21, 16, v[vgprValuC+21]             // convert C to bf16
buffer_store_short v21, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+25], v[vgprValuScalAlpha_M+2], v[vgprValuC+25]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v102                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+25], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+25]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v25, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+25], v[vgprValuC+25] // check Nan
v_bfe_u32 v70, v[vgprValuC+25], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+25], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+25], v70, v72, s[66:67]
v_lshrrev_b32 v25, 16, v[vgprValuC+25]             // convert C to bf16
buffer_store_short v25, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+29], v[vgprValuScalAlpha_M+3], v[vgprValuC+29]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v106                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+29], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+29]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v29, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+29], v[vgprValuC+29] // check Nan
v_bfe_u32 v70, v[vgprValuC+29], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+29], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+29], v70, v72, s[66:67]
v_lshrrev_b32 v29, 16, v[vgprValuC+29]             // convert C to bf16
buffer_store_short v29, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #4 (d1,d0,vc1,vc0) = */
/*    (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v74, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v77, v74, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v74, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v80, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v83, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v80, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v86, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v89, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v86, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v92, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v96, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v92, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v99, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v102, v99, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v99, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v103, v66, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDC clip if OOB. offset
buffer_load_short_d16 v106, v103, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v103, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for Beta, ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+18], v[vgprValuScalAlpha_M+0], v[vgprValuC+18]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v77                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+18], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+18]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v18, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+18], v[vgprValuC+18] // check Nan
v_bfe_u32 v70, v[vgprValuC+18], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+18], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+18], v70, v72, s[66:67]
v_lshrrev_b32 v18, 16, v[vgprValuC+18]             // convert C to bf16
buffer_store_short v18, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+22], v[vgprValuScalAlpha_M+1], v[vgprValuC+22]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v83                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+22], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+22]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v22, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+22], v[vgprValuC+22] // check Nan
v_bfe_u32 v70, v[vgprValuC+22], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+22], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+22], v70, v72, s[66:67]
v_lshrrev_b32 v22, 16, v[vgprValuC+22]             // convert C to bf16
buffer_store_short v22, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+26], v[vgprValuScalAlpha_M+2], v[vgprValuC+26]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v89                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+26], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+26]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v26, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+26], v[vgprValuC+26] // check Nan
v_bfe_u32 v70, v[vgprValuC+26], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+26], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+26], v70, v72, s[66:67]
v_lshrrev_b32 v26, 16, v[vgprValuC+26]             // convert C to bf16
buffer_store_short v26, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+30], v[vgprValuScalAlpha_M+3], v[vgprValuC+30]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+30], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+30]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v30, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+30], v[vgprValuC+30] // check Nan
v_bfe_u32 v70, v[vgprValuC+30], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+30], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+30], v70, v72, s[66:67]
v_lshrrev_b32 v30, 16, v[vgprValuC+30]             // convert C to bf16
buffer_store_short v30, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+19], v[vgprValuScalAlpha_M+0], v[vgprValuC+19]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v102                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+19], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+19]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v19, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+19], v[vgprValuC+19] // check Nan
v_bfe_u32 v70, v[vgprValuC+19], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+19], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+19], v70, v72, s[66:67]
v_lshrrev_b32 v19, 16, v[vgprValuC+19]             // convert C to bf16
buffer_store_short v19, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+23], v[vgprValuScalAlpha_M+1], v[vgprValuC+23]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v106                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+23], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+23]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v23, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+23], v[vgprValuC+23] // check Nan
v_bfe_u32 v70, v[vgprValuC+23], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+23], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+23], v70, v72, s[66:67]
v_lshrrev_b32 v23, 16, v[vgprValuC+23]             // convert C to bf16
buffer_store_short v23, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #5 (d1,d0,vc1,vc0) = */
/*    (7,2,0,0:vw1); (7,3,0,0:vw1); (8,0,0,0:vw1); (8,1,0,0:vw1); (8,2,0,0:vw1); (8,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v74, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v77, v74, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v74, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v80, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v83, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v80, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,0,0) */
s_mov_b32 s66, 100
v_add_co_u32 v65, vcc, v65, s66                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 100                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 100                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v86, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v89, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v86, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v92, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v96, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v92, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v99, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v102, v99, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v99, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(8,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v103, v66, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDC clip if OOB. offset
buffer_load_short_d16 v106, v103, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v103, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(7, 2, 0, 0), (7, 3, 0, 0), (8, 0, 0, 0), (8, 1, 0, 0), (8, 2, 0, 0), (8, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for Beta, ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+27], v[vgprValuScalAlpha_M+2], v[vgprValuC+27]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v77                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+27], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+27]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v27, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+27], v[vgprValuC+27] // check Nan
v_bfe_u32 v70, v[vgprValuC+27], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+27], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+27], v70, v72, s[66:67]
v_lshrrev_b32 v27, 16, v[vgprValuC+27]             // convert C to bf16
buffer_store_short v27, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+31], v[vgprValuScalAlpha_M+3], v[vgprValuC+31]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v83                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+31], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+31]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v31, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+31], v[vgprValuC+31] // check Nan
v_bfe_u32 v70, v[vgprValuC+31], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+31], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+31], v70, v72, s[66:67]
v_lshrrev_b32 v31, 16, v[vgprValuC+31]             // convert C to bf16
buffer_store_short v31, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+32], v[vgprValuScalAlpha_M+0], v[vgprValuC+32]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v89                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+32], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+32]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v32, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+32], v[vgprValuC+32] // check Nan
v_bfe_u32 v70, v[vgprValuC+32], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+32], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+32], v70, v72, s[66:67]
v_lshrrev_b32 v32, 16, v[vgprValuC+32]             // convert C to bf16
buffer_store_short v32, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+36], v[vgprValuScalAlpha_M+1], v[vgprValuC+36]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+36], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+36]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v36, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+36], v[vgprValuC+36] // check Nan
v_bfe_u32 v70, v[vgprValuC+36], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+36], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+36], v70, v72, s[66:67]
v_lshrrev_b32 v36, 16, v[vgprValuC+36]             // convert C to bf16
buffer_store_short v36, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+40], v[vgprValuScalAlpha_M+2], v[vgprValuC+40]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v102                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+40], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+40]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v40, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+40], v[vgprValuC+40] // check Nan
v_bfe_u32 v70, v[vgprValuC+40], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+40], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+40], v70, v72, s[66:67]
v_lshrrev_b32 v40, 16, v[vgprValuC+40]             // convert C to bf16
buffer_store_short v40, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+44], v[vgprValuScalAlpha_M+3], v[vgprValuC+44]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v106                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+44], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+44]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v44, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+44], v[vgprValuC+44] // check Nan
v_bfe_u32 v70, v[vgprValuC+44], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+44], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+44], v70, v72, s[66:67]
v_lshrrev_b32 v44, 16, v[vgprValuC+44]             // convert C to bf16
buffer_store_short v44, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #6 (d1,d0,vc1,vc0) = */
/*    (9,0,0,0:vw1); (9,1,0,0:vw1); (9,2,0,0:vw1); (9,3,0,0:vw1); (10,0,0,0:vw1); (10,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(9,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v74, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v77, v74, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v74, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v80, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v83, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v80, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v86, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v89, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v86, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(9,0,3,0) */
s_mov_b32 s66, 80                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v92, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v96, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v92, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v99, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v102, v99, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v99, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v103, v66, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDC clip if OOB. offset
buffer_load_short_d16 v106, v103, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v103, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(9, 0, 0, 0), (9, 1, 0, 0), (9, 2, 0, 0), (9, 3, 0, 0), (10, 0, 0, 0), (10, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for Beta, ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+33], v[vgprValuScalAlpha_M+0], v[vgprValuC+33]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v77                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+33], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+33]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v33, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+33], v[vgprValuC+33] // check Nan
v_bfe_u32 v70, v[vgprValuC+33], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+33], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+33], v70, v72, s[66:67]
v_lshrrev_b32 v33, 16, v[vgprValuC+33]             // convert C to bf16
buffer_store_short v33, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+37], v[vgprValuScalAlpha_M+1], v[vgprValuC+37]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v83                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+37], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+37]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v37, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+37], v[vgprValuC+37] // check Nan
v_bfe_u32 v70, v[vgprValuC+37], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+37], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+37], v70, v72, s[66:67]
v_lshrrev_b32 v37, 16, v[vgprValuC+37]             // convert C to bf16
buffer_store_short v37, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+41], v[vgprValuScalAlpha_M+2], v[vgprValuC+41]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v89                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+41], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+41]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v41, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+41], v[vgprValuC+41] // check Nan
v_bfe_u32 v70, v[vgprValuC+41], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+41], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+41], v70, v72, s[66:67]
v_lshrrev_b32 v41, 16, v[vgprValuC+41]             // convert C to bf16
buffer_store_short v41, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+45], v[vgprValuScalAlpha_M+3], v[vgprValuC+45]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+45], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+45]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v45, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+45], v[vgprValuC+45] // check Nan
v_bfe_u32 v70, v[vgprValuC+45], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+45], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+45], v70, v72, s[66:67]
v_lshrrev_b32 v45, 16, v[vgprValuC+45]             // convert C to bf16
buffer_store_short v45, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+34], v[vgprValuScalAlpha_M+0], v[vgprValuC+34]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v102                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+34], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+34]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v34, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+34], v[vgprValuC+34] // check Nan
v_bfe_u32 v70, v[vgprValuC+34], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+34], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+34], v70, v72, s[66:67]
v_lshrrev_b32 v34, 16, v[vgprValuC+34]             // convert C to bf16
buffer_store_short v34, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+38], v[vgprValuScalAlpha_M+1], v[vgprValuC+38]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v106                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+38], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+38]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v38, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+38], v[vgprValuC+38] // check Nan
v_bfe_u32 v70, v[vgprValuC+38], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+38], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+38], v70, v72, s[66:67]
v_lshrrev_b32 v38, 16, v[vgprValuC+38]             // convert C to bf16
buffer_store_short v38, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #7 (d1,d0,vc1,vc0) = */
/*    (10,2,0,0:vw1); (10,3,0,0:vw1); (11,0,0,0:vw1); (11,1,0,0:vw1); (11,2,0,0:vw1); (11,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(10,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v74, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v77, v74, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v74, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(10,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v80, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v83, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v80, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v86, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v89, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v86, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v92, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v96, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v92, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v99, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v102, v99, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v99, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(11,0,3,0) */
s_mov_b32 s66, 80                                 
v_add_co_u32 v68, vcc, v64, s66                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v103, v66, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDC clip if OOB. offset
buffer_load_short_d16 v106, v103, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v103, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(10, 2, 0, 0), (10, 3, 0, 0), (11, 0, 0, 0), (11, 1, 0, 0), (11, 2, 0, 0), (11, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for Beta, ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+42], v[vgprValuScalAlpha_M+2], v[vgprValuC+42]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v77                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+42], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+42]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v42, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+42], v[vgprValuC+42] // check Nan
v_bfe_u32 v70, v[vgprValuC+42], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+42], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+42], v70, v72, s[66:67]
v_lshrrev_b32 v42, 16, v[vgprValuC+42]             // convert C to bf16
buffer_store_short v42, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+46], v[vgprValuScalAlpha_M+3], v[vgprValuC+46]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v83                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+46], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+46]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v46, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+46], v[vgprValuC+46] // check Nan
v_bfe_u32 v70, v[vgprValuC+46], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+46], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+46], v70, v72, s[66:67]
v_lshrrev_b32 v46, 16, v[vgprValuC+46]             // convert C to bf16
buffer_store_short v46, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+35], v[vgprValuScalAlpha_M+0], v[vgprValuC+35]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v89                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+35], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+35]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v35, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+35], v[vgprValuC+35] // check Nan
v_bfe_u32 v70, v[vgprValuC+35], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+35], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+35], v70, v72, s[66:67]
v_lshrrev_b32 v35, 16, v[vgprValuC+35]             // convert C to bf16
buffer_store_short v35, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+39], v[vgprValuScalAlpha_M+1], v[vgprValuC+39]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+39], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+39]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v39, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+39], v[vgprValuC+39] // check Nan
v_bfe_u32 v70, v[vgprValuC+39], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+39], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+39], v70, v72, s[66:67]
v_lshrrev_b32 v39, 16, v[vgprValuC+39]             // convert C to bf16
buffer_store_short v39, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+43], v[vgprValuScalAlpha_M+2], v[vgprValuC+43]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v102                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+43], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+43]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v43, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+43], v[vgprValuC+43] // check Nan
v_bfe_u32 v70, v[vgprValuC+43], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+43], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+43], v70, v72, s[66:67]
v_lshrrev_b32 v43, 16, v[vgprValuC+43]             // convert C to bf16
buffer_store_short v43, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+47], v[vgprValuScalAlpha_M+3], v[vgprValuC+47]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v106                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+47], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+47]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v47, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+47], v[vgprValuC+47] // check Nan
v_bfe_u32 v70, v[vgprValuC+47], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+47], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+47], v70, v72, s[66:67]
v_lshrrev_b32 v47, 16, v[vgprValuC+47]             // convert C to bf16
buffer_store_short v47, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #8 (d1,d0,vc1,vc0) = */
/*    (12,0,0,0:vw1); (12,1,0,0:vw1); (12,2,0,0:vw1); (12,3,0,0:vw1); (13,0,0,0:vw1); (13,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(12,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v74, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v77, v74, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v74, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v80, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v83, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v80, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v86, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v89, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v86, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(12,0,3,0) */
s_mov_b32 s66, 80                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v92, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v96, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v92, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v99, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v102, v99, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v99, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v103, v66, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDC clip if OOB. offset
buffer_load_short_d16 v106, v103, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v103, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(12, 0, 0, 0), (12, 1, 0, 0), (12, 2, 0, 0), (12, 3, 0, 0), (13, 0, 0, 0), (13, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for Beta, ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+48], v[vgprValuScalAlpha_M+0], v[vgprValuC+48]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v77                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+48], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+48]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v48, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+48], v[vgprValuC+48] // check Nan
v_bfe_u32 v70, v[vgprValuC+48], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+48], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+48], v70, v72, s[66:67]
v_lshrrev_b32 v48, 16, v[vgprValuC+48]             // convert C to bf16
buffer_store_short v48, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+52], v[vgprValuScalAlpha_M+1], v[vgprValuC+52]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v83                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+52], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+52]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v52, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+52], v[vgprValuC+52] // check Nan
v_bfe_u32 v70, v[vgprValuC+52], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+52], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+52], v70, v72, s[66:67]
v_lshrrev_b32 v52, 16, v[vgprValuC+52]             // convert C to bf16
buffer_store_short v52, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+56], v[vgprValuScalAlpha_M+2], v[vgprValuC+56]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v89                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+56], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+56]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v56, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+56], v[vgprValuC+56] // check Nan
v_bfe_u32 v70, v[vgprValuC+56], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+56], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+56], v70, v72, s[66:67]
v_lshrrev_b32 v56, 16, v[vgprValuC+56]             // convert C to bf16
buffer_store_short v56, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+60], v[vgprValuScalAlpha_M+3], v[vgprValuC+60]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+60], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+60]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v60, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+60], v[vgprValuC+60] // check Nan
v_bfe_u32 v70, v[vgprValuC+60], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+60], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+60], v70, v72, s[66:67]
v_lshrrev_b32 v60, 16, v[vgprValuC+60]             // convert C to bf16
buffer_store_short v60, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+49], v[vgprValuScalAlpha_M+0], v[vgprValuC+49]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v102                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+49], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+49]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v49, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+49], v[vgprValuC+49] // check Nan
v_bfe_u32 v70, v[vgprValuC+49], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+49], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+49], v70, v72, s[66:67]
v_lshrrev_b32 v49, 16, v[vgprValuC+49]             // convert C to bf16
buffer_store_short v49, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+53], v[vgprValuScalAlpha_M+1], v[vgprValuC+53]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v106                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+53], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+53]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v53, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+53], v[vgprValuC+53] // check Nan
v_bfe_u32 v70, v[vgprValuC+53], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+53], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+53], v70, v72, s[66:67]
v_lshrrev_b32 v53, 16, v[vgprValuC+53]             // convert C to bf16
buffer_store_short v53, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #9 (d1,d0,vc1,vc0) = */
/*    (13,2,0,0:vw1); (13,3,0,0:vw1); (14,0,0,0:vw1); (14,1,0,0:vw1); (14,2,0,0:vw1); (14,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v107, BufferOOB
/* (d1,vc1,d0,vc0)=(13,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v74, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v77, v74, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v74, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v107, v74, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(13,0,3,0) */
s_mov_b32 s66, 80                                
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v80, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v83, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v80, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v107, v80, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v86, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v89, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v86, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v107, v86, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v92, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v96, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v92, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v107, v92, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v99, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v102, v99, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v99, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v99, v107, v99, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(14,0,3,0) */
s_mov_b32 s66, 80                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v103, v66, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDC clip if OOB. offset
buffer_load_short_d16 v106, v103, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v103, v67, v68, 0x1                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v103, v107, v103, s[70:71]           // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(13, 2, 0, 0), (13, 3, 0, 0), (14, 0, 0, 0), (14, 1, 0, 0), (14, 2, 0, 0), (14, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for Beta, ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+57], v[vgprValuScalAlpha_M+2], v[vgprValuC+57]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v77                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+57], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+57]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v57, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+57], v[vgprValuC+57] // check Nan
v_bfe_u32 v70, v[vgprValuC+57], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+57], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+57], v70, v72, s[66:67]
v_lshrrev_b32 v57, 16, v[vgprValuC+57]             // convert C to bf16
buffer_store_short v57, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+61], v[vgprValuScalAlpha_M+3], v[vgprValuC+61]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v83                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+61], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+61]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v61, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+61], v[vgprValuC+61] // check Nan
v_bfe_u32 v70, v[vgprValuC+61], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+61], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+61], v70, v72, s[66:67]
v_lshrrev_b32 v61, 16, v[vgprValuC+61]             // convert C to bf16
buffer_store_short v61, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+50], v[vgprValuScalAlpha_M+0], v[vgprValuC+50]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v89                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+50], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+50]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v50, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+50], v[vgprValuC+50] // check Nan
v_bfe_u32 v70, v[vgprValuC+50], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+50], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+50], v70, v72, s[66:67]
v_lshrrev_b32 v50, 16, v[vgprValuC+50]             // convert C to bf16
buffer_store_short v50, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+54], v[vgprValuScalAlpha_M+1], v[vgprValuC+54]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+54], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+54]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v54, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+54], v[vgprValuC+54] // check Nan
v_bfe_u32 v70, v[vgprValuC+54], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+54], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+54], v70, v72, s[66:67]
v_lshrrev_b32 v54, 16, v[vgprValuC+54]             // convert C to bf16
buffer_store_short v54, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+58], v[vgprValuScalAlpha_M+2], v[vgprValuC+58]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v102                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+58], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+58]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v58, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+58], v[vgprValuC+58] // check Nan
v_bfe_u32 v70, v[vgprValuC+58], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+58], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+58], v70, v72, s[66:67]
v_lshrrev_b32 v58, 16, v[vgprValuC+58]             // convert C to bf16
buffer_store_short v58, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+62], v[vgprValuScalAlpha_M+3], v[vgprValuC+62]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v106                        // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+62], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+62]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v62, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+62], v[vgprValuC+62] // check Nan
v_bfe_u32 v70, v[vgprValuC+62], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+62], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+62], v70, v72, s[66:67]
v_lshrrev_b32 v62, 16, v[vgprValuC+62]             // convert C to bf16
buffer_store_short v62, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #10 (d1,d0,vc1,vc0) = */
/*    (15,0,0,0:vw1); (15,1,0,0:vw1); (15,2,0,0:vw1); (15,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v99, BufferOOB
/* (d1,vc1,d0,vc0)=(15,0,0,0) */
v_add_co_u32 v65, vcc, v65, 4                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s66, s[sgprStrideC1J], 4                 // scale stride
v_add_u32 v66, v66, s66                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s66, s[sgprStrideD1J], 4                 // scale stride
v_add_u32 v67, v67, s66                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[66:67], v64, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v74, v66, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v99, v74, s[70:71]              // LDC clip if OOB. offset
buffer_load_short_d16 v77, v74, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v74, v67, v64, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v74, v99, v74, s[70:71]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,1,0) */
v_add_co_u32 v68, vcc, v64, 16                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v80, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v99, v80, s[70:71]              // LDC clip if OOB. offset
buffer_load_short_d16 v83, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v80, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v80, v99, v80, s[70:71]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,2,0) */
s_mov_b32 s66, 64                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v86, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v99, v86, s[70:71]              // LDC clip if OOB. offset
buffer_load_short_d16 v89, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v86, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v86, v99, v86, s[70:71]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(15,0,3,0) */
s_mov_b32 s66, 80                                 
v_add_co_u32 v68, vcc, v64, s66                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[66:67], v68, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v65, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[66:67], s[70:71]             // in0 && in1
v_add_lshl_u32 v92, v66, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v99, v92, s[70:71]              // LDC clip if OOB. offset
buffer_load_short_d16 v96, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C

v_add_lshl_u32 v92, v67, v68, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v92, v99, v92, s[70:71]              // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(15, 0, 0, 0), (15, 1, 0, 0), (15, 2, 0, 0), (15, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for Beta, ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_mov_b32 v71, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v72, 0x7fff0000                          // fp32 Nan
v_mov_b32 v73, 0x7fff                              // rounding bias for bfloat16

v_mul_f32 v[vgprValuC+51], v[vgprValuScalAlpha_M+0], v[vgprValuC+51]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v77                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+51], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+0], v[vgprValuC+51]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v51, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+51], v[vgprValuC+51] // check Nan
v_bfe_u32 v70, v[vgprValuC+51], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+51], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+51], v70, v72, s[66:67]
v_lshrrev_b32 v51, 16, v[vgprValuC+51]             // convert C to bf16
buffer_store_short v51, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+55], v[vgprValuScalAlpha_M+1], v[vgprValuC+55]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v83                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+55], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+1], v[vgprValuC+55]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v55, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+55], v[vgprValuC+55] // check Nan
v_bfe_u32 v70, v[vgprValuC+55], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+55], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+55], v70, v72, s[66:67]
v_lshrrev_b32 v55, 16, v[vgprValuC+55]             // convert C to bf16
buffer_store_short v55, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+59], v[vgprValuScalAlpha_M+2], v[vgprValuC+59]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v89                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+59], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+2], v[vgprValuC+59]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v59, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+59], v[vgprValuC+59] // check Nan
v_bfe_u32 v70, v[vgprValuC+59], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+59], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+59], v70, v72, s[66:67]
v_lshrrev_b32 v59, 16, v[vgprValuC+59]             // convert C to bf16
buffer_store_short v59, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

v_mul_f32 v[vgprValuC+63], v[vgprValuScalAlpha_M+3], v[vgprValuC+63]    // *= scaleAlphaVecVMul
v_lshlrev_b32 v68, 16, v96                         // convert bf16 to fp32
v_fmac_f32 v[vgprValuC+63], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_add_f32 v68, v[vgprValuBias_M+3], v[vgprValuC+63]                // C += bias
s_swappc_b64 s[64:65], s[54:55]
v_mov_b32 v63, v68
v_cmp_u_f32 s[66:67], v[vgprValuC+63], v[vgprValuC+63] // check Nan
v_bfe_u32 v70, v[vgprValuC+63], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v70, v[vgprValuC+63], v70, v73          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+63], v70, v72, s[66:67]
v_lshrrev_b32 v63, 16, v[vgprValuC+63]             // convert C to bf16
buffer_store_short v63, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_Activation_None_VW1:
s_setpc_b64 s[64:65]
label_Activation_Abs_VW1:
v_and_b32 v[vgprValuC+68], 0x7fffffff, v[vgprValuC+68] // Remove sign bit
s_setpc_b64 s[64:65]
label_Activation_Clippedrelu_VW1:
v_cmp_gt_f32 vcc, v[vgprValuC+68], s[sgpractivationAlpha] // x > alpha ?
v_min_f32 v[vgprValuC+68], s[sgpractivationBeta], v[vgprValuC+68] // min(x, beta)
v_cndmask_b32 v[vgprValuC+68], 0.0, v[vgprValuC+68], vcc // set x to 0 if <= alpha
s_setpc_b64 s[64:65]
label_Activation_Gelu_VW1:
v_mul_f32 v69, 0x3d372713, v[vgprValuC+68]         // k1 * x
v_fma_f32 v69, v[vgprValuC+68], v69, 1.0           // 1 + (k1 * x * x)
v_mul_f32 v69, v[vgprValuC+68], v69                // x * (1 + k1 * x * x)
v_mul_f32 v69, 0x40135761, v69                     //  (fused 2.302208)
s_nop 0                                            // 1 wait states
v_exp_f32 v69, v69                                 // exp step 2
v_add_f32 v69, 1.0, v69                            // e^2x + 1
s_nop 0                                            // 1 wait states
v_rcp_f32 v69, v69                                 // 1 / (e^2x + 1)
v_fma_f32 v69, -2.0, v69, 2.0                      //  ( + 1 (fused))
v_mul_f32 v69, v[vgprValuC+68], v69                // x * (1 + tanh(...))
v_mul_f32 v[vgprValuC+68], 0.5, v69                // 0.5 * x * (1 + tanh(...))
s_setpc_b64 s[64:65]
label_Activation_Leakyrelu_VW1:
v_mul_f32 v69, s[sgpractivationAlpha], v[vgprValuC+68] // tmp = x * alpha
v_cmp_ge_f32 vcc, v[vgprValuC+68], 0.0             // x >= 0 ?
v_cndmask_b32 v[vgprValuC+68], v69, v[vgprValuC+68], vcc // set x to tmp if < 0
s_setpc_b64 s[64:65]
label_Activation_Relu_VW1:
v_max_f32 v[vgprValuC+68], v[vgprValuC+68], 0      // x = max(0, x)
s_setpc_b64 s[64:65]
label_Activation_Sigmoid_VW1:
v_mul_f32 v[vgprValuC+68], 0xbfb8aa3b, v[vgprValuC+68] //  (fused -1.442695)
s_nop 0                                            // 1 wait states
v_exp_f32 v[vgprValuC+68], v[vgprValuC+68]         // exp step 2
v_add_f32 v[vgprValuC+68], 1.0, v[vgprValuC+68]    // 1 + exp(-x)
s_nop 0                                            // 1 wait states
v_rcp_f32 v[vgprValuC+68], v[vgprValuC+68]         // 1 / (1 + exp(-x))
s_setpc_b64 s[64:65]
label_Activation_Tanh_VW1:
v_mul_f32 v[vgprValuC+68], s[sgpractivationAlpha], v[vgprValuC+68] // x * alpha
v_mul_f32 v[vgprValuC+68], 0x4038aa3b, v[vgprValuC+68] //  (fused 2)
s_nop 0                                            // 1 wait states
v_exp_f32 v[vgprValuC+68], v[vgprValuC+68]         // exp step 2
v_add_f32 v[vgprValuC+68], 1.0, v[vgprValuC+68]    // e^2x + 1
s_nop 0                                            // 1 wait states
v_rcp_f32 v[vgprValuC+68], v[vgprValuC+68]         // 1 / (e^2x + 1)
v_fma_f32 v[vgprValuC+68], -2.0, v[vgprValuC+68], 1.0 // (-2) * (1 / (e^2x + 1)) + 1
v_mul_f32 v[vgprValuC+68], s[sgpractivationBeta], v[vgprValuC+68] // beta * tanh(x)
s_setpc_b64 s[64:65]
label_Activation_Geluscaling_VW1:
v_mul_f32 v69, 0x3d372713, v[vgprValuC+68]         // k1 * x
v_fma_f32 v69, v[vgprValuC+68], v69, 1.0           // 1 + (k1 * x * x)
v_mul_f32 v69, v[vgprValuC+68], v69                // x * (1 + k1 * x * x)
v_mul_f32 v69, 0x40135761, v69                     //  (fused 2.302208)
s_nop 0                                            // 1 wait states
v_exp_f32 v69, v69                                 // exp step 2
v_add_f32 v69, 1.0, v69                            // e^2x + 1
s_nop 0                                            // 1 wait states
v_rcp_f32 v69, v69                                 // 1 / (e^2x + 1)
v_fma_f32 v69, -2.0, v69, 2.0                      //  ( + 1 (fused))
v_mul_f32 v69, v[vgprValuC+68], v69                // x * (1 + tanh(...))
v_mul_f32 v69, 0.5, v69                            // 0.5 * x * (1 + tanh(...))
v_mul_f32 v[vgprValuC+68], s[sgpractivationAlpha], v69 // 0.5 * x * (1 + tanh(...)) * scale
s_setpc_b64 s[64:65]
s_endpgm
