#include "hygon_vendor.h"
#include "hip/hip_bfloat16.h"
#include "hip/hip_ext.h"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "hip_kernels/ag_group_gemm_kernels.h"
#include <c10/core/ScalarType.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <type_traits>
#include <stdint.h>
#include <vector>

namespace bytedance
{
namespace hygon_vendor_impl
{
namespace detail = ::bytedance::hygon_vendor::detail;
} // namespace hygon_vendor_impl
} // namespace bytedance


namespace bytedance
{
namespace hygon_vendor_impl
{
namespace
{

constexpr int AG_RANK_TILE_INFO_STRIDE = 5;
constexpr int AG_RANK_TILE_EXPERT = 0;
constexpr int AG_RANK_TILE_SORTED_START = 1;
constexpr int AG_RANK_TILE_ROWS = 2;
constexpr int AG_RANK_TILE_RANK_START = 3;
constexpr int AG_RANK_TILE_RANK_END = 4;
constexpr int AG_RANK_TILE_PROBLEM_INFO_STRIDE = 8;
constexpr int AG_RANK_TILE_PROBLEM_CHUNK_START = 5;
constexpr int AG_RANK_TILE_PROBLEM_CHUNK_END = 6;
constexpr int AG_RANK_TILE_PROBLEM_RESERVED = 7;

bool get_env_bool_local(const char *name, bool default_value)
{
    const char *value = detail::get_vendor_env(name);
    if (value == nullptr)
    {
        return default_value;
    }
    return !(value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

int get_env_int_local(const char *name, int default_value)
{
    const char *value = detail::get_vendor_env(name);
    if (value == nullptr || value[0] == '\0')
    {
        return default_value;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value)
    {
        return default_value;
    }
    return static_cast<int>(parsed);
}

int get_group_n_chunk_size_local()
{
    int value = get_env_int_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_GROUP_N_CHUNK", 256);
    if (value < 256 || value % 256 != 0)
    {
        value = 256;
    }
    return value;
}

enum class AgMt256AsmVariant
{
    Auto,
    V4DirectInput,
    V5DirectInputWgm8,
    V7DirectOutputNChunkSignal,
    V10DirectOutputOobStoreWgm8,
    V12DirectOutputNChunkSignalWgm8,
    V13DirectOutputNChunkSignalChunkMajor,
    V14DirectOutputNChunkSignalGroupedWgm,
    V15DirectOutputNChunkSignalWideGroupedWgm,
    V36DynamicPackedOutputDiagnostic,
    V36DynamicDirectOutputNChunkSignal,
    V36DynamicTNPackedInputDirectOutputNChunkSignal,
    V36DynamicTNLoaderKNDirectOutputNChunkSignal,
    // Native [E,K_local,N] route with four direct-to-LDS TN producer waves.
    // The compute body remains the Ailk fragment contract; this is an
    // isolated experiment and does not change the default variant=19 path.
    V36DynamicTNProducerKNDirectOutputNChunkSignal,
    V36DynamicDirectOutputNChunkSignalScaleInGemm,
    V36DynamicWgm8_64BitDirectOutputNChunkSignal,
    V36DynamicDirectOutputNChunkSignalFastBuffer,
    V36DynamicNNAilkDirectOutputNChunkSignal,
    // Compact v36-style Native-KN ABI.  It uses the chunk-capable code
    // object, but deliberately does not set the chunk (bit 7) contract: the
    // host metadata remains the five-field compact row-span layout and bit 6
    // selects its decoder.
    V36DynamicNNAilkCompactDirectOutputNChunkSignal,
    // Megatron FC1-only Native-KN route with M-row chunk-ready input
    // signals.  Keep the historical variant above unchanged.
    V36DynamicNNAilkChunkDirectOutputNChunkSignal,
    // Raw Hygon [local_M,K]+[M,topk] Native-KN route.  This is intentionally
    // variant 19: it keeps the v36 dynamic ready-queue scheduler and selects
    // the host-side v36 push ordering instead of the Native-KN peer defaults.
    V36DynamicNNAilkRawHygonOverlapDirectOutputNChunkSignal,
    V36DynamicNNAilkStaticDirectOutputNChunkSignal,
    V36DynamicNNAilkPackedInputDirectOutputNChunkSignal,
    V36DynamicNNAilkStreamingPackedInputDirectOutputNChunkSignal,
    V36DynamicExpertNMajorWgm1DirectOutputNChunkSignal,
    V36AutoDirect,
    V36LayoutAuto,
    V15V36Direct,
};

constexpr int kAgMt256Wgm1TaskNTileBits = 10;
constexpr int kAgMt256Wgm1MaxNTiles = 1 << kAgMt256Wgm1TaskNTileBits;
constexpr int kAgMt256Wgm1MaxProblems =
    1 << (32 - kAgMt256Wgm1TaskNTileBits);

AgMt256AsmVariant get_ag_mt256_asm_variant()
{
    const char *value =
        detail::get_vendor_env("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_VARIANT");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "auto") == 0)
    {
        return AgMt256AsmVariant::Auto;
    }
    if (std::strcmp(value, "v4") == 0 ||
        std::strcmp(value, "direct_input") == 0)
    {
        return AgMt256AsmVariant::V4DirectInput;
    }
    if (std::strcmp(value, "v5") == 0 ||
        std::strcmp(value, "direct_input_wgm8") == 0)
    {
        return AgMt256AsmVariant::V5DirectInputWgm8;
    }
    if (std::strcmp(value, "v7") == 0 ||
        std::strcmp(value, "direct_output_nchunk_signal") == 0)
    {
        return AgMt256AsmVariant::V7DirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v10") == 0 ||
        std::strcmp(value, "direct_output_oob_store_wgm8") == 0)
    {
        return AgMt256AsmVariant::V10DirectOutputOobStoreWgm8;
    }
    if (std::strcmp(value, "v12") == 0 ||
        std::strcmp(value, "direct_output_nchunk_signal_wgm8") == 0)
    {
        return AgMt256AsmVariant::V12DirectOutputNChunkSignalWgm8;
    }
    if (std::strcmp(value, "v13") == 0 ||
        std::strcmp(value, "direct_output_nchunk_signal_chunk_major") == 0)
    {
        return AgMt256AsmVariant::V13DirectOutputNChunkSignalChunkMajor;
    }
    if (std::strcmp(value, "v14") == 0 ||
        std::strcmp(value, "direct_output_nchunk_signal_grouped_wgm") == 0)
    {
        return AgMt256AsmVariant::V14DirectOutputNChunkSignalGroupedWgm;
    }
    if (std::strcmp(value, "v15") == 0 ||
        std::strcmp(value,
                    "direct_output_nchunk_signal_wide_grouped_wgm") == 0)
    {
        return AgMt256AsmVariant::V15DirectOutputNChunkSignalWideGroupedWgm;
    }
    if (std::strcmp(value, "nk_direct") == 0 ||
        std::strcmp(value, "v36_direct") == 0 ||
        std::strcmp(value,
                    "v36_dynamic_direct_output_nchunk_signal") == 0)
    {
        return AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v36_tn_packed_direct") == 0 ||
        std::strcmp(
            value,
            "v36_dynamic_tn_packed_input_direct_output_nchunk_signal") == 0)
    {
        return AgMt256AsmVariant::
            V36DynamicTNPackedInputDirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v36_tnloader_kn_direct") == 0 ||
        std::strcmp(value, "v36_kn_tnloader_direct") == 0 ||
        std::strcmp(
            value,
            "v36_dynamic_tnloader_kn_direct_output_nchunk_signal") == 0)
    {
        return AgMt256AsmVariant::
            V36DynamicTNLoaderKNDirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v36_tnproducer_kn_direct") == 0 ||
        std::strcmp(value, "v36_kn_tnproducer_direct") == 0 ||
        std::strcmp(
            value,
            "v36_dynamic_tnproducer_kn_direct_output_nchunk_signal") == 0)
    {
        return AgMt256AsmVariant::V36DynamicTNProducerKNDirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v36_tnloader_kn_megatron_direct") == 0 ||
        std::strcmp(
            value,
            "v36_dynamic_tnloader_kn_megatron_direct_output_nchunk_signal") ==
            0)
    {
        // Keep the historical variant=12 selector untouched while routing
        // this explicitly named Megatron-only ABI to its independent code
        // object and reserved-bit-4 input mapping.
        setenv("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_TNLOADER_MEGATRON", "1", 1);
        return AgMt256AsmVariant::V36DynamicTNLoaderKNDirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v36_scale_in_gemm") == 0 ||
        std::strcmp(
            value,
            "v36_dynamic_direct_output_nchunk_signal_scale_in_gemm") == 0)
    {
        return AgMt256AsmVariant::
            V36DynamicDirectOutputNChunkSignalScaleInGemm;
    }
    if (std::strcmp(value, "v36_packed_diagnostic") == 0 ||
        std::strcmp(
            value,
            "v36_dynamic_stage_ordered_ready_queue_ag_frontload16_win64_"
            "tailmerge_wgm8") == 0)
    {
        return AgMt256AsmVariant::V36DynamicPackedOutputDiagnostic;
    }
    if (std::strcmp(value, "v36_wgm8_64_direct") == 0 ||
        std::strcmp(value,
                    "v36_dynamic_wgm8_64bit_direct_output_nchunk_signal") == 0 ||
        std::strcmp(
            value,
            "v36_dynamic_wgm8_segment_explicit_hi_store_only") == 0)
    {
        return AgMt256AsmVariant::V36DynamicWgm8_64BitDirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v36_direct_fast") == 0 ||
        std::strcmp(value,
                    "v36_dynamic_direct_output_nchunk_signal_fast_buffer") == 0)
    {
        // The restored native-TN route no longer uses the physical-KN
        // fast-buffer sibling. Keep the selector as a compatibility alias.
        return AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "kn_direct") == 0 ||
        std::strcmp(value, "v36_nn_direct") == 0 ||
        std::strcmp(value, "v36_nn_ailk_direct") == 0 ||
        std::strcmp(value,
                    "v36_dynamic_nn_ailk_direct_output_nchunk_signal") == 0)
    {
        return AgMt256AsmVariant::V36DynamicNNAilkDirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v36_nn_chunk_direct") == 0 ||
        std::strcmp(value,
                    "v36_dynamic_nn_ailk_chunk_direct_output_nchunk_signal") == 0)
    {
        return AgMt256AsmVariant::V36DynamicNNAilkChunkDirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v36_nn_compact_direct") == 0 ||
        std::strcmp(value,
                    "v36_dynamic_nn_ailk_compact_direct_output_nchunk_signal") ==
            0)
    {
        // Backward-compatible spelling for the isolated compact KN route.
        // Use the chunk-capable object: unlike the original nn_direct object,
        // it contains the reserved-bit-6 physical-row decoder.
        setenv("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_NN_COMPACT_ROWS", "1", 1);
        setenv("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_NN_ROUTED_INPUT", "1", 1);
        return AgMt256AsmVariant::V36DynamicNNAilkCompactDirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v36_nn_chunk_compact_direct") == 0 ||
        std::strcmp(
            value,
            "v36_dynamic_nn_ailk_chunk_compact_direct_output_nchunk_signal") ==
            0)
    {
        // Independent compact-input Native-KN selector.  It keeps the
        // historical variant=16/nn_direct path untouched while selecting the
        // chunk-capable v36 scheduler and its compact physical-row ABI.
        setenv("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_NN_COMPACT_ROWS", "1", 1);
        setenv("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_NN_ROUTED_INPUT", "1", 1);
        return AgMt256AsmVariant::V36DynamicNNAilkChunkDirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v19") == 0 ||
        std::strcmp(value, "v36_nn_raw_hygon_overlap_direct") == 0 ||
        std::strcmp(
            value,
            "v36_dynamic_nn_ailk_raw_hygon_overlap_direct_output_nchunk_signal") ==
            0)
    {
        setenv("FTRAIN_HYGON_AG_GROUP_GEMM_NATIVE_KN_RAW_V36_OVERLAP", "1", 1);
        return AgMt256AsmVariant::
            V36DynamicNNAilkRawHygonOverlapDirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v36_nn_static_direct") == 0 ||
        std::strcmp(value,
                   "v36_dynamic_nn_ailk_static_direct_output_nchunk_signal") == 0)
    {
        return AgMt256AsmVariant::V36DynamicNNAilkStaticDirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v36_nn_packed_direct") == 0 ||
        std::strcmp(value, "v36_nn_ailk_packed_direct") == 0 ||
        std::strcmp(
            value,
            "v36_dynamic_nn_ailk_packed_input_direct_output_nchunk_signal") ==
            0)
    {
        return AgMt256AsmVariant::
            V36DynamicNNAilkPackedInputDirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v36_nn_streaming_packed_direct") == 0 ||
        std::strcmp(value, "v36_nn_ailk_streaming_packed_direct") == 0 ||
        std::strcmp(
            value,
            "v36_dynamic_nn_ailk_streaming_packed_input_direct_output_"
            "nchunk_signal") == 0)
    {
        return AgMt256AsmVariant::
            V36DynamicNNAilkStreamingPackedInputDirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v36_wgm1_direct") == 0 ||
        std::strcmp(value, "v36_expert_nmajor_wgm1_direct") == 0 ||
        std::strcmp(
            value,
            "v36_dynamic_expert_nmajor_wgm1_direct_output_nchunk_signal") == 0)
    {
        return AgMt256AsmVariant::
            V36DynamicExpertNMajorWgm1DirectOutputNChunkSignal;
    }
    if (std::strcmp(value, "v36_auto_direct") == 0 ||
        std::strcmp(value, "v36_dynamic_auto_direct") == 0)
    {
        return AgMt256AsmVariant::V36AutoDirect;
    }
    if (std::strcmp(value, "v36_layout_auto") == 0 ||
        std::strcmp(value, "v36_weight_layout_auto") == 0 ||
        std::strcmp(value, "v36_auto_layout") == 0)
    {
        return AgMt256AsmVariant::V36LayoutAuto;
    }
    if (std::strcmp(value, "v15_v36_direct") == 0)
    {
        return AgMt256AsmVariant::V15V36Direct;
    }
    return AgMt256AsmVariant::Auto;
}

bool ag_mt256_variant_direct_input(AgMt256AsmVariant variant)
{
    return variant == AgMt256AsmVariant::V4DirectInput ||
           variant == AgMt256AsmVariant::V5DirectInputWgm8 ||
           variant == AgMt256AsmVariant::V7DirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V10DirectOutputOobStoreWgm8 ||
           variant == AgMt256AsmVariant::V12DirectOutputNChunkSignalWgm8 ||
           variant == AgMt256AsmVariant::V13DirectOutputNChunkSignalChunkMajor ||
           variant == AgMt256AsmVariant::V14DirectOutputNChunkSignalGroupedWgm ||
           variant == AgMt256AsmVariant::V15DirectOutputNChunkSignalWideGroupedWgm ||
           variant == AgMt256AsmVariant::V36DynamicPackedOutputDiagnostic ||
           variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicTNPackedInputDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicTNLoaderKNDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicTNProducerKNDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalScaleInGemm ||
           variant == AgMt256AsmVariant::V36DynamicWgm8_64BitDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalFastBuffer ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkCompactDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkChunkDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkRawHygonOverlapDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkStaticDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkPackedInputDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkStreamingPackedInputDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicExpertNMajorWgm1DirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36AutoDirect ||
           variant == AgMt256AsmVariant::V36LayoutAuto ||
           variant == AgMt256AsmVariant::V15V36Direct;
}

bool ag_mt256_variant_direct_output(AgMt256AsmVariant variant)
{
    return variant == AgMt256AsmVariant::V7DirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V10DirectOutputOobStoreWgm8 ||
           variant == AgMt256AsmVariant::V12DirectOutputNChunkSignalWgm8 ||
           variant == AgMt256AsmVariant::V13DirectOutputNChunkSignalChunkMajor ||
           variant == AgMt256AsmVariant::V14DirectOutputNChunkSignalGroupedWgm ||
           variant == AgMt256AsmVariant::V15DirectOutputNChunkSignalWideGroupedWgm ||
           variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicTNPackedInputDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicTNLoaderKNDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicTNProducerKNDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalScaleInGemm ||
           variant == AgMt256AsmVariant::V36DynamicWgm8_64BitDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalFastBuffer ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkCompactDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkChunkDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkRawHygonOverlapDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkStaticDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkPackedInputDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkStreamingPackedInputDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicExpertNMajorWgm1DirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36AutoDirect ||
           variant == AgMt256AsmVariant::V36LayoutAuto ||
           variant == AgMt256AsmVariant::V15V36Direct;
}

bool ag_mt256_variant_wgm8(AgMt256AsmVariant variant)
{
    return variant == AgMt256AsmVariant::V5DirectInputWgm8 ||
           variant == AgMt256AsmVariant::V10DirectOutputOobStoreWgm8 ||
           variant == AgMt256AsmVariant::V12DirectOutputNChunkSignalWgm8 ||
           variant == AgMt256AsmVariant::V36DynamicPackedOutputDiagnostic ||
           variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicTNPackedInputDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicTNLoaderKNDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicTNProducerKNDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalScaleInGemm ||
           variant == AgMt256AsmVariant::V36DynamicWgm8_64BitDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalFastBuffer ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkCompactDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkChunkDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkRawHygonOverlapDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkStaticDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkPackedInputDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkStreamingPackedInputDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36AutoDirect ||
           variant == AgMt256AsmVariant::V36LayoutAuto ||
           variant == AgMt256AsmVariant::V15V36Direct;
}

bool ag_mt256_variant_grouped_wgm(AgMt256AsmVariant variant)
{
    return variant == AgMt256AsmVariant::V14DirectOutputNChunkSignalGroupedWgm ||
           variant == AgMt256AsmVariant::V15DirectOutputNChunkSignalWideGroupedWgm;
}

bool ag_mt256_variant_wide_grouped_wgm(AgMt256AsmVariant variant)
{
    return variant == AgMt256AsmVariant::V15DirectOutputNChunkSignalWideGroupedWgm;
}

bool ag_mt256_variant_nchunk_signal(AgMt256AsmVariant variant)
{
    return variant == AgMt256AsmVariant::V7DirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V12DirectOutputNChunkSignalWgm8 ||
           variant == AgMt256AsmVariant::V13DirectOutputNChunkSignalChunkMajor ||
           variant == AgMt256AsmVariant::V14DirectOutputNChunkSignalGroupedWgm ||
           variant == AgMt256AsmVariant::V15DirectOutputNChunkSignalWideGroupedWgm ||
           variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicTNPackedInputDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicTNLoaderKNDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicTNProducerKNDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalScaleInGemm ||
           variant == AgMt256AsmVariant::V36DynamicWgm8_64BitDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalFastBuffer ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkCompactDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkChunkDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkRawHygonOverlapDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkStaticDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkPackedInputDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkStreamingPackedInputDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicExpertNMajorWgm1DirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36AutoDirect ||
           variant == AgMt256AsmVariant::V36LayoutAuto ||
           variant == AgMt256AsmVariant::V15V36Direct;
}

bool ag_mt256_variant_native_kn(AgMt256AsmVariant variant)
{
    return variant == AgMt256AsmVariant::V36DynamicTNLoaderKNDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicTNProducerKNDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkCompactDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkChunkDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkRawHygonOverlapDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkStaticDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::
                          V36DynamicNNAilkPackedInputDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::
                          V36DynamicNNAilkStreamingPackedInputDirectOutputNChunkSignal;
}

bool ag_mt256_variant_nn_ailk(AgMt256AsmVariant variant)
{
    return variant == AgMt256AsmVariant::V36DynamicNNAilkDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkCompactDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkChunkDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkRawHygonOverlapDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::V36DynamicNNAilkStaticDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::
                          V36DynamicNNAilkPackedInputDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::
                          V36DynamicNNAilkStreamingPackedInputDirectOutputNChunkSignal;
}

bool ag_mt256_variant_native_kn_chunk(AgMt256AsmVariant variant)
{
    return variant ==
           AgMt256AsmVariant::V36DynamicNNAilkChunkDirectOutputNChunkSignal;
}

bool ag_mt256_variant_native_kn_packed(AgMt256AsmVariant variant)
{
    return variant == AgMt256AsmVariant::
                          V36DynamicNNAilkPackedInputDirectOutputNChunkSignal ||
           variant == AgMt256AsmVariant::
                          V36DynamicNNAilkStreamingPackedInputDirectOutputNChunkSignal;
}

bool ag_mt256_variant_native_kn_streaming_pack(AgMt256AsmVariant variant)
{
    return variant == AgMt256AsmVariant::
                          V36DynamicNNAilkStreamingPackedInputDirectOutputNChunkSignal;
}

AgMt256AsmVariant ag_mt256_resolve_variant(AgMt256AsmVariant requested,
                                           int n_tiles,
                                           int full_m = 0,
                                           int /*topk*/ = 0,
                                           int n_dim = 0,
                                           bool legacy_weight_nk = false)
{
    if (requested == AgMt256AsmVariant::V36AutoDirect)
    {
        const int wgm1_min_m = std::max(
            1,
            get_env_int_local(
                "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_WGM1_MIN_M", 32768));
        if (full_m >= wgm1_min_m && n_tiles <= kAgMt256Wgm1MaxNTiles)
        {
            return AgMt256AsmVariant::
                V36DynamicExpertNMajorWgm1DirectOutputNChunkSignal;
        }
        return AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal;
    }
    if (requested == AgMt256AsmVariant::V36LayoutAuto)
    {
        // Layout is part of the public MoEReduceRS weight contract. Native
        // [E,K_local,N] weights always stay native-KN; only the independently
        // selected routed-A input strategy changes between direct and packed.
        // Keep a native-only packed override for controlled end-to-end A/B
        // runs; unlike selecting v36_nn_ailk_packed_direct globally, it leaves
        // the legacy [E,N,K_local] FC1-backward route on layout-auto.
        if (!legacy_weight_nk)
        {
            const int native_packed_min_m = std::max(
                1,
                get_env_int_local(
                    "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_LAYOUT_AUTO_NATIVE_PACKED_MIN_M",
                    1));
            if (get_env_bool_local(
                    "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_LAYOUT_AUTO_NATIVE_PACKED",
                    false) &&
                full_m >= native_packed_min_m)
            {
                return AgMt256AsmVariant::
                    V36DynamicNNAilkPackedInputDirectOutputNChunkSignal;
            }
            // Both variants consume the public [E,K_local,N] weight directly;
            // "packed" refers only to routed A rows. End-to-end Megatron MoE
            // measurements show that the extra routed-A packing pass loses to
            // direct input across the supported training shapes, including
            // large ragged-M batches. Keep layout-auto on no-pack for every M;
            // packed and streaming-packed variants remain explicitly
            // selectable for diagnostic A/B runs.
            return AgMt256AsmVariant::
                V36DynamicNNAilkDirectOutputNChunkSignal;
        }
        if (full_m >= get_env_int_local(
                         "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_WGM1_MIN_M", 32768) &&
            n_tiles <= kAgMt256Wgm1MaxNTiles)
        {
            return AgMt256AsmVariant::
                V36DynamicExpertNMajorWgm1DirectOutputNChunkSignal;
        }
        // The layout-auto legacy [E,N,K_local] route uses TN packed-input
        // for every supported output width by default.  Keep the environment
        // override for controlled regressions, but do not make small shapes
        // silently fall back to the unpacked WGM8 path.
        const int tn_packed_min_n = std::max(
            256,
            get_env_int_local(
                "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_TN_PACKED_MIN_N", 256));
        if (n_dim >= tn_packed_min_n)
        {
            return AgMt256AsmVariant::
                V36DynamicTNPackedInputDirectOutputNChunkSignal;
        }
        return AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal;
    }
    if (requested == AgMt256AsmVariant::
                         V36DynamicExpertNMajorWgm1DirectOutputNChunkSignal &&
        n_tiles > kAgMt256Wgm1MaxNTiles)
    {
        // WGM1 packs the physical N tile into ten task-map bits. The WGM8
        // dynamic scheduler has no corresponding fixed-N gate, so use it as
        // the automatic source-only fallback for wider outputs.
        return AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal;
    }
    if (requested != AgMt256AsmVariant::V15V36Direct)
    {
        return requested;
    }
    // v36 owns the dynamic task-claim workspace and computes its physical
    // N-tile count from the 112-byte contract at launch time.  Keep v15 as
    // an explicit fallback/compare variant, but do not impose a stale fixed
    // tile-count gate on the automatic route.
    return AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal;
}

template <typename T>
__device__ inline float load_as_float(const T &value);

template <>
__device__ inline float load_as_float<__half>(const __half &value)
{
    return __half2float(value);
}

template <>
__device__ inline float load_as_float<__hip_bfloat16>(
    const __hip_bfloat16 &value)
{
    return __bfloat162float(value);
}

template <typename T>
__device__ inline T store_from_float(float value);

template <>
__device__ inline __half store_from_float<__half>(float value)
{
    return __float2half(value);
}

template <>
__device__ inline __hip_bfloat16 store_from_float<__hip_bfloat16>(float value)
{
    return __float2bfloat16(value);
}

using ag_half4_t =
    __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using ag_bf16x4_t =
    __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
using ag_float4_t =
    __attribute__((__vector_size__(4 * sizeof(float)))) float;
using ag_uint4_t =
    __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;

__device__ inline uint16_t ag_float_to_bf16_bits(float value)
{
    uint32_t bits = *reinterpret_cast<uint32_t *>(&value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>(bits >> 16);
}

__device__ inline ag_float4_t ag_mmac_f16(ag_half4_t a,
                                          ag_half4_t b,
                                          ag_float4_t c)
{
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx928__)
    return __builtin_amdgcn_mmac_f32_16x16x16f16(a, b, c);
#else
    return c;
#endif
}

__device__ inline void ag_ds_read_m32x16_b16(ag_uint4_t &dst,
                                             uint32_t byte_offset)
{
    asm volatile("ds_read_m32x16_b16 %0 %1\n\t"
                 : "=v"(dst)
                 : "v"(byte_offset)
                 : "memory");
}

__device__ inline void ag_wait_lgkmcnt0()
{
    asm volatile("s_waitcnt lgkmcnt(0)\n\t" ::: "memory");
}

__device__ inline ag_half4_t ag_half4_from_u32x2(const ag_uint4_t &src,
                                                 int pair)
{
    ag_half4_t out;
    const uint32_t lo = src[pair * 2 + 0];
    const uint32_t hi = src[pair * 2 + 1];
    reinterpret_cast<uint32_t *>(&out)[0] = lo;
    reinterpret_cast<uint32_t *>(&out)[1] = hi;
    return out;
}

__device__ inline ag_bf16x4_t ag_bf16x4_from_u32x2(const ag_uint4_t &src,
                                                   int pair)
{
    ag_bf16x4_t out;
    const uint32_t lo = src[pair * 2 + 0];
    const uint32_t hi = src[pair * 2 + 1];
    reinterpret_cast<uint32_t *>(&out)[0] = lo;
    reinterpret_cast<uint32_t *>(&out)[1] = hi;
    return out;
}

__device__ inline ag_float4_t ag_mmac_bf16(ag_bf16x4_t a,
                                           ag_bf16x4_t b,
                                           ag_float4_t c)
{
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx928__)
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
#else
    return c;
#endif
}

__device__ inline int ag_dsread_m32x16_lane_offset_elems(int lane)
{
    return (lane >> 2) * 32 + (lane & 3) * 8;
}

__device__ inline int ag_dsread_m32x16_b16_fragment_slot(int row16, int k32)
{
    const int kg = k32 & 3;
    const int half = k32 >> 4;
    const int inner = (k32 >> 2) & 3;
    return kg * 128 + half * 16 + inner * 32 + row16;
}

__device__ inline int ag_dsread_m32x16_b16_swizzled_col(int row16, int col32)
{
    constexpr int kVec = 8;
    constexpr int kPerPhase = 4;
    constexpr int kMaxPhase = 2;
    const int phase = (row16 / kPerPhase) % kMaxPhase;
    return ((col32 / kVec) ^ phase) * kVec + (col32 & (kVec - 1));
}

__device__ inline int ag_dsread_m32x16_b16_swizzled_slot(int row16, int col32)
{
    return row16 * 32 +
           ag_dsread_m32x16_b16_swizzled_col(row16, col32);
}

__device__ inline int ag_dsread_m32x16_b16_swizzled_physical_slot(int slot)
{
    const int row = slot >> 5;
    const int col = slot & 31;
    return row * 32 + ag_dsread_m32x16_b16_swizzled_col(row, col);
}

__device__ inline int ag_dsread_m32x16_b16_swizzled_lane_offset_elems(int lane)
{
    return ag_dsread_m32x16_b16_swizzled_physical_slot(
        ag_dsread_m32x16_lane_offset_elems(lane));
}

__device__ inline void ag_wait_signal_ge(const int *signals,
                                         int index,
                                         int signal_value)
{
    const volatile int *signal =
        reinterpret_cast<const volatile int *>(signals + index);
    while (*signal < signal_value)
    {
        __builtin_amdgcn_s_sleep(1);
    }
}

__device__ inline int ag_token_chunk_id(int token,
                                        int full_m,
                                        int world_size,
                                        int signal_chunk_rows,
                                        int chunks_per_rank)
{
    const int local_m = full_m / world_size;
    int owner_rank = token / local_m;
    owner_rank = owner_rank < 0 ? 0 : owner_rank;
    owner_rank = owner_rank >= world_size ? world_size - 1 : owner_rank;
    int local_row = token - owner_rank * local_m;
    local_row = local_row < 0 ? 0 : local_row;
    int chunk = local_row / signal_chunk_rows;
    chunk = chunk < 0 ? 0 : chunk;
    chunk = chunk >= chunks_per_rank ? chunks_per_rank - 1 : chunk;
    return owner_rank * chunks_per_rank + chunk;
}

template <typename T>
__device__ inline void zero_vec4(T &value)
{
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = 0;
    }
}

__device__ inline ag_half4_t load_f16x4_k_contiguous(const _Float16 *ptr,
                                                     int64_t base,
                                                     int k,
                                                     int k_dim)
{
    ag_half4_t value;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = (k + i < k_dim) ? ptr[base + i] : static_cast<_Float16>(0);
    }
    return value;
}

__device__ inline ag_half4_t load_f16x4_weight_kn(const _Float16 *weight,
                                                  int expert,
                                                  int k,
                                                  int n,
                                                  int k_dim,
                                                  int n_dim)
{
    ag_half4_t value;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = (k + i < k_dim && n < n_dim)
                       ? weight[(static_cast<int64_t>(expert) * k_dim + k + i) *
                                    n_dim +
                                n]
                       : static_cast<_Float16>(0);
    }
    return value;
}

__device__ inline ag_half4_t load_f16x4_weight_nk(const _Float16 *weight,
                                                  int expert,
                                                  int k,
                                                  int n,
                                                  int k_dim,
                                                  int n_dim)
{
    ag_half4_t value;
    const int64_t base =
        (static_cast<int64_t>(expert) * n_dim + n) * k_dim + k;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = (k + i < k_dim && n < n_dim)
                       ? weight[base + i]
                       : static_cast<_Float16>(0);
    }
    return value;
}

__device__ inline ag_half4_t load_f16x4_weight_shared_bn(
    const _Float16 *sh_b,
    int k,
    int n,
    int bn)
{
    ag_half4_t value;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = sh_b[(k + i) * bn + n];
    }
    return value;
}

__device__ inline ag_bf16x4_t load_bf16x4_k_contiguous(const uint16_t *ptr,
                                                       int64_t base,
                                                       int k,
                                                       int k_dim)
{
    ag_bf16x4_t value;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = (k + i < k_dim) ? ptr[base + i] : 0;
    }
    return value;
}

__device__ inline ag_bf16x4_t load_bf16x4_weight_kn(const uint16_t *weight,
                                                    int expert,
                                                    int k,
                                                    int n,
                                                    int k_dim,
                                                    int n_dim)
{
    ag_bf16x4_t value;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = (k + i < k_dim && n < n_dim)
                       ? weight[(static_cast<int64_t>(expert) * k_dim + k + i) *
                                    n_dim +
                                n]
                       : 0;
    }
    return value;
}

__device__ inline ag_bf16x4_t load_bf16x4_weight_nk(const uint16_t *weight,
                                                    int expert,
                                                    int k,
                                                    int n,
                                                    int k_dim,
                                                    int n_dim)
{
    ag_bf16x4_t value;
    const int64_t base =
        (static_cast<int64_t>(expert) * n_dim + n) * k_dim + k;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = (k + i < k_dim && n < n_dim) ? weight[base + i] : 0;
    }
    return value;
}

template <typename IndexT>
__device__ inline int load_index(const void *ptr, int offset)
{
    return static_cast<int>(static_cast<const IndexT *>(ptr)[offset]);
}

template <typename T, typename IndexT>
__global__ void ag_group_gemm_reference_kernel(const T *full_input,
                                               const T *weight,
                                               const void *topk_ids,
                                               T *output,
                                               int full_m,
                                               int n_dim,
                                               int k_dim,
                                               int num_experts,
                                               int topk)
{
    const int64_t total = static_cast<int64_t>(full_m) * topk * n_dim;
    const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total)
    {
        return;
    }

    const int n = linear % n_dim;
    const int flat_row = linear / n_dim;
    const int token = flat_row / topk;
    const int expert = load_index<IndexT>(topk_ids, flat_row);
    if (expert < 0 || expert >= num_experts)
    {
        output[linear] = store_from_float<T>(0.0f);
        return;
    }

    float acc = 0.0f;
    const T *x = full_input + static_cast<int64_t>(token) * k_dim;
    const T *w = weight + (static_cast<int64_t>(expert) * k_dim * n_dim) + n;
    for (int k = 0; k < k_dim; ++k)
    {
        acc += load_as_float<T>(x[k]) *
               load_as_float<T>(w[static_cast<int64_t>(k) * n_dim]);
    }
    output[linear] = store_from_float<T>(acc);
}

template <typename IndexT>
__global__ void count_topk_by_expert_rank_kernel(const void *topk_ids,
                                                 int *expert_rank_counts,
                                                 int full_m,
                                                 int topk,
                                                 int num_experts,
                                                 int world_size)
{
    const int total = full_m * topk;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    const int expert = load_index<IndexT>(topk_ids, idx);
    if (expert >= 0 && expert < num_experts)
    {
        const int local_m = full_m / world_size;
        const int token = idx / topk;
        const int owner_rank = token / local_m;
        atomicAdd(expert_rank_counts + expert * world_size + owner_rank, 1);
    }
}

__global__ void build_expert_rank_offsets_kernel(
    const int *expert_rank_counts,
    int *expert_rank_offsets,
    int *global_expert_offsets,
    int *rank_cursors,
    int *ntiles,
    int num_experts,
    int world_size,
    int block_m)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }
    int offset = 0;
    const int rank_stride = world_size + 1;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        global_expert_offsets[expert] = offset;
        int expert_count = 0;
        rank_cursors[expert * rank_stride] = 0;
        for (int rank = 0; rank < world_size; ++rank)
        {
            const int bucket = expert * world_size + rank;
            rank_cursors[expert * rank_stride + rank] = expert_count;
            expert_rank_offsets[bucket] = offset + expert_count;
            expert_count += expert_rank_counts[bucket];
        }
        rank_cursors[expert * rank_stride + world_size] = expert_count;
        offset += ((expert_count + block_m - 1) / block_m) * block_m;
    }
    global_expert_offsets[num_experts] = offset;
    expert_rank_offsets[num_experts * world_size] = offset;
    *ntiles = 0;
}

__global__ void init_sorted_token_ids_kernel(int *sorted_token_ids,
                                             int sorted_capacity)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sorted_capacity)
    {
        sorted_token_ids[idx] = -1;
    }
}

__global__ void use_flat_sorted_rows_as_gather_index_kernel(
    const int *sorted_token_ids,
    int *sorted_gather_index,
    int sorted_capacity)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sorted_capacity)
    {
        sorted_gather_index[idx] = sorted_token_ids[idx];
    }
}

template <typename IndexT>
__global__ void scatter_sorted_token_ids_by_expert_rank_kernel(
    const void *topk_ids,
    int *sorted_token_ids,
    int *sorted_gather_index,
    int *expert_rank_counters,
    const int *expert_rank_offsets,
    int full_m,
    int topk,
    int num_experts,
    int world_size)
{
    const int total = full_m * topk;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    const int expert = load_index<IndexT>(topk_ids, idx);
    if (expert < 0 || expert >= num_experts)
    {
        return;
    }
    const int local_m = full_m / world_size;
    const int token = idx / topk;
    const int owner_rank = token / local_m;
    const int bucket = expert * world_size + owner_rank;
    const int in_bucket = atomicAdd(expert_rank_counters + bucket, 1);
    const int sorted_pos = expert_rank_offsets[bucket] + in_bucket;
    sorted_token_ids[sorted_pos] = idx;
    if (sorted_gather_index != nullptr)
    {
        sorted_gather_index[sorted_pos] = token;
    }
}

// Deterministic alternative to the atomic scatter above.  Each
// (expert, owner-rank) bucket is produced by one lane, preserving token and
// top-k order while keeping the same expert/rank offsets contract.
template <typename IndexT>
__global__ void scatter_sorted_token_ids_by_expert_rank_stable_kernel(
    const void *topk_ids,
    int *sorted_token_ids,
    int *sorted_gather_index,
    const int *expert_rank_offsets,
    int full_m,
    int topk,
    int num_experts,
    int world_size)
{
    const int expert = blockIdx.x;
    const int bucket_rank = threadIdx.x;
    if (expert >= num_experts || bucket_rank >= world_size)
    {
        return;
    }

    const int local_m = full_m / world_size;
    const int token_begin = bucket_rank * local_m;
    const int token_end = bucket_rank + 1 == world_size
                              ? full_m
                              : token_begin + local_m;
    const int bucket = expert * world_size + bucket_rank;
    int sorted_pos = expert_rank_offsets[bucket];
    for (int token = token_begin; token < token_end; ++token)
    {
        const int base = token * topk;
        for (int choice = 0; choice < topk; ++choice)
        {
            const int idx = base + choice;
            if (load_index<IndexT>(topk_ids, idx) != expert)
            {
                continue;
            }
            sorted_token_ids[sorted_pos] = idx;
            if (sorted_gather_index != nullptr)
            {
                sorted_gather_index[sorted_pos] = token;
            }
            ++sorted_pos;
        }
    }
}

__device__ inline int find_rank_segment_in_expert(
    const int *rank_cursors,
    int expert,
    int world_size,
    int rel_offset)
{
    int fallback_rank = 0;
    const int rank_stride = world_size + 1;
    for (int rank = 0; rank < MAX_LOCAL_WORLD_SIZE; ++rank)
    {
        if (rank >= world_size)
        {
            break;
        }
        const int start = rank_cursors[expert * rank_stride + rank];
        const int end = rank_cursors[expert * rank_stride + rank + 1];
        if (end > start)
        {
            fallback_rank = rank;
        }
        if (rel_offset < end)
        {
            return rank;
        }
    }
    return fallback_rank;
}

__device__ inline int ag_moe_tile_stage(int segment_end,
                                        int current_rank,
                                        int world_size,
                                        bool zigzag_order)
{
    const int forward =
        (segment_end - current_rank + world_size) % world_size;
    if (!zigzag_order || forward == 0 || world_size <= 1)
    {
        return forward;
    }
    const int backward = world_size - forward;
    return forward <= backward ? 2 * forward - 1 : 2 * backward;
}

__global__ void build_expert_rank_tile_descriptors_kernel(
    const int *rank_cursors,
    const int *global_expert_offsets,
    int *rank_tile_info,
    int *tile_experts,
    int *tile_starts,
    int *tile_rank_starts,
    int *tile_rank_ends,
    int *ntiles,
    int num_experts,
    int world_size,
    int current_rank,
    bool strict_rank_tile,
    bool split_local_rank,
    int block_m,
    bool zigzag_stage_order)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }
    int tile = 0;
    const int rank_stride = world_size + 1;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        int stage_counts[MAX_LOCAL_WORLD_SIZE];
        int stage_offsets[MAX_LOCAL_WORLD_SIZE];
        for (int stage = 0; stage < MAX_LOCAL_WORLD_SIZE; ++stage)
        {
            stage_counts[stage] = 0;
            stage_offsets[stage] = tile;
        }

        const int expert_base = global_expert_offsets[expert];
        const int expert_rows =
            rank_cursors[expert * rank_stride + world_size];

        if (!strict_rank_tile)
        {
            for (int rel_start = 0; rel_start < expert_rows;
                 rel_start += block_m)
            {
                const int remaining = expert_rows - rel_start;
                const int rows = remaining < block_m ? remaining : block_m;
                const int tile_end = rel_start + rows;
                const int tile_rank_start = find_rank_segment_in_expert(
                    rank_cursors, expert, world_size, rel_start);
                const int tile_rank_end = find_rank_segment_in_expert(
                    rank_cursors, expert, world_size, tile_end - 1);
                const bool split_tile =
                    split_local_rank && tile_rank_start <= current_rank &&
                    current_rank <= tile_rank_end &&
                    tile_rank_start != tile_rank_end;
                if (split_tile)
                {
                    int cursor = rel_start;
                    while (cursor < tile_end)
                    {
                        const int owner_rank = find_rank_segment_in_expert(
                            rank_cursors, expert, world_size, cursor);
                        const int rank_end_offset =
                            rank_cursors[expert * rank_stride + owner_rank + 1];
                        const int segment_end =
                            rank_end_offset > cursor && rank_end_offset < tile_end
                                ? rank_end_offset
                                : tile_end;
                        const int stage = ag_moe_tile_stage(
                            owner_rank,
                            current_rank,
                            world_size,
                            zigzag_stage_order);
                        stage_counts[stage] += 1;
                        cursor = segment_end;
                    }
                }
                else
                {
                    const int rank_end = find_rank_segment_in_expert(
                        rank_cursors,
                        expert,
                        world_size,
                        rel_start + rows - 1);
                    const int stage = ag_moe_tile_stage(
                        rank_end,
                        current_rank,
                        world_size,
                        zigzag_stage_order);
                    stage_counts[stage] += 1;
                }
            }

            int stage_running = tile;
            for (int stage = 0; stage < world_size; ++stage)
            {
                stage_offsets[stage] = stage_running;
                stage_running += stage_counts[stage];
                stage_counts[stage] = 0;
            }

            for (int rel_start = 0; rel_start < expert_rows;
                 rel_start += block_m)
            {
                const int remaining = expert_rows - rel_start;
                const int rows = remaining < block_m ? remaining : block_m;
                const int tile_end = rel_start + rows;
                const int tile_rank_start = find_rank_segment_in_expert(
                    rank_cursors, expert, world_size, rel_start);
                const int tile_rank_end = find_rank_segment_in_expert(
                    rank_cursors, expert, world_size, tile_end - 1);
                const bool split_tile =
                    split_local_rank && tile_rank_start <= current_rank &&
                    current_rank <= tile_rank_end &&
                    tile_rank_start != tile_rank_end;
                if (split_tile)
                {
                    int cursor = rel_start;
                    while (cursor < tile_end)
                    {
                        const int owner_rank = find_rank_segment_in_expert(
                            rank_cursors, expert, world_size, cursor);
                        const int rank_end_offset =
                            rank_cursors[expert * rank_stride + owner_rank + 1];
                        const int segment_end =
                            rank_end_offset > cursor && rank_end_offset < tile_end
                                ? rank_end_offset
                                : tile_end;
                        const int piece_rows = segment_end - cursor;
                        const int stage = ag_moe_tile_stage(
                            owner_rank,
                            current_rank,
                            world_size,
                            zigzag_stage_order);
                        const int out_tile =
                            stage_offsets[stage] + stage_counts[stage]++;
                        int *info = rank_tile_info +
                                    out_tile * AG_RANK_TILE_INFO_STRIDE;
                        info[AG_RANK_TILE_EXPERT] = expert;
                        info[AG_RANK_TILE_SORTED_START] = expert_base + cursor;
                        info[AG_RANK_TILE_ROWS] = piece_rows;
                        info[AG_RANK_TILE_RANK_START] = owner_rank;
                        info[AG_RANK_TILE_RANK_END] = owner_rank;
                        tile_experts[out_tile] = expert;
                        tile_starts[out_tile] = expert_base + cursor;
                        tile_rank_starts[out_tile] = owner_rank;
                        tile_rank_ends[out_tile] = owner_rank;
                        cursor = segment_end;
                    }
                }
                else
                {
                    const int rank_start = find_rank_segment_in_expert(
                        rank_cursors, expert, world_size, rel_start);
                    const int rank_end = find_rank_segment_in_expert(
                        rank_cursors,
                        expert,
                        world_size,
                        rel_start + rows - 1);
                    const int stage = ag_moe_tile_stage(
                        rank_end,
                        current_rank,
                        world_size,
                        zigzag_stage_order);
                    const int out_tile =
                        stage_offsets[stage] + stage_counts[stage]++;
                    int *info = rank_tile_info +
                                out_tile * AG_RANK_TILE_INFO_STRIDE;
                    info[AG_RANK_TILE_EXPERT] = expert;
                    info[AG_RANK_TILE_SORTED_START] = expert_base + rel_start;
                    info[AG_RANK_TILE_ROWS] = rows;
                    info[AG_RANK_TILE_RANK_START] = rank_start;
                    info[AG_RANK_TILE_RANK_END] = rank_end;
                    tile_experts[out_tile] = expert;
                    tile_starts[out_tile] = expert_base + rel_start;
                    tile_rank_starts[out_tile] = rank_start;
                    tile_rank_ends[out_tile] = rank_end;
                }
            }
            tile = stage_running;
            continue;
        }

        for (int owner_rank = 0; owner_rank < MAX_LOCAL_WORLD_SIZE; ++owner_rank)
        {
            if (owner_rank >= world_size)
            {
                break;
            }
            const int rank_start =
                rank_cursors[expert * rank_stride + owner_rank];
            const int rank_end =
                rank_cursors[expert * rank_stride + owner_rank + 1];
            const int rank_rows = rank_end - rank_start;
            if (rank_rows <= 0)
            {
                continue;
            }
            const int stage =
                ag_moe_tile_stage(owner_rank,
                                  current_rank,
                                  world_size,
                                  zigzag_stage_order);
            stage_counts[stage] += (rank_rows + block_m - 1) / block_m;
        }

        int stage_running = tile;
        for (int stage = 0; stage < world_size; ++stage)
        {
            stage_offsets[stage] = stage_running;
            stage_running += stage_counts[stage];
            stage_counts[stage] = 0;
        }

        for (int owner_rank = 0; owner_rank < MAX_LOCAL_WORLD_SIZE; ++owner_rank)
        {
            if (owner_rank >= world_size)
            {
                break;
            }
            const int rank_start =
                rank_cursors[expert * rank_stride + owner_rank];
            const int rank_end =
                rank_cursors[expert * rank_stride + owner_rank + 1];
            const int rank_rows = rank_end - rank_start;
            if (rank_rows <= 0)
            {
                continue;
            }
            const int stage =
                ag_moe_tile_stage(owner_rank,
                                  current_rank,
                                  world_size,
                                  zigzag_stage_order);
            const int tiles_this_rank = (rank_rows + block_m - 1) / block_m;
            for (int t = 0; t < tiles_this_rank; ++t)
            {
                const int rel_start = rank_start + t * block_m;
                const int remaining = rank_rows - t * block_m;
                const int rows =
                    remaining < block_m ? remaining : block_m;
                const int out_tile =
                    stage_offsets[stage] + stage_counts[stage]++;
                int *info =
                    rank_tile_info + out_tile * AG_RANK_TILE_INFO_STRIDE;
                info[AG_RANK_TILE_EXPERT] = expert;
                info[AG_RANK_TILE_SORTED_START] = expert_base + rel_start;
                info[AG_RANK_TILE_ROWS] = rows;
                info[AG_RANK_TILE_RANK_START] = owner_rank;
                info[AG_RANK_TILE_RANK_END] = owner_rank;
                tile_experts[out_tile] = expert;
                tile_starts[out_tile] = expert_base + rel_start;
                tile_rank_starts[out_tile] = owner_rank;
                tile_rank_ends[out_tile] = owner_rank;
            }
        }
        tile = stage_running;
    }
    *ntiles = tile;
}

__global__ void build_rank_tile_problem_info_kernel(
    const int *rank_tile_info,
    const int *sorted_gather_index,
    const int *ntiles,
    int *rank_tile_problem_info,
    int full_m,
    int world_size,
    int signal_chunk_rows,
    int chunks_per_rank,
    int rank_tile_capacity,
    bool nn_direct_chunk_signal)
{
    const int tile = blockIdx.x;
    if (tile >= rank_tile_capacity || tile >= *ntiles)
    {
        return;
    }

    const int *src = rank_tile_info + tile * AG_RANK_TILE_INFO_STRIDE;
    int *dst = rank_tile_problem_info +
               tile * AG_RANK_TILE_PROBLEM_INFO_STRIDE;
    if (threadIdx.x < AG_RANK_TILE_INFO_STRIDE)
    {
        dst[threadIdx.x] = src[threadIdx.x];
    }
    if (threadIdx.x == 0)
    {
        dst[AG_RANK_TILE_PROBLEM_CHUNK_START] = -1;
        dst[AG_RANK_TILE_PROBLEM_CHUNK_END] = -1;
        dst[AG_RANK_TILE_PROBLEM_RESERVED] =
            nn_direct_chunk_signal ? 0x80 : 0;
    }
    __syncthreads();

    if (signal_chunk_rows <= 0 || chunks_per_rank <= 0 ||
        sorted_gather_index == nullptr)
    {
        return;
    }

    const int sorted_start = src[AG_RANK_TILE_SORTED_START];
    const int tile_rows = src[AG_RANK_TILE_ROWS];
    int local_min = world_size * chunks_per_rank;
    int local_max = -1;
    for (int row = threadIdx.x; row < tile_rows; row += blockDim.x)
    {
        const int token = sorted_gather_index[sorted_start + row];
        if (token >= 0 && token < full_m)
        {
            const int chunk_id = ag_token_chunk_id(token,
                                                   full_m,
                                                   world_size,
                                                   signal_chunk_rows,
                                                   chunks_per_rank);
            local_min = chunk_id < local_min ? chunk_id : local_min;
            local_max = chunk_id > local_max ? chunk_id : local_max;
        }
    }

    __shared__ int chunk_min;
    __shared__ int chunk_max;
    if (threadIdx.x == 0)
    {
        chunk_min = world_size * chunks_per_rank;
        chunk_max = -1;
    }
    __syncthreads();
    atomicMin(&chunk_min, local_min);
    atomicMax(&chunk_max, local_max);
    __syncthreads();
    if (threadIdx.x == 0 && chunk_max >= 0)
    {
        dst[AG_RANK_TILE_PROBLEM_CHUNK_START] = chunk_min;
        dst[AG_RANK_TILE_PROBLEM_CHUNK_END] = chunk_max;
    }
}

template <typename T>
__launch_bounds__(256, 1) __global__
    void ag_group_gemm_tiled_128x256x32_kernel(const T *full_input,
                                               const T *weight,
                                               const int *sorted_token_ids,
                                               const int *tile_experts,
                                               const int *tile_starts,
                                               const int *tile_rank_starts,
                                               const int *tile_rank_ends,
                                               const int *ntiles,
                                               const int *rank_signals,
                                               T *output,
                                               int full_m,
                                               int n_dim,
                                               int k_dim,
                                               int topk,
                                               int world_size,
                                               int signal_value)
{
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 32;
    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int expert = tile_experts[tile_m];
    const int sorted_start = tile_starts[tile_m];
    const int n_base = tile_n * BN;

    if (rank_signals != nullptr)
    {
        const int segment_start = tile_rank_starts[tile_m];
        const int segment_end = tile_rank_ends[tile_m];
        if (segment_start >= 0 && segment_end < world_size)
        {
            if (threadIdx.x == 0)
            {
                for (int src_rank = segment_start; src_rank <= segment_end;
                     ++src_rank)
                {
                    const volatile int *signal =
                        reinterpret_cast<const volatile int *>(rank_signals +
                                                               src_rank);
                    while (*signal < signal_value)
                    {
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int linear = threadIdx.x; linear < BM * BN; linear += blockDim.x)
    {
        const int row_in_tile = linear / BN;
        const int n = n_base + (linear - row_in_tile * BN);
        if (n >= n_dim)
        {
            continue;
        }

        const int flat_row = sorted_token_ids[sorted_start + row_in_tile];
        if (flat_row < 0 || flat_row >= full_m * topk)
        {
            continue;
        }

        const int token = flat_row / topk;

        float acc = 0.0f;
        for (int k0 = 0; k0 < k_dim; k0 += BK)
        {
            const int kend = (k0 + BK < k_dim) ? (k0 + BK) : k_dim;
            for (int k = k0; k < kend; ++k)
            {
                const T a = full_input[static_cast<int64_t>(token) * k_dim + k];
                const T b =
                    weight[(static_cast<int64_t>(expert) * k_dim + k) *
                               n_dim +
                           n];
                acc += load_as_float<T>(a) * load_as_float<T>(b);
            }
        }
        output[static_cast<int64_t>(flat_row) * n_dim + n] =
            store_from_float<T>(acc);
    }
}

template <typename T>
__launch_bounds__(256, 1) __global__
    void ag_group_gemm_tiled_128x256x32_smem64x128_kernel(
        const T *full_input,
        const T *weight,
        const int *sorted_token_ids,
        const int *tile_experts,
        const int *tile_starts,
        const int *tile_rank_starts,
        const int *tile_rank_ends,
        const int *ntiles,
        const int *rank_signals,
        T *output,
        int full_m,
        int n_dim,
        int k_dim,
        int topk,
        int world_size,
        int signal_value)
{
    constexpr int SUB_BM = 64;
    constexpr int SUB_BN = 128;
    constexpr int BK = 32;
    constexpr int COLS_PER_THREAD = 32;

    const int outer_tile_m = blockIdx.x / 2;
    const int row_group = blockIdx.x - outer_tile_m * 2;
    const int n_base = blockIdx.y * SUB_BN;
    if (outer_tile_m >= *ntiles)
    {
        return;
    }

    const int expert = tile_experts[outer_tile_m];
    const int sorted_start = tile_starts[outer_tile_m] + row_group * SUB_BM;
    const int tid = threadIdx.x;
    const int row = tid / 4;
    const int col_base = (tid - row * 4) * COLS_PER_THREAD;

    if (rank_signals != nullptr)
    {
        const int segment_start = tile_rank_starts[outer_tile_m];
        const int segment_end = tile_rank_ends[outer_tile_m];
        if (segment_start >= 0 && segment_end < world_size)
        {
            if (tid == 0)
            {
                for (int src_rank = segment_start; src_rank <= segment_end;
                     ++src_rank)
                {
                    const volatile int *signal =
                        reinterpret_cast<const volatile int *>(rank_signals +
                                                               src_rank);
                    while (*signal < signal_value)
                    {
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
            __syncthreads();
        }
    }

    __shared__ T sh_a[SUB_BM * BK];
    __shared__ T sh_b[BK * SUB_BN];

    float acc[COLS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < COLS_PER_THREAD; ++i)
    {
        acc[i] = 0.0f;
    }

    const int flat_row = sorted_token_ids[sorted_start + row];
    const bool valid_row = flat_row >= 0 && flat_row < full_m * topk;

    for (int k0 = 0; k0 < k_dim; k0 += BK)
    {
        for (int idx = tid; idx < SUB_BM * BK; idx += blockDim.x)
        {
            const int load_row = idx / BK;
            const int load_k = idx - load_row * BK;
            const int load_flat_row = sorted_token_ids[sorted_start + load_row];
            T value = store_from_float<T>(0.0f);
            if (load_flat_row >= 0 && load_flat_row < full_m * topk)
            {
                const int load_token = load_flat_row / topk;
                const int k = k0 + load_k;
                if (k < k_dim)
                {
                    value =
                        full_input[static_cast<int64_t>(load_token) * k_dim + k];
                }
            }
            sh_a[idx] = value;
        }

        for (int idx = tid; idx < BK * SUB_BN; idx += blockDim.x)
        {
            const int load_k = idx / SUB_BN;
            const int load_n = idx - load_k * SUB_BN;
            T value = store_from_float<T>(0.0f);
            const int k = k0 + load_k;
            const int n = n_base + load_n;
            if (k < k_dim && n < n_dim)
            {
                value = weight[(static_cast<int64_t>(expert) * k_dim + k) *
                                   n_dim +
                               n];
            }
            sh_b[idx] = value;
        }

        __syncthreads();

        if (valid_row)
        {
#pragma unroll
            for (int kk = 0; kk < BK; ++kk)
            {
                const int k = k0 + kk;
                if (k >= k_dim)
                {
                    break;
                }
                const float a = load_as_float<T>(sh_a[row * BK + kk]);
#pragma unroll
                for (int j = 0; j < COLS_PER_THREAD; ++j)
                {
                    const int col = col_base + j;
                    acc[j] +=
                        a * load_as_float<T>(sh_b[kk * SUB_BN + col]);
                }
            }
        }

        __syncthreads();
    }

    if (valid_row)
    {
#pragma unroll
        for (int j = 0; j < COLS_PER_THREAD; ++j)
        {
            const int n = n_base + col_base + j;
            if (n < n_dim)
            {
                output[static_cast<int64_t>(flat_row) * n_dim + n] =
                    store_from_float<T>(acc[j]);
            }
        }
    }
}

__launch_bounds__(1024, 1) __global__
    void ag_group_gemm_tiled_128x128x16_lds_mmac_f16_kernel(
        const _Float16 *full_input,
        const _Float16 *weight,
        const int *sorted_token_ids,
        const int *tile_experts,
        const int *tile_starts,
        const int *tile_rank_starts,
        const int *tile_rank_ends,
        const int *ntiles,
        const int *rank_signals,
        _Float16 *output,
        int full_m,
        int n_dim,
        int k_dim,
        int topk,
        int world_size,
        int signal_value)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int WM = 32;
    constexpr int WN = 32;
    constexpr int BK = 32;
    constexpr int MMAC_K = 16;
    constexpr int WARP_SIZE = 64;
    constexpr int WAVES_PER_BLOCK = 16;
    constexpr int A_LOAD_VEC = 4;
    constexpr int A_LOADS = BM * (BK / A_LOAD_VEC);
    constexpr int LDS_A_ELEMS = BK * BM;
    constexpr int LDS_B_ELEMS = BK * BN;

    const int tile_m = blockIdx.x;
    const int n_base = blockIdx.y * BN;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int wave_idx = tid / WARP_SIZE;
    const int lane = tid - wave_idx * WARP_SIZE;
    const int wave_m = wave_idx >> 2;
    const int wave_n = wave_idx & 3;
    const int row_base = wave_m * WM;
    const int col_base = n_base + wave_n * WN;
    const int lane_row16 = lane & 15;
    const int lane_col4 = lane >> 4;

    const int expert = tile_experts[tile_m];
    const int sorted_start = tile_starts[tile_m];

    if (rank_signals != nullptr)
    {
        const int segment_start = tile_rank_starts[tile_m];
        const int segment_end = tile_rank_ends[tile_m];
        if (segment_start >= 0 && segment_end < world_size)
        {
            if (tid == 0)
            {
                for (int src_rank = segment_start; src_rank <= segment_end;
                     ++src_rank)
                {
                    const volatile int *signal =
                        reinterpret_cast<const volatile int *>(rank_signals +
                                                               src_rank);
                    while (*signal < signal_value)
                    {
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
            __syncthreads();
        }
    }

    __shared__ _Float16 shmem[LDS_A_ELEMS + LDS_B_ELEMS];
    _Float16 *sh_a = shmem;
    _Float16 *sh_b = shmem + LDS_A_ELEMS;

    ag_float4_t acc00;
    ag_float4_t acc01;
    ag_float4_t acc10;
    ag_float4_t acc11;
    zero_vec4(acc00);
    zero_vec4(acc01);
    zero_vec4(acc10);
    zero_vec4(acc11);

    for (int k0 = 0; k0 < k_dim; k0 += BK)
    {
        if (tid < A_LOADS)
        {
            const int m = tid / (BK / A_LOAD_VEC);
            const int kk = (tid - m * (BK / A_LOAD_VEC)) * A_LOAD_VEC;
            const int flat_row_a = sorted_token_ids[sorted_start + m];
            const bool valid_a = flat_row_a >= 0 && flat_row_a < full_m * topk;
            const int token_a = valid_a ? (flat_row_a / topk) : 0;
            const int gk = k0 + kk;
            ag_half4_t value;
            zero_vec4(value);
            if (valid_a && gk < k_dim)
            {
                value = load_f16x4_k_contiguous(
                    full_input,
                    static_cast<int64_t>(token_a) * k_dim + gk,
                    gk,
                    k_dim);
            }
#pragma unroll
            for (int i = 0; i < A_LOAD_VEC; ++i)
            {
                sh_a[(kk + i) * BM + m] = value[i];
            }
        }

        for (int idx = tid; idx < LDS_B_ELEMS; idx += blockDim.x)
        {
            const int kk = idx / BN;
            const int n_off = idx - kk * BN;
            const int n = n_base + n_off;
            _Float16 value = static_cast<_Float16>(0);
            if (k0 + kk < k_dim && n < n_dim)
            {
                value = weight[(static_cast<int64_t>(expert) * k_dim + k0 +
                                kk) *
                                   n_dim +
                               n];
            }
            sh_b[idx] = value;
        }

        __syncthreads();

        const int lane_k = lane / 4;
        const int lane_group = (lane & 3) * 8;
        const uint32_t a_lds_offset0 =
            static_cast<uint32_t>((lane_k * BM + row_base + lane_group) *
                                  sizeof(_Float16));
        const uint32_t b_lds_offset0 =
            static_cast<uint32_t>((LDS_A_ELEMS + lane_k * BN +
                                   wave_n * WN + lane_group) *
                                  sizeof(_Float16));
        const uint32_t a_lds_offset1 =
            static_cast<uint32_t>(((MMAC_K + lane_k) * BM + row_base +
                                   lane_group) *
                                  sizeof(_Float16));
        const uint32_t b_lds_offset1 =
            static_cast<uint32_t>((LDS_A_ELEMS + (MMAC_K + lane_k) * BN +
                                   wave_n * WN + lane_group) *
                                  sizeof(_Float16));

        ag_uint4_t ra0;
        ag_uint4_t rb0;
        ag_uint4_t ra1;
        ag_uint4_t rb1;
        ag_ds_read_m32x16_b16(ra0, a_lds_offset0);
        ag_ds_read_m32x16_b16(rb0, b_lds_offset0);
        ag_ds_read_m32x16_b16(ra1, a_lds_offset1);
        ag_ds_read_m32x16_b16(rb1, b_lds_offset1);
        ag_wait_lgkmcnt0();

        const ag_half4_t a00 = ag_half4_from_u32x2(ra0, 0);
        const ag_half4_t a01 = ag_half4_from_u32x2(ra0, 1);
        const ag_half4_t b00 = ag_half4_from_u32x2(rb0, 0);
        const ag_half4_t b01 = ag_half4_from_u32x2(rb0, 1);
        acc00 = ag_mmac_f16(a00, b00, acc00);
        acc01 = ag_mmac_f16(a00, b01, acc01);
        acc10 = ag_mmac_f16(a01, b00, acc10);
        acc11 = ag_mmac_f16(a01, b01, acc11);

        const ag_half4_t a10 = ag_half4_from_u32x2(ra1, 0);
        const ag_half4_t a11 = ag_half4_from_u32x2(ra1, 1);
        const ag_half4_t b10 = ag_half4_from_u32x2(rb1, 0);
        const ag_half4_t b11 = ag_half4_from_u32x2(rb1, 1);
        acc00 = ag_mmac_f16(a10, b10, acc00);
        acc01 = ag_mmac_f16(a10, b11, acc01);
        acc10 = ag_mmac_f16(a11, b10, acc10);
        acc11 = ag_mmac_f16(a11, b11, acc11);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const int col0 = col_base + i * 4 + lane_col4;
        const int row0 = row_base + lane_row16;
        const int row1 = row0 + 16;
        const int flat0 = sorted_token_ids[sorted_start + row0];
        const int flat1 = sorted_token_ids[sorted_start + row1];
        if (flat0 >= 0 && flat0 < full_m * topk && col0 < n_dim)
        {
            output[static_cast<int64_t>(flat0) * n_dim + col0] =
                static_cast<_Float16>(acc00[i]);
        }
        if (flat0 >= 0 && flat0 < full_m * topk && col0 + 16 < n_dim)
        {
            output[static_cast<int64_t>(flat0) * n_dim + col0 + 16] =
                static_cast<_Float16>(acc01[i]);
        }
        if (flat1 >= 0 && flat1 < full_m * topk && col0 < n_dim)
        {
            output[static_cast<int64_t>(flat1) * n_dim + col0] =
                static_cast<_Float16>(acc10[i]);
        }
        if (flat1 >= 0 && flat1 < full_m * topk && col0 + 16 < n_dim)
        {
            output[static_cast<int64_t>(flat1) * n_dim + col0 + 16] =
                static_cast<_Float16>(acc11[i]);
        }
    }
}

__launch_bounds__(512, 1) __global__
    void ag_group_gemm_tiled_128x128x32_mmac_f16_kernel(
        const _Float16 *full_input,
        const _Float16 *weight,
        const int *sorted_token_ids,
        const int *tile_experts,
        const int *tile_starts,
        const int *tile_rank_starts,
        const int *tile_rank_ends,
        const int *ntiles,
        const int *rank_signals,
        _Float16 *output,
        int full_m,
        int n_dim,
        int k_dim,
        int topk,
        int world_size,
        bool weight_nk,
        int signal_value,
        bool stage_b)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int WARP_SIZE = 64;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int NLOOP = BN / 16;

    const int tile_m = blockIdx.x;
    const int n_base = blockIdx.y * BN;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int warp_idx = tid / WARP_SIZE;
    const int lane = tid - warp_idx * WARP_SIZE;
    const int rowid = lane & 15;
    const int col_lane = lane >> 4;
    const int expert = tile_experts[tile_m];
    const int sorted_start = tile_starts[tile_m];
    const int row_in_tile = warp_idx * 16 + rowid;
    const int flat_row = sorted_token_ids[sorted_start + row_in_tile];
    const bool valid_row = flat_row >= 0 && flat_row < full_m * topk;
    const int token = valid_row ? (flat_row / topk) : 0;
    stage_b = stage_b && !weight_nk;

    __shared__ _Float16 sh_b[BK * BN];

    if (rank_signals != nullptr)
    {
        const int segment_start = tile_rank_starts[tile_m];
        const int segment_end = tile_rank_ends[tile_m];
        if (segment_start >= 0 && segment_end < world_size)
        {
            if (tid == 0)
            {
                for (int src_rank = segment_start; src_rank <= segment_end;
                     ++src_rank)
                {
                    const volatile int *signal =
                        reinterpret_cast<const volatile int *>(rank_signals +
                                                               src_rank);
                    while (*signal < signal_value)
                    {
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
            __syncthreads();
        }
    }

    ag_float4_t acc[NLOOP];
#pragma unroll
    for (int nn = 0; nn < NLOOP; ++nn)
    {
        zero_vec4(acc[nn]);
    }

    const int64_t a_base = static_cast<int64_t>(token) * k_dim;
    for (int k0 = 0; k0 < k_dim; k0 += BK)
    {
        if (stage_b)
        {
            for (int idx = tid; idx < BK * BN; idx += blockDim.x)
            {
                const int kk = idx / BN;
                const int n_off = idx - kk * BN;
                const int gk = k0 + kk;
                const int n = n_base + n_off;
                _Float16 value = static_cast<_Float16>(0);
                if (gk < k_dim && n < n_dim)
                {
                    value =
                        weight[(static_cast<int64_t>(expert) * k_dim + gk) *
                                   n_dim +
                               n];
                }
                sh_b[idx] = value;
            }
            __syncthreads();
        }

        const int k_lane = k0 + col_lane * 8;
        ag_half4_t a0;
        ag_half4_t a1;
        zero_vec4(a0);
        zero_vec4(a1);
        if (valid_row)
        {
            a0 = load_f16x4_k_contiguous(full_input, a_base + k_lane, k_lane, k_dim);
            a1 = load_f16x4_k_contiguous(
                full_input, a_base + k_lane + 4, k_lane + 4, k_dim);
        }

#pragma unroll
        for (int nn = 0; nn < NLOOP; ++nn)
        {
            const int b_col = n_base + nn * 16 + rowid;
            ag_half4_t b0;
            ag_half4_t b1;
            if (stage_b)
            {
                const int sh_n = nn * 16 + rowid;
                const int sh_k = col_lane * 8;
                b0 = load_f16x4_weight_shared_bn(sh_b, sh_k, sh_n, BN);
                b1 = load_f16x4_weight_shared_bn(sh_b, sh_k + 4, sh_n, BN);
            }
            else
            {
                b0 = weight_nk ? load_f16x4_weight_nk(weight,
                                                      expert,
                                                      k_lane,
                                                      b_col,
                                                      k_dim,
                                                      n_dim)
                               : load_f16x4_weight_kn(weight,
                                                      expert,
                                                      k_lane,
                                                      b_col,
                                                      k_dim,
                                                      n_dim);
                b1 = weight_nk ? load_f16x4_weight_nk(weight,
                                                      expert,
                                                      k_lane + 4,
                                                      b_col,
                                                      k_dim,
                                                      n_dim)
                               : load_f16x4_weight_kn(weight,
                                                      expert,
                                                      k_lane + 4,
                                                      b_col,
                                                      k_dim,
                                                      n_dim);
            }
            acc[nn] = ag_mmac_f16(a0, b0, acc[nn]);
            acc[nn] = ag_mmac_f16(a1, b1, acc[nn]);
        }
        if (stage_b)
        {
            __syncthreads();
        }
    }

    if (valid_row)
    {
#pragma unroll
        for (int nn = 0; nn < NLOOP; ++nn)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int n = n_base + nn * 16 + col_lane + i * 4;
                if (n < n_dim)
                {
                    output[static_cast<int64_t>(flat_row) * n_dim + n] =
                        static_cast<_Float16>(acc[nn][i]);
                }
            }
        }
    }
}

__launch_bounds__(512, 1) __global__
    void ag_group_gemm_tiled_128x128x32_mmac_bf16_kernel(
        const uint16_t *full_input,
        const uint16_t *weight,
        const int *sorted_token_ids,
        const int *tile_experts,
        const int *tile_starts,
        const int *tile_rank_starts,
        const int *tile_rank_ends,
        const int *ntiles,
        const int *rank_signals,
        uint16_t *output,
        int full_m,
        int n_dim,
        int k_dim,
        int topk,
        int world_size,
        bool weight_nk,
        int signal_value)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int WARP_SIZE = 64;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int NLOOP = BN / 16;

    const int tile_m = blockIdx.x;
    const int n_base = blockIdx.y * BN;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int warp_idx = tid / WARP_SIZE;
    const int lane = tid - warp_idx * WARP_SIZE;
    const int rowid = lane & 15;
    const int col_lane = lane >> 4;
    const int expert = tile_experts[tile_m];
    const int sorted_start = tile_starts[tile_m];
    const int row_in_tile = warp_idx * 16 + rowid;
    const int flat_row = sorted_token_ids[sorted_start + row_in_tile];
    const bool valid_row = flat_row >= 0 && flat_row < full_m * topk;
    const int token = valid_row ? (flat_row / topk) : 0;

    if (rank_signals != nullptr)
    {
        const int segment_start = tile_rank_starts[tile_m];
        const int segment_end = tile_rank_ends[tile_m];
        if (segment_start >= 0 && segment_end < world_size)
        {
            if (tid == 0)
            {
                for (int src_rank = segment_start; src_rank <= segment_end;
                     ++src_rank)
                {
                    const volatile int *signal =
                        reinterpret_cast<const volatile int *>(rank_signals +
                                                               src_rank);
                    while (*signal < signal_value)
                    {
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
            __syncthreads();
        }
    }

    ag_float4_t acc[NLOOP];
#pragma unroll
    for (int nn = 0; nn < NLOOP; ++nn)
    {
        zero_vec4(acc[nn]);
    }

    const int64_t a_base = static_cast<int64_t>(token) * k_dim;
    for (int k0 = 0; k0 < k_dim; k0 += 32)
    {
        const int k_lane = k0 + col_lane * 8;
        ag_bf16x4_t a0;
        ag_bf16x4_t a1;
        zero_vec4(a0);
        zero_vec4(a1);
        if (valid_row)
        {
            a0 = load_bf16x4_k_contiguous(
                full_input, a_base + k_lane, k_lane, k_dim);
            a1 = load_bf16x4_k_contiguous(
                full_input, a_base + k_lane + 4, k_lane + 4, k_dim);
        }

#pragma unroll
        for (int nn = 0; nn < NLOOP; ++nn)
        {
            const int b_col = n_base + nn * 16 + rowid;
            ag_bf16x4_t b0 =
                weight_nk ? load_bf16x4_weight_nk(
                                weight, expert, k_lane, b_col, k_dim, n_dim)
                          : load_bf16x4_weight_kn(
                                weight, expert, k_lane, b_col, k_dim, n_dim);
            ag_bf16x4_t b1 =
                weight_nk ? load_bf16x4_weight_nk(weight,
                                                  expert,
                                                  k_lane + 4,
                                                  b_col,
                                                  k_dim,
                                                  n_dim)
                          : load_bf16x4_weight_kn(weight,
                                                  expert,
                                                  k_lane + 4,
                                                  b_col,
                                                  k_dim,
                                                  n_dim);
            acc[nn] = ag_mmac_bf16(a0, b0, acc[nn]);
            acc[nn] = ag_mmac_bf16(a1, b1, acc[nn]);
        }
    }

    if (valid_row)
    {
#pragma unroll
        for (int nn = 0; nn < NLOOP; ++nn)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int n = n_base + nn * 16 + col_lane + i * 4;
                if (n < n_dim)
                {
                    output[static_cast<int64_t>(flat_row) * n_dim + n] =
                        ag_float_to_bf16_bits(acc[nn][i]);
                }
            }
        }
    }
}

__launch_bounds__(512, 1) __global__
    void ag_group_gemm_tiled_128x128x32_mmac_bf16_rank_tile_kernel(
        const uint16_t *full_input,
        const uint16_t *weight,
        const int *sorted_token_ids,
        const int *sorted_gather_index,
        const int *rank_tile_info,
        const int *rank_tile_problem_info,
        const int *ntiles,
        const int *rank_signals,
        uint16_t *output,
        int full_m,
        int n_dim,
        int k_dim,
        int topk,
        int world_size,
        int signal_value,
        int signal_chunk_rows,
        int chunks_per_rank)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int WARP_SIZE = 64;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int NLOOP = BN / 16;

    const int tile_m = blockIdx.x;
    const int n_base = blockIdx.y * BN;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int warp_idx = tid / WARP_SIZE;
    const int lane = tid - warp_idx * WARP_SIZE;
    const int rowid = lane & 15;
    const int col_lane = lane >> 4;

    const int *tile_info =
        rank_tile_problem_info != nullptr
            ? rank_tile_problem_info +
                  tile_m * AG_RANK_TILE_PROBLEM_INFO_STRIDE
            : rank_tile_info + tile_m * AG_RANK_TILE_INFO_STRIDE;
    const int expert = tile_info[AG_RANK_TILE_EXPERT];
    const int sorted_start = tile_info[AG_RANK_TILE_SORTED_START];
    const int tile_rows = tile_info[AG_RANK_TILE_ROWS];

    if (rank_signals != nullptr)
    {
        const bool use_chunk_signal =
            signal_chunk_rows > 0 && chunks_per_rank > 0;
        if (use_chunk_signal)
        {
            if (rank_tile_problem_info != nullptr)
            {
                const int chunk_start =
                    tile_info[AG_RANK_TILE_PROBLEM_CHUNK_START];
                const int chunk_end = tile_info[AG_RANK_TILE_PROBLEM_CHUNK_END];
                if (chunk_start >= 0 && chunk_end >= chunk_start)
                {
                    for (int chunk = chunk_start + tid; chunk <= chunk_end;
                         chunk += blockDim.x)
                    {
                        ag_wait_signal_ge(rank_signals, chunk, signal_value);
                    }
                }
            }
            else
            {
                for (int row = tid; row < tile_rows; row += blockDim.x)
                {
                    const int token = sorted_gather_index[sorted_start + row];
                    if (token >= 0 && token < full_m)
                    {
                        const int chunk_id =
                            ag_token_chunk_id(token,
                                              full_m,
                                              world_size,
                                              signal_chunk_rows,
                                              chunks_per_rank);
                        ag_wait_signal_ge(rank_signals, chunk_id, signal_value);
                    }
                }
            }
            __syncthreads();
        }
        else
        {
            const int segment_start = tile_info[AG_RANK_TILE_RANK_START];
            const int segment_end = tile_info[AG_RANK_TILE_RANK_END];
            if (segment_start >= 0 && segment_end < world_size)
            {
                if (tid == 0)
                {
                    for (int src_rank = segment_start; src_rank <= segment_end;
                         ++src_rank)
                    {
                        ag_wait_signal_ge(rank_signals,
                                          src_rank,
                                          signal_value);
                    }
                }
                __syncthreads();
            }
        }
    }

    const int row_in_tile = warp_idx * 16 + rowid;
    const bool row_active = row_in_tile < tile_rows;
    const int flat_row =
        row_active ? sorted_token_ids[sorted_start + row_in_tile] : -1;
    const int token =
        row_active ? sorted_gather_index[sorted_start + row_in_tile] : -1;
    const bool valid_row = flat_row >= 0 && flat_row < full_m * topk &&
                           token >= 0 && token < full_m;

    ag_float4_t acc[NLOOP];
#pragma unroll
    for (int nn = 0; nn < NLOOP; ++nn)
    {
        zero_vec4(acc[nn]);
    }

    const int64_t a_base = static_cast<int64_t>(token) * k_dim;
    for (int k0 = 0; k0 < k_dim; k0 += 32)
    {
        const int k_lane = k0 + col_lane * 8;
        ag_bf16x4_t a0;
        ag_bf16x4_t a1;
        zero_vec4(a0);
        zero_vec4(a1);
        if (valid_row)
        {
            a0 = load_bf16x4_k_contiguous(
                full_input, a_base + k_lane, k_lane, k_dim);
            a1 = load_bf16x4_k_contiguous(
                full_input, a_base + k_lane + 4, k_lane + 4, k_dim);
        }

#pragma unroll
        for (int nn = 0; nn < NLOOP; ++nn)
        {
            const int b_col = n_base + nn * 16 + rowid;
            const ag_bf16x4_t b0 =
                load_bf16x4_weight_kn(weight,
                                      expert,
                                      k_lane,
                                      b_col,
                                      k_dim,
                                      n_dim);
            const ag_bf16x4_t b1 =
                load_bf16x4_weight_kn(weight,
                                      expert,
                                      k_lane + 4,
                                      b_col,
                                      k_dim,
                                      n_dim);
            acc[nn] = ag_mmac_bf16(a0, b0, acc[nn]);
            acc[nn] = ag_mmac_bf16(a1, b1, acc[nn]);
        }
    }

    if (valid_row)
    {
#pragma unroll
        for (int nn = 0; nn < NLOOP; ++nn)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int n = n_base + nn * 16 + col_lane + i * 4;
                if (n < n_dim)
                {
                    output[static_cast<int64_t>(flat_row) * n_dim + n] =
                        ag_float_to_bf16_bits(acc[nn][i]);
                }
            }
        }
    }
}

template <bool UseTileReady>
__launch_bounds__(512, 1) __global__
    void ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_m32_kernel(
        const uint16_t *full_input,
        const uint16_t *weight,
	        const int *sorted_token_ids,
	        const int *sorted_gather_index,
	        const int *rank_tile_info,
	        const int *rank_tile_problem_info,
	        const int *ntiles,
	        const int *rank_signals,
	        int *rank_tile_ready_flags,
        int *n_chunk_done_counts,
        int *n_chunk_done_flags,
        uint16_t *output,
        int full_m,
        int n_dim,
	        int k_dim,
	        int topk,
	        int world_size,
	        int signal_value,
	        int signal_chunk_rows,
	        int chunks_per_rank)
{
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 32;
    constexpr int MMAC_K = 16;
    constexpr int WARP_SIZE = 64;
    constexpr int N_CHUNKS = 2;
    constexpr int WAVE_M = 32;
    constexpr int N_PER_WAVE = BN / N_CHUNKS;
    constexpr int N_MMACS_PER_WAVE = N_PER_WAVE / 16;
    constexpr int A_LOAD_VEC = 4;
    constexpr int A_LOADS = BM * (BK / A_LOAD_VEC);
    constexpr int LDS_A_ELEMS = BK * BM;
    constexpr int LDS_B_ELEMS = BK * BN;

    const int tile_m = blockIdx.x;
    const int n_base = blockIdx.y * BN;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int wave_idx = tid / WARP_SIZE;
    const int lane = tid - wave_idx * WARP_SIZE;
    const int wave_m = wave_idx >> 1;
    const int wave_n_chunk = wave_idx & 1;
    const int row_base = wave_m * WAVE_M;
    const int col_base = n_base + wave_n_chunk * N_PER_WAVE;
    const int lane_row16 = lane & 15;
    const int lane_col4 = lane >> 4;

	    const int *tile_info =
	        rank_tile_problem_info != nullptr
	            ? rank_tile_problem_info +
	                  tile_m * AG_RANK_TILE_PROBLEM_INFO_STRIDE
	            : rank_tile_info + tile_m * AG_RANK_TILE_INFO_STRIDE;
	    const int expert = tile_info[AG_RANK_TILE_EXPERT];
	    const int sorted_start = tile_info[AG_RANK_TILE_SORTED_START];
	    const int tile_rows = tile_info[AG_RANK_TILE_ROWS];

	    if (rank_signals != nullptr)
	    {
	        const bool use_chunk_signal =
	            signal_chunk_rows > 0 && chunks_per_rank > 0;
	        if (use_chunk_signal)
	        {
	            if (rank_tile_problem_info != nullptr)
	            {
	                const int chunk_start =
	                    tile_info[AG_RANK_TILE_PROBLEM_CHUNK_START];
	                const int chunk_end =
	                    tile_info[AG_RANK_TILE_PROBLEM_CHUNK_END];
	                if (chunk_start >= 0 && chunk_end >= chunk_start)
	                {
	                    for (int chunk = chunk_start + tid; chunk <= chunk_end;
	                         chunk += blockDim.x)
	                    {
	                        ag_wait_signal_ge(rank_signals, chunk, signal_value);
	                    }
	                }
	            }
	            else
	            {
	                for (int row = tid; row < tile_rows; row += blockDim.x)
	                {
	                    const int token =
	                        sorted_gather_index[sorted_start + row];
	                    if (token >= 0 && token < full_m)
	                    {
	                        const int chunk_id =
	                            ag_token_chunk_id(token,
	                                              full_m,
	                                              world_size,
	                                              signal_chunk_rows,
	                                              chunks_per_rank);
	                        ag_wait_signal_ge(rank_signals,
	                                          chunk_id,
	                                          signal_value);
	                    }
	                }
	            }
	            __syncthreads();
	        }
	        else
	        {
	            const int segment_start = tile_info[AG_RANK_TILE_RANK_START];
	            const int segment_end = tile_info[AG_RANK_TILE_RANK_END];
	            if (segment_start >= 0 && segment_end < world_size)
	            {
	                if constexpr (UseTileReady)
	                {
	                    if (rank_tile_ready_flags != nullptr && blockIdx.y != 0)
	                    {
	                        if (tid == 0)
	                        {
	                            ag_wait_signal_ge(rank_tile_ready_flags,
	                                              tile_m,
	                                              signal_value);
	                        }
	                    }
	                    else if (tid == 0)
	                    {
	                        for (int src_rank = segment_start;
	                             src_rank <= segment_end;
	                             ++src_rank)
	                        {
	                            ag_wait_signal_ge(rank_signals,
	                                              src_rank,
	                                              signal_value);
	                        }
	                        if (rank_tile_ready_flags != nullptr)
	                        {
	                            __threadfence_system();
	                            atomicExch(rank_tile_ready_flags + tile_m,
	                                       signal_value);
	                        }
	                    }
	                }
	                else if (tid == 0)
	                {
	                    for (int src_rank = segment_start;
	                         src_rank <= segment_end;
	                         ++src_rank)
	                    {
	                        ag_wait_signal_ge(rank_signals,
	                                          src_rank,
	                                          signal_value);
	                    }
	                }
	                __syncthreads();
	            }
	        }
	    }

    __shared__ uint16_t shmem[LDS_A_ELEMS + LDS_B_ELEMS];
    uint16_t *sh_a = shmem;
    uint16_t *sh_b = shmem + LDS_A_ELEMS;

    ag_float4_t acc0[N_MMACS_PER_WAVE];
    ag_float4_t acc1[N_MMACS_PER_WAVE];
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
        zero_vec4(acc0[nn]);
        zero_vec4(acc1[nn]);
    }

    for (int k0 = 0; k0 < k_dim; k0 += BK)
    {
        for (int load = tid; load < A_LOADS; load += blockDim.x)
        {
            const int m = load / (BK / A_LOAD_VEC);
            const int kk = (load - m * (BK / A_LOAD_VEC)) * A_LOAD_VEC;
            const int token_a =
                (m < tile_rows) ? sorted_gather_index[sorted_start + m] : -1;
            const bool valid_a = token_a >= 0 && token_a < full_m;
            const int gk = k0 + kk;
            ag_bf16x4_t value;
            zero_vec4(value);
            if (valid_a && gk < k_dim)
            {
                value = load_bf16x4_k_contiguous(
                    full_input,
                    static_cast<int64_t>(token_a) * k_dim + gk,
                    gk,
                    k_dim);
            }
#pragma unroll
            for (int i = 0; i < A_LOAD_VEC; ++i)
            {
                sh_a[(kk + i) * BM + m] = value[i];
            }
        }

        for (int idx = tid; idx < LDS_B_ELEMS; idx += blockDim.x)
        {
            const int kk = idx / BN;
            const int n_off = idx - kk * BN;
            const int n = n_base + n_off;
            uint16_t value = 0;
            if (k0 + kk < k_dim && n < n_dim)
            {
                value = weight[(static_cast<int64_t>(expert) * k_dim + k0 +
                                kk) *
                                   n_dim +
                               n];
            }
            sh_b[idx] = value;
        }

        __syncthreads();

        const int lane_k = lane / 4;
        const int lane_group = (lane & 3) * 8;
#pragma unroll
        for (int kk0 = 0; kk0 < BK; kk0 += MMAC_K)
        {
            const uint32_t a_lds_offset =
                static_cast<uint32_t>(((kk0 + lane_k) * BM + row_base +
                                       lane_group) *
                                      sizeof(uint16_t));
            ag_uint4_t ra;
            ag_ds_read_m32x16_b16(ra, a_lds_offset);

#pragma unroll
            for (int nn32 = 0; nn32 < N_MMACS_PER_WAVE / 2; ++nn32)
            {
                const uint32_t b_lds_offset =
                    static_cast<uint32_t>((LDS_A_ELEMS +
                                           (kk0 + lane_k) * BN +
                                           wave_n_chunk * N_PER_WAVE +
                                           nn32 * 32 + lane_group) *
                                          sizeof(uint16_t));
                ag_uint4_t rb;
                ag_ds_read_m32x16_b16(rb, b_lds_offset);
                ag_wait_lgkmcnt0();

                const ag_bf16x4_t a0 = ag_bf16x4_from_u32x2(ra, 0);
                const ag_bf16x4_t a1 = ag_bf16x4_from_u32x2(ra, 1);
                const ag_bf16x4_t b0 = ag_bf16x4_from_u32x2(rb, 0);
                const ag_bf16x4_t b1 = ag_bf16x4_from_u32x2(rb, 1);
                acc0[nn32 * 2 + 0] =
                    ag_mmac_bf16(a0, b0, acc0[nn32 * 2 + 0]);
                acc0[nn32 * 2 + 1] =
                    ag_mmac_bf16(a0, b1, acc0[nn32 * 2 + 1]);
                acc1[nn32 * 2 + 0] =
                    ag_mmac_bf16(a1, b0, acc1[nn32 * 2 + 0]);
                acc1[nn32 * 2 + 1] =
                    ag_mmac_bf16(a1, b1, acc1[nn32 * 2 + 1]);
            }
        }
        __syncthreads();
    }

    const int row0 = row_base + lane_row16;
    const int row1 = row0 + 16;
    const int flat0 =
        (row0 < tile_rows) ? sorted_token_ids[sorted_start + row0] : -1;
    const int flat1 =
        (row1 < tile_rows) ? sorted_token_ids[sorted_start + row1] : -1;
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const int n = col_base + nn * 16 + lane_col4 + i * 4;
            if (flat0 >= 0 && flat0 < full_m * topk && n < n_dim)
            {
                output[static_cast<int64_t>(flat0) * n_dim + n] =
                    ag_float_to_bf16_bits(acc0[nn][i]);
            }
            if (flat1 >= 0 && flat1 < full_m * topk && n < n_dim)
            {
                output[static_cast<int64_t>(flat1) * n_dim + n] =
                    ag_float_to_bf16_bits(acc1[nn][i]);
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && n_chunk_done_counts != nullptr &&
        n_chunk_done_flags != nullptr)
    {
        __threadfence_system();
        const int done = atomicAdd(n_chunk_done_counts + blockIdx.y, 1) + 1;
        if (done == *ntiles)
        {
            n_chunk_done_flags[blockIdx.y] = signal_value;
        }
    }
}

template <bool UseTileReady, bool PairBReads>
__launch_bounds__(1024, 1) __global__
    void ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg16_m32_n64_kernel(
        const uint16_t *full_input,
        const uint16_t *weight,
	        const int *sorted_token_ids,
	        const int *sorted_gather_index,
	        const int *rank_tile_info,
	        const int *rank_tile_problem_info,
	        const int *ntiles,
	        const int *rank_signals,
	        int *rank_tile_ready_flags,
        int *n_chunk_done_counts,
        int *n_chunk_done_flags,
        uint16_t *output,
        int full_m,
        int n_dim,
	        int k_dim,
	        int topk,
	        int world_size,
	        int signal_value,
	        int signal_chunk_rows,
	        int chunks_per_rank)
{
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 32;
    constexpr int MMAC_K = 16;
    constexpr int WARP_SIZE = 64;
    constexpr int N_CHUNKS = 4;
    constexpr int WAVE_M = 32;
    constexpr int N_PER_WAVE = BN / N_CHUNKS;
    constexpr int N_MMACS_PER_WAVE = N_PER_WAVE / 16;
    constexpr int A_LOAD_VEC = 4;
    constexpr int A_LOADS = BM * (BK / A_LOAD_VEC);
    constexpr int LDS_A_ELEMS = BK * BM;
    constexpr int LDS_B_ELEMS = BK * BN;

    const int tile_m = blockIdx.x;
    const int n_base = blockIdx.y * BN;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int wave_idx = tid / WARP_SIZE;
    const int lane = tid - wave_idx * WARP_SIZE;
    const int wave_m = wave_idx / N_CHUNKS;
    const int wave_n_chunk = wave_idx - wave_m * N_CHUNKS;
    const int row_base = wave_m * WAVE_M;
    const int col_base = n_base + wave_n_chunk * N_PER_WAVE;
    const int lane_row16 = lane & 15;
    const int lane_col4 = lane >> 4;

	    const int *tile_info =
	        rank_tile_problem_info != nullptr
	            ? rank_tile_problem_info +
	                  tile_m * AG_RANK_TILE_PROBLEM_INFO_STRIDE
	            : rank_tile_info + tile_m * AG_RANK_TILE_INFO_STRIDE;
	    const int expert = tile_info[AG_RANK_TILE_EXPERT];
	    const int sorted_start = tile_info[AG_RANK_TILE_SORTED_START];
	    const int tile_rows = tile_info[AG_RANK_TILE_ROWS];

	    if (rank_signals != nullptr)
	    {
	        const bool use_chunk_signal =
	            signal_chunk_rows > 0 && chunks_per_rank > 0;
	        if (use_chunk_signal)
	        {
	            if (rank_tile_problem_info != nullptr)
	            {
	                const int chunk_start =
	                    tile_info[AG_RANK_TILE_PROBLEM_CHUNK_START];
	                const int chunk_end =
	                    tile_info[AG_RANK_TILE_PROBLEM_CHUNK_END];
	                if (chunk_start >= 0 && chunk_end >= chunk_start)
	                {
	                    for (int chunk = chunk_start + tid; chunk <= chunk_end;
	                         chunk += blockDim.x)
	                    {
	                        ag_wait_signal_ge(rank_signals, chunk, signal_value);
	                    }
	                }
	            }
	            else
	            {
	                for (int row = tid; row < tile_rows; row += blockDim.x)
	                {
	                    const int token =
	                        sorted_gather_index[sorted_start + row];
	                    if (token >= 0 && token < full_m)
	                    {
	                        const int chunk_id =
	                            ag_token_chunk_id(token,
	                                              full_m,
	                                              world_size,
	                                              signal_chunk_rows,
	                                              chunks_per_rank);
	                        ag_wait_signal_ge(rank_signals,
	                                          chunk_id,
	                                          signal_value);
	                    }
	                }
	            }
	            __syncthreads();
	        }
	        else
	        {
	            const int segment_start = tile_info[AG_RANK_TILE_RANK_START];
	            const int segment_end = tile_info[AG_RANK_TILE_RANK_END];
	            if (segment_start >= 0 && segment_end < world_size)
	            {
	                if constexpr (UseTileReady)
	                {
	                    if (rank_tile_ready_flags != nullptr && blockIdx.y != 0)
	                    {
	                        if (tid == 0)
	                        {
	                            ag_wait_signal_ge(rank_tile_ready_flags,
	                                              tile_m,
	                                              signal_value);
	                        }
	                    }
	                    else if (tid == 0)
	                    {
	                        for (int src_rank = segment_start;
	                             src_rank <= segment_end;
	                             ++src_rank)
	                        {
	                            ag_wait_signal_ge(rank_signals,
	                                              src_rank,
	                                              signal_value);
	                        }
	                        if (rank_tile_ready_flags != nullptr)
	                        {
	                            __threadfence_system();
	                            atomicExch(rank_tile_ready_flags + tile_m,
	                                       signal_value);
	                        }
	                    }
	                }
	                else if (tid == 0)
	                {
	                    for (int src_rank = segment_start;
	                         src_rank <= segment_end;
	                         ++src_rank)
	                    {
	                        ag_wait_signal_ge(rank_signals,
	                                          src_rank,
	                                          signal_value);
	                    }
	                }
	                __syncthreads();
	            }
	        }
	    }

    __shared__ uint16_t shmem[LDS_A_ELEMS + LDS_B_ELEMS];
    uint16_t *sh_a = shmem;
    uint16_t *sh_b = shmem + LDS_A_ELEMS;

    ag_float4_t acc0[N_MMACS_PER_WAVE];
    ag_float4_t acc1[N_MMACS_PER_WAVE];
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
        zero_vec4(acc0[nn]);
        zero_vec4(acc1[nn]);
    }

    for (int k0 = 0; k0 < k_dim; k0 += BK)
    {
        if (tid < A_LOADS)
        {
            const int m = tid / (BK / A_LOAD_VEC);
            const int kk = (tid - m * (BK / A_LOAD_VEC)) * A_LOAD_VEC;
            const int token_a =
                (m < tile_rows) ? sorted_gather_index[sorted_start + m] : -1;
            const bool valid_a = token_a >= 0 && token_a < full_m;
            const int gk = k0 + kk;
            ag_bf16x4_t value;
            zero_vec4(value);
            if (valid_a && gk < k_dim)
            {
                value = load_bf16x4_k_contiguous(
                    full_input,
                    static_cast<int64_t>(token_a) * k_dim + gk,
                    gk,
                    k_dim);
            }
#pragma unroll
            for (int i = 0; i < A_LOAD_VEC; ++i)
            {
                sh_a[(kk + i) * BM + m] = value[i];
            }
        }

        for (int idx = tid; idx < LDS_B_ELEMS; idx += blockDim.x)
        {
            const int kk = idx / BN;
            const int n_off = idx - kk * BN;
            const int n = n_base + n_off;
            uint16_t value = 0;
            if (k0 + kk < k_dim && n < n_dim)
            {
                value = weight[(static_cast<int64_t>(expert) * k_dim + k0 +
                                kk) *
                                   n_dim +
                               n];
            }
            sh_b[idx] = value;
        }

        __syncthreads();

        const int lane_k = lane / 4;
        const int lane_group = (lane & 3) * 8;
#pragma unroll
        for (int kk0 = 0; kk0 < BK; kk0 += MMAC_K)
        {
            const uint32_t a_lds_offset =
                static_cast<uint32_t>(((kk0 + lane_k) * BM + row_base +
                                       lane_group) *
                                      sizeof(uint16_t));
            ag_uint4_t ra;
            ag_ds_read_m32x16_b16(ra, a_lds_offset);

            if constexpr (PairBReads)
            {
                ag_uint4_t rb0;
                ag_uint4_t rb1;
                const uint32_t b0_lds_offset =
                    static_cast<uint32_t>((LDS_A_ELEMS +
                                           (kk0 + lane_k) * BN +
                                           wave_n_chunk * N_PER_WAVE +
                                           lane_group) *
                                          sizeof(uint16_t));
                const uint32_t b1_lds_offset =
                    static_cast<uint32_t>((LDS_A_ELEMS +
                                           (kk0 + lane_k) * BN +
                                           wave_n_chunk * N_PER_WAVE + 32 +
                                           lane_group) *
                                          sizeof(uint16_t));
                ag_ds_read_m32x16_b16(rb0, b0_lds_offset);
                ag_ds_read_m32x16_b16(rb1, b1_lds_offset);
                ag_wait_lgkmcnt0();

                const ag_bf16x4_t a0 = ag_bf16x4_from_u32x2(ra, 0);
                const ag_bf16x4_t a1 = ag_bf16x4_from_u32x2(ra, 1);
                const ag_bf16x4_t b00 = ag_bf16x4_from_u32x2(rb0, 0);
                const ag_bf16x4_t b01 = ag_bf16x4_from_u32x2(rb0, 1);
                const ag_bf16x4_t b10 = ag_bf16x4_from_u32x2(rb1, 0);
                const ag_bf16x4_t b11 = ag_bf16x4_from_u32x2(rb1, 1);
                acc0[0] = ag_mmac_bf16(a0, b00, acc0[0]);
                acc0[1] = ag_mmac_bf16(a0, b01, acc0[1]);
                acc1[0] = ag_mmac_bf16(a1, b00, acc1[0]);
                acc1[1] = ag_mmac_bf16(a1, b01, acc1[1]);
                acc0[2] = ag_mmac_bf16(a0, b10, acc0[2]);
                acc0[3] = ag_mmac_bf16(a0, b11, acc0[3]);
                acc1[2] = ag_mmac_bf16(a1, b10, acc1[2]);
                acc1[3] = ag_mmac_bf16(a1, b11, acc1[3]);
            }
            else
            {
#pragma unroll
                for (int nn32 = 0; nn32 < N_MMACS_PER_WAVE / 2; ++nn32)
                {
                    const uint32_t b_lds_offset =
                        static_cast<uint32_t>((LDS_A_ELEMS +
                                               (kk0 + lane_k) * BN +
                                               wave_n_chunk * N_PER_WAVE +
                                               nn32 * 32 + lane_group) *
                                              sizeof(uint16_t));
                    ag_uint4_t rb;
                    ag_ds_read_m32x16_b16(rb, b_lds_offset);
                    ag_wait_lgkmcnt0();

                    const ag_bf16x4_t a0 = ag_bf16x4_from_u32x2(ra, 0);
                    const ag_bf16x4_t a1 = ag_bf16x4_from_u32x2(ra, 1);
                    const ag_bf16x4_t b0 = ag_bf16x4_from_u32x2(rb, 0);
                    const ag_bf16x4_t b1 = ag_bf16x4_from_u32x2(rb, 1);
                    acc0[nn32 * 2 + 0] =
                        ag_mmac_bf16(a0, b0, acc0[nn32 * 2 + 0]);
                    acc0[nn32 * 2 + 1] =
                        ag_mmac_bf16(a0, b1, acc0[nn32 * 2 + 1]);
                    acc1[nn32 * 2 + 0] =
                        ag_mmac_bf16(a1, b0, acc1[nn32 * 2 + 0]);
                    acc1[nn32 * 2 + 1] =
                        ag_mmac_bf16(a1, b1, acc1[nn32 * 2 + 1]);
                }
            }
        }
        __syncthreads();
    }

    const int row0 = row_base + lane_row16;
    const int row1 = row0 + 16;
    const int flat0 =
        (row0 < tile_rows) ? sorted_token_ids[sorted_start + row0] : -1;
    const int flat1 =
        (row1 < tile_rows) ? sorted_token_ids[sorted_start + row1] : -1;
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const int n = col_base + nn * 16 + lane_col4 + i * 4;
            if (flat0 >= 0 && flat0 < full_m * topk && n < n_dim)
            {
                output[static_cast<int64_t>(flat0) * n_dim + n] =
                    ag_float_to_bf16_bits(acc0[nn][i]);
            }
            if (flat1 >= 0 && flat1 < full_m * topk && n < n_dim)
            {
                output[static_cast<int64_t>(flat1) * n_dim + n] =
                    ag_float_to_bf16_bits(acc1[nn][i]);
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && n_chunk_done_counts != nullptr &&
        n_chunk_done_flags != nullptr)
    {
        __threadfence_system();
        const int done = atomicAdd(n_chunk_done_counts + blockIdx.y, 1) + 1;
        if (done == *ntiles)
        {
            n_chunk_done_flags[blockIdx.y] = signal_value;
        }
    }
}

template <bool UseTileReady, bool UseDsReadSwizzle = false>
__launch_bounds__(512, 1) __global__
    void ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_fragment_kernel(
        const uint16_t *full_input,
        const uint16_t *weight,
        const int *sorted_token_ids,
        const int *sorted_gather_index,
        const int *rank_tile_info,
        const int *rank_tile_problem_info,
        const int *ntiles,
        const int *rank_signals,
        int *rank_tile_ready_flags,
        int *n_chunk_done_counts,
        int *n_chunk_done_flags,
        uint16_t *output,
        int full_m,
        int n_dim,
        int k_dim,
        int topk,
        int world_size,
        int signal_value,
        int signal_chunk_rows,
        int chunks_per_rank)
{
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 32;
    constexpr int WARP_SIZE = 64;
    constexpr int N_CHUNKS = 2;
    constexpr int WAVE_M = 32;
    constexpr int N_PER_WAVE = BN / N_CHUNKS;
    constexpr int N_MMACS_PER_WAVE = N_PER_WAVE / 16;
    constexpr int A_LOAD_VEC = 4;
    constexpr int B_LOAD_VEC = 4;
    constexpr int A_LOADS = BM * (BK / A_LOAD_VEC);
    constexpr int B_LOADS_VEC = BK * (BN / B_LOAD_VEC);
    constexpr int FRAGMENT_ELEMS = 16 * BK;
    constexpr int LDS_A_ELEMS = BK * BM;
    constexpr int LDS_B_ELEMS = BK * BN;

    const int tile_m = blockIdx.x;
    const int n_base = blockIdx.y * BN;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int wave_idx = tid / WARP_SIZE;
    const int lane = tid - wave_idx * WARP_SIZE;
    const int wave_m = wave_idx >> 1;
    const int wave_n_chunk = wave_idx & 1;
    const int row_base = wave_m * WAVE_M;
    const int col_base = n_base + wave_n_chunk * N_PER_WAVE;
    const int lane_row16 = lane & 15;
    const int lane_col4 = lane >> 4;

    const int *tile_info =
        rank_tile_problem_info != nullptr
            ? rank_tile_problem_info +
                  tile_m * AG_RANK_TILE_PROBLEM_INFO_STRIDE
            : rank_tile_info + tile_m * AG_RANK_TILE_INFO_STRIDE;
    const int expert = tile_info[AG_RANK_TILE_EXPERT];
    const int sorted_start = tile_info[AG_RANK_TILE_SORTED_START];
    const int tile_rows = tile_info[AG_RANK_TILE_ROWS];

    if (rank_signals != nullptr)
    {
        const bool use_chunk_signal =
            signal_chunk_rows > 0 && chunks_per_rank > 0;
        if (use_chunk_signal)
        {
            if (rank_tile_problem_info != nullptr)
            {
                const int chunk_start =
                    tile_info[AG_RANK_TILE_PROBLEM_CHUNK_START];
                const int chunk_end = tile_info[AG_RANK_TILE_PROBLEM_CHUNK_END];
                if (chunk_start >= 0 && chunk_end >= chunk_start)
                {
                    for (int chunk = chunk_start + tid; chunk <= chunk_end;
                         chunk += blockDim.x)
                    {
                        ag_wait_signal_ge(rank_signals, chunk, signal_value);
                    }
                }
            }
            else
            {
                for (int row = tid; row < tile_rows; row += blockDim.x)
                {
                    const int token = sorted_gather_index[sorted_start + row];
                    if (token >= 0 && token < full_m)
                    {
                        const int chunk_id =
                            ag_token_chunk_id(token,
                                              full_m,
                                              world_size,
                                              signal_chunk_rows,
                                              chunks_per_rank);
                        ag_wait_signal_ge(rank_signals, chunk_id, signal_value);
                    }
                }
            }
            __syncthreads();
        }
        else
        {
            const int segment_start = tile_info[AG_RANK_TILE_RANK_START];
            const int segment_end = tile_info[AG_RANK_TILE_RANK_END];
            if (segment_start >= 0 && segment_end < world_size)
            {
                if constexpr (UseTileReady)
                {
                    if (rank_tile_ready_flags != nullptr && blockIdx.y != 0)
                    {
                        if (tid == 0)
                        {
                            ag_wait_signal_ge(rank_tile_ready_flags,
                                              tile_m,
                                              signal_value);
                        }
                    }
                    else if (tid == 0)
                    {
                        for (int src_rank = segment_start;
                             src_rank <= segment_end;
                             ++src_rank)
                        {
                            ag_wait_signal_ge(rank_signals,
                                              src_rank,
                                              signal_value);
                        }
                        if (rank_tile_ready_flags != nullptr)
                        {
                            __threadfence_system();
                            atomicExch(rank_tile_ready_flags + tile_m,
                                       signal_value);
                        }
                    }
                }
                else if (tid == 0)
                {
                    for (int src_rank = segment_start; src_rank <= segment_end;
                         ++src_rank)
                    {
                        ag_wait_signal_ge(rank_signals,
                                          src_rank,
                                          signal_value);
                    }
                }
                __syncthreads();
            }
        }
    }

    __shared__ uint16_t shmem[LDS_A_ELEMS + LDS_B_ELEMS];
    uint16_t *sh_a = shmem;
    uint16_t *sh_b = shmem + LDS_A_ELEMS;

    ag_float4_t acc0[N_MMACS_PER_WAVE];
    ag_float4_t acc1[N_MMACS_PER_WAVE];
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
        zero_vec4(acc0[nn]);
        zero_vec4(acc1[nn]);
    }

    for (int k0 = 0; k0 < k_dim; k0 += BK)
    {
        for (int load = tid; load < A_LOADS; load += blockDim.x)
        {
            const int m = load / (BK / A_LOAD_VEC);
            const int kk = (load - m * (BK / A_LOAD_VEC)) * A_LOAD_VEC;
            const int token_a =
                (m < tile_rows) ? sorted_gather_index[sorted_start + m] : -1;
            const bool valid_a = token_a >= 0 && token_a < full_m;
            const int gk = k0 + kk;
            ag_bf16x4_t value;
            zero_vec4(value);
            if (valid_a && gk < k_dim)
            {
                value = load_bf16x4_k_contiguous(
                    full_input,
                    static_cast<int64_t>(token_a) * k_dim + gk,
                    gk,
                    k_dim);
            }
#pragma unroll
            for (int i = 0; i < A_LOAD_VEC; ++i)
            {
                const int k = kk + i;
                const int row_group = m >> 4;
                const int row16 = m & 15;
                const int slot =
                    row_group * FRAGMENT_ELEMS +
                    (UseDsReadSwizzle
                         ? ag_dsread_m32x16_b16_swizzled_physical_slot(
                               ag_dsread_m32x16_b16_fragment_slot(row16, k))
                         : ag_dsread_m32x16_b16_fragment_slot(row16, k));
                sh_a[slot] = value[i];
            }
        }

        for (int load = tid; load < B_LOADS_VEC; load += blockDim.x)
        {
            const int kk = load / (BN / B_LOAD_VEC);
            const int n_vec = load - kk * (BN / B_LOAD_VEC);
            const int n_off = n_vec * B_LOAD_VEC;
            const int n = n_base + n_off;
            ag_bf16x4_t value;
            zero_vec4(value);
            if (k0 + kk < k_dim && n < n_dim)
            {
                const int64_t base =
                    (static_cast<int64_t>(expert) * k_dim + k0 + kk) *
                        n_dim +
                    n;
                if (n + B_LOAD_VEC <= n_dim)
                {
                    value =
                        *reinterpret_cast<const ag_bf16x4_t *>(weight + base);
                }
                else
                {
#pragma unroll
                    for (int i = 0; i < B_LOAD_VEC; ++i)
                    {
                        value[i] = (n + i < n_dim) ? weight[base + i] : 0;
                    }
                }
            }
            const int n_tile = n_off >> 4;
            const int n16 = n_off & 15;
            if constexpr (UseDsReadSwizzle)
            {
#pragma unroll
                for (int i = 0; i < B_LOAD_VEC; ++i)
                {
                    const int slot =
                        n_tile * FRAGMENT_ELEMS +
                        ag_dsread_m32x16_b16_swizzled_physical_slot(
                            ag_dsread_m32x16_b16_fragment_slot(n16 + i, kk));
                    sh_b[slot] = value[i];
                }
            }
            else
            {
                const int slot =
                    n_tile * FRAGMENT_ELEMS +
                    ag_dsread_m32x16_b16_fragment_slot(n16, kk);
                *reinterpret_cast<ag_bf16x4_t *>(sh_b + slot) = value;
            }
        }

        __syncthreads();

        const int lane_matrix_offset =
            ag_dsread_m32x16_lane_offset_elems(lane);
        const int ds_lane_offset =
            UseDsReadSwizzle
                ? ag_dsread_m32x16_b16_swizzled_lane_offset_elems(lane)
                : lane_matrix_offset;
        const int row_group0 = wave_m * 2;
        const uint32_t a0_lds_offset =
            static_cast<uint32_t>((row_group0 * FRAGMENT_ELEMS +
                                   ds_lane_offset) *
                                  sizeof(uint16_t));
        const uint32_t a1_lds_offset =
            static_cast<uint32_t>(((row_group0 + 1) * FRAGMENT_ELEMS +
                                   ds_lane_offset) *
                                  sizeof(uint16_t));
        ag_uint4_t ra0;
        ag_uint4_t ra1;
        ag_ds_read_m32x16_b16(ra0, a0_lds_offset);
        ag_ds_read_m32x16_b16(ra1, a1_lds_offset);

#pragma unroll
        for (int nn0 = 0; nn0 < N_MMACS_PER_WAVE; nn0 += 2)
        {
            const int n_tile0 = wave_n_chunk * N_MMACS_PER_WAVE + nn0;
            const int n_tile1 = n_tile0 + 1;
            const uint32_t b0_lds_offset =
                static_cast<uint32_t>((LDS_A_ELEMS +
                                       n_tile0 * FRAGMENT_ELEMS +
                                       ds_lane_offset) *
                                      sizeof(uint16_t));
            const uint32_t b1_lds_offset =
                static_cast<uint32_t>((LDS_A_ELEMS +
                                       n_tile1 * FRAGMENT_ELEMS +
                                       ds_lane_offset) *
                                      sizeof(uint16_t));
            ag_uint4_t rb0;
            ag_uint4_t rb1;
            ag_ds_read_m32x16_b16(rb0, b0_lds_offset);
            ag_ds_read_m32x16_b16(rb1, b1_lds_offset);
            ag_wait_lgkmcnt0();

            const ag_bf16x4_t a00 = ag_bf16x4_from_u32x2(ra0, 0);
            const ag_bf16x4_t a01 = ag_bf16x4_from_u32x2(ra0, 1);
            const ag_bf16x4_t a10 = ag_bf16x4_from_u32x2(ra1, 0);
            const ag_bf16x4_t a11 = ag_bf16x4_from_u32x2(ra1, 1);
            const ag_bf16x4_t b00 = ag_bf16x4_from_u32x2(rb0, 0);
            const ag_bf16x4_t b01 = ag_bf16x4_from_u32x2(rb0, 1);
            const ag_bf16x4_t b10 = ag_bf16x4_from_u32x2(rb1, 0);
            const ag_bf16x4_t b11 = ag_bf16x4_from_u32x2(rb1, 1);

            acc0[nn0 + 0] = ag_mmac_bf16(a00, b00, acc0[nn0 + 0]);
            acc0[nn0 + 0] = ag_mmac_bf16(a01, b01, acc0[nn0 + 0]);
            acc0[nn0 + 1] = ag_mmac_bf16(a00, b10, acc0[nn0 + 1]);
            acc0[nn0 + 1] = ag_mmac_bf16(a01, b11, acc0[nn0 + 1]);
            acc1[nn0 + 0] = ag_mmac_bf16(a10, b00, acc1[nn0 + 0]);
            acc1[nn0 + 0] = ag_mmac_bf16(a11, b01, acc1[nn0 + 0]);
            acc1[nn0 + 1] = ag_mmac_bf16(a10, b10, acc1[nn0 + 1]);
            acc1[nn0 + 1] = ag_mmac_bf16(a11, b11, acc1[nn0 + 1]);
        }
        __syncthreads();
    }

    const int row0 = row_base + lane_row16;
    const int row1 = row0 + 16;
    const int flat0 =
        (row0 < tile_rows) ? sorted_token_ids[sorted_start + row0] : -1;
    const int flat1 =
        (row1 < tile_rows) ? sorted_token_ids[sorted_start + row1] : -1;
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const int n = col_base + nn * 16 + lane_col4 + i * 4;
            if (flat0 >= 0 && flat0 < full_m * topk && n < n_dim)
            {
                output[static_cast<int64_t>(flat0) * n_dim + n] =
                    ag_float_to_bf16_bits(acc0[nn][i]);
            }
            if (flat1 >= 0 && flat1 < full_m * topk && n < n_dim)
            {
                output[static_cast<int64_t>(flat1) * n_dim + n] =
                    ag_float_to_bf16_bits(acc1[nn][i]);
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && n_chunk_done_counts != nullptr &&
        n_chunk_done_flags != nullptr)
    {
        __threadfence_system();
        const int done = atomicAdd(n_chunk_done_counts + blockIdx.y, 1) + 1;
        if (done == *ntiles)
        {
            n_chunk_done_flags[blockIdx.y] = signal_value;
        }
    }
}

template <bool UseTileReady,
          bool UseDsReadFragmentLayout,
          bool UseDsReadSwizzle = false>
__launch_bounds__(1024, 1) __global__
    void ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_kernel(
        const uint16_t *full_input,
        const uint16_t *weight,
        const int *sorted_token_ids,
        const int *sorted_gather_index,
        const int *rank_tile_info,
        const int *rank_tile_problem_info,
        const int *ntiles,
        const int *rank_signals,
        int *rank_tile_ready_flags,
        int *n_chunk_done_counts,
        int *n_chunk_done_flags,
        uint16_t *output,
        int full_m,
        int n_dim,
        int k_dim,
        int topk,
        int world_size,
        int signal_value,
        int signal_chunk_rows,
        int chunks_per_rank)
{
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 32;
    constexpr int MMAC_K = 16;
    constexpr int WARP_SIZE = 64;
    constexpr int WAVES_PER_BLOCK = 16;
    constexpr int N_CHUNKS = 2;
    constexpr int WAVE_M = 16;
    constexpr int N_PER_WAVE = BN / N_CHUNKS;
    constexpr int N_MMACS_PER_WAVE = N_PER_WAVE / 16;
    constexpr int A_LOAD_VEC = 4;
    constexpr int B_LOAD_VEC = 4;
    constexpr int A_LOADS = BM * (BK / A_LOAD_VEC);
    constexpr int B_LOADS_VEC = BK * (BN / B_LOAD_VEC);
    constexpr int FRAGMENT_ELEMS = 16 * BK;
    constexpr int LDS_A_ELEMS = BK * BM;
    constexpr int LDS_B_ELEMS = BK * BN;

    const int tile_m = blockIdx.x;
    const int n_base = blockIdx.y * BN;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int wave_idx = tid / WARP_SIZE;
    const int lane = tid - wave_idx * WARP_SIZE;
    const int wave_m = wave_idx >> 1;
    const int wave_n_chunk = wave_idx & 1;
    const int row_base = wave_m * WAVE_M;
    const int col_base = n_base + wave_n_chunk * N_PER_WAVE;
    const int lane_row16 = lane & 15;
    const int lane_col4 = lane >> 4;

    const int *tile_info =
        rank_tile_problem_info != nullptr
            ? rank_tile_problem_info +
                  tile_m * AG_RANK_TILE_PROBLEM_INFO_STRIDE
            : rank_tile_info + tile_m * AG_RANK_TILE_INFO_STRIDE;
    const int expert = tile_info[AG_RANK_TILE_EXPERT];
    const int sorted_start = tile_info[AG_RANK_TILE_SORTED_START];
    const int tile_rows = tile_info[AG_RANK_TILE_ROWS];

    if (rank_signals != nullptr)
    {
        const bool use_chunk_signal =
            signal_chunk_rows > 0 && chunks_per_rank > 0;
        if (use_chunk_signal)
        {
            if (rank_tile_problem_info != nullptr)
            {
                const int chunk_start =
                    tile_info[AG_RANK_TILE_PROBLEM_CHUNK_START];
                const int chunk_end = tile_info[AG_RANK_TILE_PROBLEM_CHUNK_END];
                if (chunk_start >= 0 && chunk_end >= chunk_start)
                {
                    for (int chunk = chunk_start + tid; chunk <= chunk_end;
                         chunk += blockDim.x)
                    {
                        ag_wait_signal_ge(rank_signals, chunk, signal_value);
                    }
                }
            }
            else
            {
                for (int row = tid; row < tile_rows; row += blockDim.x)
                {
                    const int token = sorted_gather_index[sorted_start + row];
                    if (token >= 0 && token < full_m)
                    {
                        const int chunk_id =
                            ag_token_chunk_id(token,
                                              full_m,
                                              world_size,
                                              signal_chunk_rows,
                                              chunks_per_rank);
                        ag_wait_signal_ge(rank_signals, chunk_id, signal_value);
                    }
                }
            }
            __syncthreads();
        }
        else
        {
            const int segment_start = tile_info[AG_RANK_TILE_RANK_START];
            const int segment_end = tile_info[AG_RANK_TILE_RANK_END];
            if (segment_start >= 0 && segment_end < world_size)
            {
                if constexpr (UseTileReady)
                {
                    if (rank_tile_ready_flags != nullptr && blockIdx.y != 0)
                    {
                        if (tid == 0)
                        {
                            ag_wait_signal_ge(rank_tile_ready_flags,
                                              tile_m,
                                              signal_value);
                        }
                    }
                    else if (tid == 0)
                    {
                        for (int src_rank = segment_start;
                             src_rank <= segment_end;
                             ++src_rank)
                        {
                            ag_wait_signal_ge(rank_signals,
                                              src_rank,
                                              signal_value);
                        }
                        if (rank_tile_ready_flags != nullptr)
                        {
                            __threadfence_system();
                            atomicExch(rank_tile_ready_flags + tile_m,
                                       signal_value);
                        }
                    }
                }
                else if (tid == 0)
                {
                    for (int src_rank = segment_start; src_rank <= segment_end;
                         ++src_rank)
                    {
                        ag_wait_signal_ge(rank_signals,
                                          src_rank,
                                          signal_value);
                    }
                }
                __syncthreads();
            }
        }
    }

    __shared__ uint16_t shmem[LDS_A_ELEMS + LDS_B_ELEMS];
    uint16_t *sh_a = shmem;
    uint16_t *sh_b = shmem + LDS_A_ELEMS;

    ag_float4_t acc[N_MMACS_PER_WAVE];
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
        zero_vec4(acc[nn]);
    }

    for (int k0 = 0; k0 < k_dim; k0 += BK)
    {
        if (tid < A_LOADS)
        {
            const int m = tid / (BK / A_LOAD_VEC);
            const int kk = (tid - m * (BK / A_LOAD_VEC)) * A_LOAD_VEC;
            const int token_a =
                (m < tile_rows) ? sorted_gather_index[sorted_start + m] : -1;
            const bool valid_a = token_a >= 0 && token_a < full_m;
            const int gk = k0 + kk;
            ag_bf16x4_t value;
            zero_vec4(value);
            if (valid_a && gk < k_dim)
            {
                value = load_bf16x4_k_contiguous(
                    full_input,
                    static_cast<int64_t>(token_a) * k_dim + gk,
                    gk,
                    k_dim);
            }
#pragma unroll
            for (int i = 0; i < A_LOAD_VEC; ++i)
            {
                if constexpr (UseDsReadSwizzle)
                {
                    const int k = kk + i;
                    const int row_group = m >> 4;
                    const int row16 = m & 15;
                    const int slot =
                        row_group * FRAGMENT_ELEMS +
                        ag_dsread_m32x16_b16_swizzled_physical_slot(
                            ag_dsread_m32x16_b16_fragment_slot(row16, k));
                    sh_a[slot] = value[i];
                }
                else if constexpr (UseDsReadFragmentLayout)
                {
                    const int k = kk + i;
                    const int row_group = m >> 4;
                    const int row16 = m & 15;
                    const int slot =
                        row_group * FRAGMENT_ELEMS +
                        ag_dsread_m32x16_b16_fragment_slot(row16, k);
                    sh_a[slot] = value[i];
                }
                else
                {
                    sh_a[(kk + i) * BM + m] = value[i];
                }
            }
        }

        if constexpr (UseDsReadFragmentLayout)
        {
            for (int load = tid; load < B_LOADS_VEC; load += blockDim.x)
            {
                const int kk = load / (BN / B_LOAD_VEC);
                const int n_vec = load - kk * (BN / B_LOAD_VEC);
                const int n_off = n_vec * B_LOAD_VEC;
                const int n = n_base + n_off;
                ag_bf16x4_t value;
                zero_vec4(value);
                if (k0 + kk < k_dim && n < n_dim)
                {
                    const int64_t base =
                        (static_cast<int64_t>(expert) * k_dim + k0 + kk) *
                            n_dim +
                        n;
                    if (n + B_LOAD_VEC <= n_dim)
                    {
                        value = *reinterpret_cast<const ag_bf16x4_t *>(
                            weight + base);
                    }
                    else
                    {
#pragma unroll
                        for (int i = 0; i < B_LOAD_VEC; ++i)
                        {
                            value[i] = (n + i < n_dim) ? weight[base + i] : 0;
                        }
                    }
                }
                const int n_tile = n_off >> 4;
                const int n16 = n_off & 15;
                if constexpr (UseDsReadSwizzle)
                {
#pragma unroll
                    for (int i = 0; i < B_LOAD_VEC; ++i)
                    {
                        const int slot =
                            n_tile * FRAGMENT_ELEMS +
                            ag_dsread_m32x16_b16_swizzled_physical_slot(
                                ag_dsread_m32x16_b16_fragment_slot(n16 + i,
                                                                   kk));
                        sh_b[slot] = value[i];
                    }
                }
                else
                {
                    const int slot =
                        n_tile * FRAGMENT_ELEMS +
                        ag_dsread_m32x16_b16_fragment_slot(n16, kk);
                    *reinterpret_cast<ag_bf16x4_t *>(sh_b + slot) = value;
                }
            }
        }
        else
        {
            for (int idx = tid; idx < LDS_B_ELEMS; idx += blockDim.x)
            {
                const int kk = idx / BN;
                const int n_off = idx - kk * BN;
                const int n = n_base + n_off;
                uint16_t value = 0;
                if (k0 + kk < k_dim && n < n_dim)
                {
                    value =
                        weight[(static_cast<int64_t>(expert) * k_dim + k0 +
                                kk) *
                                   n_dim +
                               n];
                }
                sh_b[idx] = value;
            }
        }

        __syncthreads();

        if constexpr (UseDsReadFragmentLayout)
        {
            const int lane_matrix_offset =
                ag_dsread_m32x16_lane_offset_elems(lane);
            const int a_lane_matrix_offset =
                UseDsReadSwizzle
                    ? ag_dsread_m32x16_b16_swizzled_lane_offset_elems(lane)
                    : lane_matrix_offset;
            const uint32_t a_lds_offset =
                static_cast<uint32_t>((wave_m * FRAGMENT_ELEMS +
                                       a_lane_matrix_offset) *
                                      sizeof(uint16_t));
            ag_uint4_t ra;
            ag_ds_read_m32x16_b16(ra, a_lds_offset);

#pragma unroll
            for (int nn0 = 0; nn0 < N_MMACS_PER_WAVE; nn0 += 2)
            {
                const int n_tile0 = wave_n_chunk * N_MMACS_PER_WAVE + nn0;
                const int b_lane_matrix_offset =
                    UseDsReadSwizzle
                        ? ag_dsread_m32x16_b16_swizzled_lane_offset_elems(lane)
                        : lane_matrix_offset;
                const uint32_t b0_lds_offset =
                    static_cast<uint32_t>((LDS_A_ELEMS +
                                           n_tile0 * FRAGMENT_ELEMS +
                                           b_lane_matrix_offset) *
                                          sizeof(uint16_t));
                const int n_tile1 = n_tile0 + 1;
                const uint32_t b1_lds_offset =
                    static_cast<uint32_t>((LDS_A_ELEMS +
                                           n_tile1 * FRAGMENT_ELEMS +
                                           b_lane_matrix_offset) *
                                          sizeof(uint16_t));
                ag_uint4_t rb0;
                ag_uint4_t rb1;
                ag_ds_read_m32x16_b16(rb0, b0_lds_offset);
                ag_ds_read_m32x16_b16(rb1, b1_lds_offset);
                ag_wait_lgkmcnt0();

                const ag_bf16x4_t a0 = ag_bf16x4_from_u32x2(ra, 0);
                const ag_bf16x4_t a1 = ag_bf16x4_from_u32x2(ra, 1);

                const ag_bf16x4_t b00 = ag_bf16x4_from_u32x2(rb0, 0);
                const ag_bf16x4_t b01 = ag_bf16x4_from_u32x2(rb0, 1);
                const ag_bf16x4_t b10 = ag_bf16x4_from_u32x2(rb1, 0);
                const ag_bf16x4_t b11 = ag_bf16x4_from_u32x2(rb1, 1);
                acc[nn0 + 0] = ag_mmac_bf16(a0, b00, acc[nn0 + 0]);
                acc[nn0 + 0] = ag_mmac_bf16(a1, b01, acc[nn0 + 0]);
                acc[nn0 + 1] = ag_mmac_bf16(a0, b10, acc[nn0 + 1]);
                acc[nn0 + 1] = ag_mmac_bf16(a1, b11, acc[nn0 + 1]);
            }
        }
        else
        {
            const int lane_k = lane / 4;
            const int lane_group = (lane & 3) * 8;
#pragma unroll
            for (int kk0 = 0; kk0 < BK; kk0 += MMAC_K)
            {
                const uint32_t a_lds_offset =
                    static_cast<uint32_t>(((kk0 + lane_k) * BM + row_base +
                                           lane_group) *
                                          sizeof(uint16_t));
                ag_uint4_t ra;
                ag_ds_read_m32x16_b16(ra, a_lds_offset);

#pragma unroll
                for (int nn32 = 0; nn32 < N_MMACS_PER_WAVE / 2; ++nn32)
                {
                    const uint32_t b_lds_offset =
                        static_cast<uint32_t>((LDS_A_ELEMS +
                                               (kk0 + lane_k) * BN +
                                               wave_n_chunk * N_PER_WAVE +
                                               nn32 * 32 + lane_group) *
                                              sizeof(uint16_t));
                    ag_uint4_t rb;
                    ag_ds_read_m32x16_b16(rb, b_lds_offset);
                    ag_wait_lgkmcnt0();

                    const ag_bf16x4_t a0 = ag_bf16x4_from_u32x2(ra, 0);
                    const ag_bf16x4_t b0 = ag_bf16x4_from_u32x2(rb, 0);
                    const ag_bf16x4_t b1 = ag_bf16x4_from_u32x2(rb, 1);
                    acc[nn32 * 2 + 0] =
                        ag_mmac_bf16(a0, b0, acc[nn32 * 2 + 0]);
                    acc[nn32 * 2 + 1] =
                        ag_mmac_bf16(a0, b1, acc[nn32 * 2 + 1]);
                }
            }
        }
        __syncthreads();
    }

    const int row = row_base + lane_row16;
    const int flat_row =
        (row < tile_rows) ? sorted_token_ids[sorted_start + row] : -1;
    if (flat_row >= 0 && flat_row < full_m * topk)
    {
#pragma unroll
        for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int n = col_base + nn * 16 + lane_col4 + i * 4;
                if (n < n_dim)
                {
                    output[static_cast<int64_t>(flat_row) * n_dim + n] =
                        ag_float_to_bf16_bits(acc[nn][i]);
                }
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && n_chunk_done_counts != nullptr &&
        n_chunk_done_flags != nullptr)
    {
        __threadfence_system();
        const int done = atomicAdd(n_chunk_done_counts + blockIdx.y, 1) + 1;
        if (done == *ntiles)
        {
            n_chunk_done_flags[blockIdx.y] = signal_value;
        }
    }
}

template <typename T>
__global__ void pack_weight_kn_to_nk_kernel(const T *weight,
                                            T *packed_weight,
                                            int num_experts,
                                            int k_dim,
                                            int n_dim)
{
    const int64_t total =
        static_cast<int64_t>(num_experts) * k_dim * n_dim;
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    const int k = idx % k_dim;
    const int64_t tmp = idx / k_dim;
    const int n = tmp % n_dim;
    const int expert = tmp / n_dim;
    packed_weight[idx] =
        weight[(static_cast<int64_t>(expert) * k_dim + k) * n_dim + n];
}

__global__ void wait_rank_signals_kernel(const int *rank_signals,
                                         int world_size,
                                         int signal_value)
{
    if (threadIdx.x != 0 || blockIdx.x != 0 || rank_signals == nullptr)
    {
        return;
    }
    for (int src_rank = 0; src_rank < world_size; ++src_rank)
    {
        const volatile int *signal =
            reinterpret_cast<const volatile int *>(rank_signals + src_rank);
        while (*signal < signal_value)
        {
            __builtin_amdgcn_s_sleep(1);
        }
    }
}

// The host-side hipStreamWriteValue32 operation is stream ordered, but on
// gfx936 it does not provide a system-scope release for a peer IPC mapping.
// Run the release in a one-thread kernel after the peer DMA on the same push
// stream.  The compiler lowers this exact __hip_atomic_store form to the
// system-visible release sequence used by DeepEP's IPC kernels.
__global__ void publish_signal_release_kernel(int *remote_signal,
                                              uint32_t signal_value)
{
    if (blockIdx.x == 0 && threadIdx.x == 0 && remote_signal != nullptr)
    {
        __hip_atomic_store(remote_signal,
                           static_cast<int>(signal_value),
                           __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

__global__ void system_fence_kernel()
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        __threadfence_system();
    }
}

template <typename T>
void launch_reference_forward(const void *full_input,
                              const void *weight,
                              const void *topk_ids,
                              void *output,
                              int full_m,
                              int n_dim,
                              int k_dim,
                              int num_experts,
                              int topk,
                              c10::ScalarType index_dtype,
                              hipStream_t stream)
{
    const int threads = 256;
    const int64_t total = static_cast<int64_t>(full_m) * topk * n_dim;
    const dim3 grid((total + threads - 1) / threads);
    if (index_dtype == c10::ScalarType::Long)
    {
        hipLaunchKernelGGL((ag_group_gemm_reference_kernel<T, int64_t>),
                           grid,
                           dim3(threads),
                           0,
                           stream,
                           static_cast<const T *>(full_input),
                           static_cast<const T *>(weight),
                           topk_ids,
                           static_cast<T *>(output),
                           full_m,
                           n_dim,
                           k_dim,
                           num_experts,
                           topk);
    }
    else
    {
        hipLaunchKernelGGL((ag_group_gemm_reference_kernel<T, int32_t>),
                           grid,
                           dim3(threads),
                           0,
                           stream,
                           static_cast<const T *>(full_input),
                           static_cast<const T *>(weight),
                           topk_ids,
                           static_cast<T *>(output),
                           full_m,
                           n_dim,
                           k_dim,
                           num_experts,
                           topk);
    }
}

} // namespace

void ag_group_gemm_prepare_metadata_impl(const void *topk_ids,
                                         int *sorted_token_ids,
                                         int *sorted_gather_index,
                                         int *expert_rank_counts,
                                         int *expert_rank_counters,
                                         int *expert_rank_offsets,
                                         int *global_expert_offsets,
                                         int *rank_cursors,
                                         int *rank_tile_info,
                                         int *tile_experts,
	                                         int *tile_starts,
	                                         int *tile_rank_starts,
	                                         int *tile_rank_ends,
	                                         int *rank_tile_problem_info,
	                                         int *ntiles,
	                                         int full_m,
	                                         int topk,
	                                         int num_experts,
	                                         int world_size,
	                                         int rank,
	                                         int sorted_capacity,
	                                         int rank_tile_capacity,
                                         int signal_chunk_rows,
                                         int chunks_per_rank,
	                                         bool strict_rank_tile,
	                                         bool split_local_rank,
	                                         bool build_problem_info,
	                                         c10::ScalarType index_dtype,
	                                         hipStream_t stream)
{
    const int BM =
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_NATIVE_CONTRACT",
                           false)
            ? 256
            : 128;
    const int threads = 256;
    const int total = full_m * topk;
    const dim3 flat_grid((total + threads - 1) / threads);
    const dim3 sorted_grid((sorted_capacity + threads - 1) / threads);
    const int num_buckets = num_experts * world_size;
    const bool stable_expert_rank_sort = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_STABLE_EXPERT_RANK_SORT", false);
    const bool nn_direct_chunk_signal = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_NN_CHUNK_SIGNAL", false);
    const bool native_kn_zigzag_tile_stage = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_NATIVE_KN_ZIGZAG_TILE_STAGE", false);

    CUDA_CHECK(hipMemsetAsync(
        expert_rank_counts, 0, sizeof(int) * num_buckets, stream));
    CUDA_CHECK(hipMemsetAsync(
        expert_rank_counters, 0, sizeof(int) * num_buckets, stream));
    CUDA_CHECK(hipMemsetAsync(ntiles, 0, sizeof(int), stream));

    if (index_dtype == c10::ScalarType::Long)
    {
        hipLaunchKernelGGL((count_topk_by_expert_rank_kernel<int64_t>),
                           flat_grid,
                           dim3(threads),
                           0,
                           stream,
                           topk_ids,
                           expert_rank_counts,
                           full_m,
                           topk,
                           num_experts,
                           world_size);
    }
    else
    {
        hipLaunchKernelGGL((count_topk_by_expert_rank_kernel<int32_t>),
                           flat_grid,
                           dim3(threads),
                           0,
                           stream,
                           topk_ids,
                           expert_rank_counts,
                           full_m,
                           topk,
                           num_experts,
                           world_size);
    }

    hipLaunchKernelGGL(build_expert_rank_offsets_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       expert_rank_counts,
                       expert_rank_offsets,
                       global_expert_offsets,
                       rank_cursors,
                       ntiles,
                       num_experts,
                       world_size,
                       BM);

    hipLaunchKernelGGL(init_sorted_token_ids_kernel,
                       sorted_grid,
                       dim3(threads),
                       0,
                       stream,
                       sorted_token_ids,
                       sorted_capacity);

    if (stable_expert_rank_sort)
    {
        const dim3 stable_grid(num_experts);
        const dim3 stable_block(64);
        if (index_dtype == c10::ScalarType::Long)
        {
            hipLaunchKernelGGL(
                (scatter_sorted_token_ids_by_expert_rank_stable_kernel<
                    int64_t>),
                stable_grid,
                stable_block,
                0,
                stream,
                topk_ids,
                sorted_token_ids,
                sorted_gather_index,
                expert_rank_offsets,
                full_m,
                topk,
                num_experts,
                world_size);
        }
        else
        {
            hipLaunchKernelGGL(
                (scatter_sorted_token_ids_by_expert_rank_stable_kernel<
                    int32_t>),
                stable_grid,
                stable_block,
                0,
                stream,
                topk_ids,
                sorted_token_ids,
                sorted_gather_index,
                expert_rank_offsets,
                full_m,
                topk,
                num_experts,
                world_size);
        }
    }
    else
    {
        CUDA_CHECK(hipMemsetAsync(
            expert_rank_counters, 0, sizeof(int) * num_buckets, stream));
        if (index_dtype == c10::ScalarType::Long)
        {
            hipLaunchKernelGGL(
                (scatter_sorted_token_ids_by_expert_rank_kernel<int64_t>),
                flat_grid,
                dim3(threads),
                0,
                stream,
                topk_ids,
                sorted_token_ids,
                sorted_gather_index,
                expert_rank_counters,
                expert_rank_offsets,
                full_m,
                topk,
                num_experts,
                world_size);
        }
        else
        {
            hipLaunchKernelGGL(
                (scatter_sorted_token_ids_by_expert_rank_kernel<int32_t>),
                flat_grid,
                dim3(threads),
                0,
                stream,
                topk_ids,
                sorted_token_ids,
                sorted_gather_index,
                expert_rank_counters,
                expert_rank_offsets,
                full_m,
                topk,
                num_experts,
                world_size);
        }
    }

    hipLaunchKernelGGL(build_expert_rank_tile_descriptors_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       rank_cursors,
                       global_expert_offsets,
                       rank_tile_info,
                       tile_experts,
                       tile_starts,
                       tile_rank_starts,
                       tile_rank_ends,
                       ntiles,
                       num_experts,
                       world_size,
	                       rank,
	                       strict_rank_tile,
	                       split_local_rank,
	                       BM,
                       native_kn_zigzag_tile_stage);
	    if (build_problem_info && rank_tile_problem_info != nullptr)
	    {
	        hipLaunchKernelGGL(build_rank_tile_problem_info_kernel,
	                           dim3(rank_tile_capacity),
	                           dim3(128),
	                           0,
	                           stream,
	                           rank_tile_info,
	                           sorted_gather_index,
	                           ntiles,
	                           rank_tile_problem_info,
	                           full_m,
	                           world_size,
                           signal_chunk_rows,
                           chunks_per_rank,
                           rank_tile_capacity,
                           nn_direct_chunk_signal);
	    }
}

void ag_group_gemm_prepare_metadata_flat_input_impl(
    const void *topk_ids,
    int *sorted_token_ids,
    int *sorted_gather_index,
    int *expert_rank_counts,
    int *expert_rank_counters,
    int *expert_rank_offsets,
    int *global_expert_offsets,
    int *rank_cursors,
    int *rank_tile_info,
    int *tile_experts,
    int *tile_starts,
    int *tile_rank_starts,
    int *tile_rank_ends,
    int *rank_tile_problem_info,
    int *ntiles,
    int full_m,
    int topk,
    int num_experts,
    int world_size,
    int rank,
    int sorted_capacity,
    int rank_tile_capacity,
    int signal_chunk_rows,
    int chunks_per_rank,
    bool strict_rank_tile,
    bool build_problem_info,
    c10::ScalarType index_dtype,
    hipStream_t stream)
{
    ag_group_gemm_prepare_metadata_impl(topk_ids,
                                        sorted_token_ids,
                                        sorted_gather_index,
                                        expert_rank_counts,
                                        expert_rank_counters,
                                        expert_rank_offsets,
                                        global_expert_offsets,
                                        rank_cursors,
                                        rank_tile_info,
                                        tile_experts,
                                        tile_starts,
                                        tile_rank_starts,
                                        tile_rank_ends,
                                        rank_tile_problem_info,
                                        ntiles,
                                        full_m,
                                        topk,
                                        num_experts,
                                        world_size,
                                        rank,
                                        sorted_capacity,
                                        rank_tile_capacity,
                                        signal_chunk_rows,
                                        chunks_per_rank,
                                        strict_rank_tile,
                                        false,
                                        build_problem_info,
                                        index_dtype,
                                        stream);

    FTRAIN_HYGON_CHECK(sorted_gather_index != nullptr);
    constexpr int threads = 256;
    const int blocks = (sorted_capacity + threads - 1) / threads;
    hipLaunchKernelGGL(use_flat_sorted_rows_as_gather_index_kernel,
                       dim3(blocks),
                       dim3(threads),
                       0,
                       stream,
                       sorted_token_ids,
                       sorted_gather_index,
                       sorted_capacity);
    CUDA_CHECK(hipGetLastError());
}

struct __attribute__((packed)) AgMt256ContractRankTileArgs
{
    const uint16_t *input_buffer;
    const uint16_t *weight;
    const int32_t *sorted_expert_offsets;
    uint16_t *output;
    const int32_t *barrier_ptr;
    const int32_t *tile_info;
    uint32_t problem_count;
    uint32_t n_dim;
    uint32_t k_dim;
    uint32_t topk;
    uint32_t rank;
    uint32_t wait_signal_value;
    uint32_t epilogue_mode;
    uint32_t chunk_size_n;
    uint32_t num_done_groups;
    uint32_t reserved;
    void *topk_stage;
    int32_t *chunk_done_counters;
    int32_t *chunk_done_flags;
};

static_assert(sizeof(AgMt256ContractRankTileArgs) == 112,
              "MT256 native contract args must stay 112 bytes");

// The v36 scale-in-GEMM experiment transports a device pointer through the
// two words immediately preceding its regular task-claim workspace.  Keep the
// write on the producer stream: hipMemcpyAsync from a stack value does not
// retain that source storage until a deferred H2D transfer consumes it.
__global__ void ag_mt256_store_router_scale_pointer_kernel(
    int32_t *control,
    uint64_t router_scale_bits)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        control[0] = static_cast<int32_t>(router_scale_bits);
        control[1] = static_cast<int32_t>(router_scale_bits >> 32);
    }
}

struct __attribute__((packed)) AgMt128LegacyGroupedArgs
{
    uint32_t grouped_sizes;
    void *device_arguments;
    void *args_ptr;
    uint32_t wg_table;
    uint32_t gsu;
    uint32_t reserved;
};

static_assert(sizeof(AgMt128LegacyGroupedArgs) == 32,
              "MT128 legacy grouped args must stay 32 bytes");

struct AgMt256Scratch
{
    void *ptr = nullptr;
    size_t bytes = 0;
};

struct AgMt256StreamingPackContext
{
    hipStream_t stream = nullptr;
    hipEvent_t state_ready = nullptr;
    hipEvent_t seed_ready = nullptr;
    int device = -1;
};

thread_local AgMt256Scratch g_mt256_native_packed_input;
thread_local AgMt256Scratch g_mt256_native_packed_output;
thread_local AgMt256Scratch g_mt256_native_tile_info;
thread_local AgMt256Scratch g_mt256_native_expert_offsets;
thread_local AgMt256Scratch g_mt256_v36_direct_control;
thread_local AgMt256Scratch g_mt256_native_mtail_main_rank_info;
thread_local AgMt256Scratch g_mt256_native_mtail_tail_rank_info;
thread_local AgMt256Scratch g_mt256_native_mtail_counts;
thread_local AgMt256Scratch g_mt256_native_mtail_mt128_args;
thread_local AgMt256Scratch g_mt256_native_mtail_mt128_rowmap;
thread_local AgMt256Scratch g_mt256_native_streaming_pack_state;
thread_local AgMt256StreamingPackContext g_mt256_streaming_pack_context;

bool ag_mt256_ensure_streaming_pack_context(
    AgMt256StreamingPackContext *context)
{
    if (context == nullptr)
    {
        return false;
    }
    int device = -1;
    hipError_t err = hipGetDevice(&device);
    if (err != hipSuccess)
    {
        return false;
    }
    if (context->stream != nullptr && context->state_ready != nullptr &&
        context->seed_ready != nullptr && context->device == device)
    {
        return true;
    }
    if (context->stream != nullptr)
    {
        hipStreamSynchronize(context->stream);
    }
    if (context->state_ready != nullptr)
    {
        hipEventDestroy(context->state_ready);
    }
    if (context->seed_ready != nullptr)
    {
        hipEventDestroy(context->seed_ready);
    }
    if (context->stream != nullptr)
    {
        hipStreamDestroy(context->stream);
    }
    *context = AgMt256StreamingPackContext{};

    int priority = 0;
    if (get_env_bool_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_STREAM_PACK_HIGH_PRIORITY", false))
    {
        int least_priority = 0;
        int greatest_priority = 0;
        err = hipDeviceGetStreamPriorityRange(
            &least_priority, &greatest_priority);
        if (err != hipSuccess)
        {
            return false;
        }
        priority = greatest_priority;
    }
    err = hipStreamCreateWithPriority(
        &context->stream, hipStreamNonBlocking, priority);
    if (err != hipSuccess)
    {
        *context = AgMt256StreamingPackContext{};
        return false;
    }
    err = hipEventCreateWithFlags(
        &context->state_ready, hipEventDisableTiming);
    if (err == hipSuccess)
    {
        err = hipEventCreateWithFlags(
            &context->seed_ready, hipEventDisableTiming);
    }
    if (err != hipSuccess)
    {
        if (context->state_ready != nullptr)
        {
            hipEventDestroy(context->state_ready);
        }
        hipStreamDestroy(context->stream);
        *context = AgMt256StreamingPackContext{};
        return false;
    }
    context->device = device;
    return true;
}

bool ag_mt256_ensure_scratch(AgMt256Scratch *scratch, size_t bytes, void **ptr)
{
    *ptr = nullptr;
    if (scratch == nullptr || bytes == 0)
    {
        return false;
    }
    if (scratch->ptr != nullptr && scratch->bytes >= bytes)
    {
        *ptr = scratch->ptr;
        return true;
    }
    if (scratch->ptr != nullptr)
    {
        hipFree(scratch->ptr);
        scratch->ptr = nullptr;
        scratch->bytes = 0;
    }
    hipError_t err = hipMalloc(&scratch->ptr, bytes);
    if (err != hipSuccess)
    {
        scratch->ptr = nullptr;
        scratch->bytes = 0;
        return false;
    }
    scratch->bytes = bytes;
    *ptr = scratch->ptr;
    return true;
}

__global__ void ag_mt256_build_native_contract_info_kernel(
    const int *__restrict__ rank_tile_info,
    int *__restrict__ contract_tile_info,
    int *__restrict__ sorted_expert_offsets,
    int problem_count,
    int max_experts)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < problem_count)
    {
        const int *src = rank_tile_info + idx * AG_RANK_TILE_INFO_STRIDE;
        contract_tile_info[idx] = src[AG_RANK_TILE_EXPERT];
        contract_tile_info[problem_count + idx] = idx * 256;
        contract_tile_info[2 * problem_count + idx] =
            src[AG_RANK_TILE_RANK_START];
        contract_tile_info[3 * problem_count + idx] =
            src[AG_RANK_TILE_RANK_END];
        contract_tile_info[4 * problem_count + idx] =
            src[AG_RANK_TILE_ROWS];
    }
    if (idx <= max_experts)
    {
        sorted_expert_offsets[idx] = 0;
    }
}

__global__ void ag_mt256_build_native_direct_input_info_kernel(
    const int *__restrict__ rank_tile_info,
    int *__restrict__ contract_tile_info,
    int problem_count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < problem_count)
    {
        const int *src = rank_tile_info + idx * AG_RANK_TILE_INFO_STRIDE;
        contract_tile_info[idx] = src[AG_RANK_TILE_EXPERT];
        contract_tile_info[problem_count + idx] =
            src[AG_RANK_TILE_SORTED_START];
        contract_tile_info[2 * problem_count + idx] =
            src[AG_RANK_TILE_RANK_START];
        contract_tile_info[3 * problem_count + idx] =
            src[AG_RANK_TILE_RANK_END];
        contract_tile_info[4 * problem_count + idx] =
            src[AG_RANK_TILE_ROWS];
    }
}

// Stage-signal candidate ABI: rank_tile_info records physical owner ranks
// while the V40 ring publishes the linear arrival slot for each hop.  Convert
// strict one-owner tiles to that linear slot before the hand-written routed
// kernel polls barrier_ptr.  Keep this isolated behind an explicit switch;
// the historical metadata remains unchanged by default.
__global__ void ag_mt256_build_native_direct_input_stage_info_kernel(
    const int *__restrict__ rank_tile_info,
    int *__restrict__ contract_tile_info,
    int problem_count,
    int rank,
    int world_size,
    int body_slot_base)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < problem_count)
    {
        const int *src = rank_tile_info + idx * AG_RANK_TILE_INFO_STRIDE;
        const int owner_start = src[AG_RANK_TILE_RANK_START];
        const int owner_end = src[AG_RANK_TILE_RANK_END];
        auto signal_slot = [=](int owner) {
            if (owner < 0 || owner >= world_size || world_size <= 1)
            {
                return owner;
            }
            if (body_slot_base > 0)
            {
                return body_slot_base + owner;
            }
            return (owner - rank + world_size) % world_size;
        };
        contract_tile_info[idx] = src[AG_RANK_TILE_EXPERT];
        contract_tile_info[problem_count + idx] =
            src[AG_RANK_TILE_SORTED_START];
        contract_tile_info[2 * problem_count + idx] =
            signal_slot(owner_start);
        int slot_end = signal_slot(owner_end);
        contract_tile_info[3 * problem_count + idx] = slot_end;
        contract_tile_info[4 * problem_count + idx] =
            src[AG_RANK_TILE_ROWS];
    }
}

__global__ void ag_mt256_build_native_direct_input_chunk_info_kernel(
    const int *__restrict__ rank_tile_problem_info,
    int *__restrict__ contract_tile_info,
    int problem_count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < problem_count)
    {
        const int *src = rank_tile_problem_info +
                         idx * AG_RANK_TILE_PROBLEM_INFO_STRIDE;
        contract_tile_info[idx] = src[AG_RANK_TILE_EXPERT];
        contract_tile_info[problem_count + idx] =
            src[AG_RANK_TILE_SORTED_START];
        contract_tile_info[2 * problem_count + idx] =
            src[AG_RANK_TILE_RANK_START];
        contract_tile_info[3 * problem_count + idx] =
            src[AG_RANK_TILE_RANK_END];
        contract_tile_info[4 * problem_count + idx] =
            src[AG_RANK_TILE_ROWS];
        contract_tile_info[5 * problem_count + idx] =
            src[AG_RANK_TILE_PROBLEM_CHUNK_START];
        contract_tile_info[6 * problem_count + idx] =
            src[AG_RANK_TILE_PROBLEM_CHUNK_END];
    }
}

__global__ void ag_mt256_build_native_direct_input_rank_chunk_info_kernel(
    const int *__restrict__ rank_tile_problem_info,
    int *__restrict__ contract_tile_info,
    int problem_count,
    int chunks_per_rank,
    int rank,
    int world_size,
    int chunk_slot_base)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= problem_count)
    {
        return;
    }
    const int *src = rank_tile_problem_info +
                     index * AG_RANK_TILE_PROBLEM_INFO_STRIDE;
    int chunk_start = src[AG_RANK_TILE_PROBLEM_CHUNK_START];
    int chunk_end = src[AG_RANK_TILE_PROBLEM_CHUNK_END];
    if (chunk_start < 0 || chunk_end < chunk_start || chunks_per_rank <= 0)
    {
        const int rank_start = src[AG_RANK_TILE_RANK_START];
        const int rank_end = src[AG_RANK_TILE_RANK_END];
        chunk_start = rank_start * chunks_per_rank;
        chunk_end = (rank_end + 1) * chunks_per_rank - 1;
    }
    contract_tile_info[index] = src[AG_RANK_TILE_EXPERT];
    contract_tile_info[problem_count + index] =
        src[AG_RANK_TILE_SORTED_START];
    const int start_owner = chunk_start / chunks_per_rank;
    const int start_chunk = chunk_start - start_owner * chunks_per_rank;
    const int end_owner = chunk_end / chunks_per_rank;
    const int end_chunk = chunk_end - end_owner * chunks_per_rank;
    const int start_stage =
        (start_owner - rank + world_size) % world_size;
    const int end_stage = (end_owner - rank + world_size) % world_size;
    contract_tile_info[2 * problem_count + index] =
        chunk_slot_base + start_stage * chunks_per_rank + start_chunk;
    contract_tile_info[3 * problem_count + index] =
        chunk_slot_base + end_stage * chunks_per_rank + end_chunk;
    contract_tile_info[4 * problem_count + index] = src[AG_RANK_TILE_ROWS];
}

// Build the same five-field SoA ABI as the direct NN/KN kernel, but use the
// high 16 bits of the row-count field to encode the first physical-rank split.
// The Megatron FC1 input is rank-major all-gather storage whose per-rank rows
// are already stable expert-major.  Hence one or two source-rank spans can be
// addressed from two uniform bases; tiles touching three or more ranks retain
// split=0 and use the validated per-row sorted_gather_index fallback.
__global__ void ag_mt256_build_native_direct_input_compact_info_kernel(
    const int *__restrict__ rank_tile_info,
    const int *__restrict__ sorted_gather_index,
    int *__restrict__ contract_tile_info,
    int problem_count,
    int full_m,
    int world_size,
    int max_compact_segments)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= problem_count)
    {
        return;
    }

    const int *src = rank_tile_info + idx * AG_RANK_TILE_INFO_STRIDE;
    const int sorted_start = src[AG_RANK_TILE_SORTED_START];
    const int rows = src[AG_RANK_TILE_ROWS];
    contract_tile_info[idx] = src[AG_RANK_TILE_EXPERT];
    contract_tile_info[problem_count + idx] = sorted_start;
    contract_tile_info[2 * problem_count + idx] =
        src[AG_RANK_TILE_RANK_START];
    contract_tile_info[3 * problem_count + idx] =
        src[AG_RANK_TILE_RANK_END];

    int split = 0;
    const int local_m = world_size > 0 ? full_m / world_size : 0;
    if (sorted_gather_index != nullptr && rows > 0 && rows <= 0xffff &&
        local_m > 0)
    {
        int segments = 1;
        split = rows;
        int previous_token = sorted_gather_index[sorted_start];
        int previous_rank = previous_token / local_m;
        bool physically_contiguous = previous_token >= 0;
        for (int row = 1; row < rows; ++row)
        {
            const int token = sorted_gather_index[sorted_start + row];
            const int owner_rank = token / local_m;
            if (owner_rank != previous_rank)
            {
                if (segments == 1)
                {
                    split = row;
                }
                ++segments;
                previous_rank = owner_rank;
            }
            else
            {
                // A rank transition jumps to a different all-gather shard;
                // that is exactly what the second uniform base represents.
                // Only rows within one rank segment must be consecutive.
                physically_contiguous = physically_contiguous &&
                                      token == previous_token + 1;
            }
            previous_token = token;
        }
        if (!physically_contiguous || segments > max_compact_segments)
        {
            split = 0;
        }
    }
    contract_tile_info[4 * problem_count + idx] = rows | (split << 16);
}

__global__ void ag_mt256_build_expert_n_major_wgm1_task_map_kernel(
    const int *__restrict__ rank_tile_info,
    int32_t *__restrict__ task_map,
    int problem_count,
    int n_tiles,
    int task_group_n_tiles,
    int readiness_window_stages)
{
    const int problem = blockIdx.x * blockDim.x + threadIdx.x;
    if (problem >= problem_count)
    {
        return;
    }

    const int expert =
        rank_tile_info[problem * AG_RANK_TILE_INFO_STRIDE + AG_RANK_TILE_EXPERT];
    const int tile_group_size =
        task_group_n_tiles > 0 && task_group_n_tiles < n_tiles
            ? task_group_n_tiles
            : n_tiles;
    if (readiness_window_stages > 0)
    {
        const int stage =
            rank_tile_info[problem * AG_RANK_TILE_INFO_STRIDE +
                           AG_RANK_TILE_RANK_END];
        const int readiness_window = stage / readiness_window_stages;
        int group_begin = 0;
        int group_problem_count = 0;
        int local_problem = 0;
        for (int other = 0; other < problem_count; ++other)
        {
            const int *other_info =
                rank_tile_info + other * AG_RANK_TILE_INFO_STRIDE;
            const int other_window =
                other_info[AG_RANK_TILE_RANK_END] / readiness_window_stages;
            const int other_expert = other_info[AG_RANK_TILE_EXPERT];
            if (other_window < readiness_window ||
                (other_window == readiness_window && other_expert < expert))
            {
                ++group_begin;
            }
            else if (other_window == readiness_window && other_expert == expert)
            {
                local_problem += other < problem ? 1 : 0;
                ++group_problem_count;
            }
        }

        for (int n_tile = 0; n_tile < n_tiles; ++n_tile)
        {
            const int tile_group_begin =
                (n_tile / tile_group_size) * tile_group_size;
            const int tile_group_count =
                min(tile_group_size, n_tiles - tile_group_begin);
            const int task = tile_group_begin * problem_count +
                             group_begin * tile_group_count +
                             (n_tile - tile_group_begin) *
                                 group_problem_count +
                             local_problem;
            task_map[task] = static_cast<int32_t>(
                (static_cast<uint32_t>(problem)
                 << kAgMt256Wgm1TaskNTileBits) |
                static_cast<uint32_t>(n_tile));
        }
        return;
    }

    int expert_begin = problem;
    while (expert_begin > 0 &&
           rank_tile_info[(expert_begin - 1) * AG_RANK_TILE_INFO_STRIDE +
                          AG_RANK_TILE_EXPERT] == expert)
    {
        --expert_begin;
    }
    int expert_end = problem + 1;
    while (expert_end < problem_count &&
           rank_tile_info[expert_end * AG_RANK_TILE_INFO_STRIDE +
                          AG_RANK_TILE_EXPERT] == expert)
    {
        ++expert_end;
    }

    const int expert_problem_count = expert_end - expert_begin;
    const int local_problem = problem - expert_begin;
    for (int n_tile = 0; n_tile < n_tiles; ++n_tile)
    {
        const int tile_group_begin =
            (n_tile / tile_group_size) * tile_group_size;
        const int tile_group_count =
            min(tile_group_size, n_tiles - tile_group_begin);
        const int task = tile_group_begin * problem_count +
                         expert_begin * tile_group_count +
                         (n_tile - tile_group_begin) *
                             expert_problem_count +
                         local_problem;
        task_map[task] = static_cast<int32_t>(
            (static_cast<uint32_t>(problem)
             << kAgMt256Wgm1TaskNTileBits) |
            static_cast<uint32_t>(n_tile));
    }
}

__global__ void ag_mt256_split_mtail_rank_tile_info_kernel(
    const int *__restrict__ rank_tile_info,
    int *__restrict__ main_rank_tile_info,
    int *__restrict__ tail_rank_tile_info,
    int *__restrict__ counts,
    int problem_count,
    int tail_threshold)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= problem_count)
    {
        return;
    }
    const int *src = rank_tile_info + idx * AG_RANK_TILE_INFO_STRIDE;
    const int rows = src[AG_RANK_TILE_ROWS];
    if (rows <= 0)
    {
        return;
    }
    const bool use_tail = rows < 256 && rows <= tail_threshold;
    int *dst_base = use_tail ? tail_rank_tile_info : main_rank_tile_info;
    const int dst_idx = atomicAdd(counts + (use_tail ? 1 : 0), 1);
    int *dst = dst_base + dst_idx * AG_RANK_TILE_INFO_STRIDE;
    dst[AG_RANK_TILE_EXPERT] = src[AG_RANK_TILE_EXPERT];
    dst[AG_RANK_TILE_SORTED_START] = src[AG_RANK_TILE_SORTED_START];
    dst[AG_RANK_TILE_ROWS] = rows;
    dst[AG_RANK_TILE_RANK_START] = src[AG_RANK_TILE_RANK_START];
    dst[AG_RANK_TILE_RANK_END] = src[AG_RANK_TILE_RANK_END];
}

constexpr int AG_MT128_LEGACY_ARGS_BYTES = 196;
constexpr int AG_MT128_LEGACY_ROWS = 256;

__device__ inline void ag_mt128_store_u32(uint8_t *base,
                                          int offset,
                                          uint32_t value)
{
    *reinterpret_cast<uint32_t *>(base + offset) = value;
}

__device__ inline void ag_mt128_store_ptr(uint8_t *base,
                                          int offset,
                                          const void *ptr)
{
    const uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
    ag_mt128_store_u32(base, offset, static_cast<uint32_t>(value));
    ag_mt128_store_u32(base, offset + 4, static_cast<uint32_t>(value >> 32));
}

__device__ inline void ag_mt128_store_f32(uint8_t *base,
                                          int offset,
                                          float value)
{
    union
    {
        float f;
        uint32_t u;
    } bits{value};
    ag_mt128_store_u32(base, offset, bits.u);
}

__global__ void ag_mt256_build_mtail_mt128_legacy_args_kernel(
    const int *__restrict__ tail_rank_tile_info,
    const int *__restrict__ sorted_token_ids,
    const uint16_t *__restrict__ full_input,
    const uint16_t *__restrict__ weight_kn,
    uint16_t *__restrict__ direct_output,
    uint8_t *__restrict__ args,
    int *__restrict__ rowmap,
    int problem_count,
    int full_m,
    int n_dim,
    int k_dim)
{
    const int problem = blockIdx.x;
    if (problem >= problem_count)
    {
        return;
    }

    const int *tile =
        tail_rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
    const int expert = tile[AG_RANK_TILE_EXPERT];
    const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
    const int tile_rows = tile[AG_RANK_TILE_ROWS];
    int *problem_rowmap = rowmap + problem * AG_MT128_LEGACY_ROWS;
    int fallback_row = 0;
    if (tile_rows > 0)
    {
        fallback_row = sorted_token_ids[sorted_start];
        if (fallback_row < 0 || fallback_row >= full_m * 6)
        {
            fallback_row = 0;
        }
    }

    for (int row = threadIdx.x; row < AG_MT128_LEGACY_ROWS; row += blockDim.x)
    {
        int flat_row = fallback_row;
        if (row < tile_rows)
        {
            flat_row = sorted_token_ids[sorted_start + row];
            if (flat_row < 0 || flat_row >= full_m * 6)
            {
                flat_row = fallback_row;
            }
        }
        problem_rowmap[row] = flat_row;
    }

    if (threadIdx.x != 0)
    {
        return;
    }

    uint8_t *problem_args =
        args + static_cast<int64_t>(problem) * AG_MT128_LEGACY_ARGS_BYTES;
    uint16_t *problem_output = direct_output;
    const uint16_t *problem_weight =
        weight_kn + static_cast<int64_t>(expert) * k_dim * n_dim;

    ag_mt128_store_u32(problem_args, 0x00, static_cast<uint32_t>(n_dim));
    ag_mt128_store_u32(problem_args,
                       0x04,
                       static_cast<uint32_t>(AG_MT128_LEGACY_ROWS));
    ag_mt128_store_u32(problem_args, 0x08, 1u);
    ag_mt128_store_u32(problem_args, 0x0c, static_cast<uint32_t>(k_dim));
    ag_mt128_store_ptr(problem_args, 0x10, problem_output);
    ag_mt128_store_ptr(problem_args, 0x18, problem_rowmap);
    ag_mt128_store_ptr(problem_args, 0x20, problem_weight);
    ag_mt128_store_ptr(problem_args, 0x28, full_input);
    ag_mt128_store_u32(problem_args, 0x30, static_cast<uint32_t>(n_dim));
    ag_mt128_store_u32(
        problem_args,
        0x34,
        static_cast<uint32_t>(AG_MT128_LEGACY_ROWS * n_dim));
    ag_mt128_store_u32(problem_args, 0x38, static_cast<uint32_t>(n_dim));
    ag_mt128_store_u32(
        problem_args,
        0x3c,
        static_cast<uint32_t>(AG_MT128_LEGACY_ROWS * n_dim));
    ag_mt128_store_u32(problem_args, 0x40, static_cast<uint32_t>(n_dim));
    ag_mt128_store_u32(
        problem_args,
        0x44,
        static_cast<uint32_t>(static_cast<int64_t>(k_dim) * n_dim));
    ag_mt128_store_u32(problem_args, 0x48, static_cast<uint32_t>(k_dim));
    ag_mt128_store_u32(
        problem_args,
        0x4c,
        static_cast<uint32_t>(static_cast<int64_t>(full_m) * k_dim));
    ag_mt128_store_f32(problem_args, 0x50, 1.0f);
    ag_mt128_store_f32(problem_args, 0x60, 0.0f);
}

__global__ void ag_mt256_pack_native_rank_tiles_kernel(
    const uint16_t *__restrict__ full_input,
    const int *__restrict__ sorted_gather_index,
    const int *__restrict__ rank_tile_info,
    uint16_t *__restrict__ packed_input,
    int problem_count,
    int full_m,
    int k_dim)
{
    const int64_t rows = static_cast<int64_t>(problem_count) * 256;
    const int64_t total = rows * k_dim;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride)
    {
        const int row = static_cast<int>(idx / k_dim);
        const int k = static_cast<int>(idx - static_cast<int64_t>(row) * k_dim);
        const int problem = row / 256;
        const int local_row = row - problem * 256;
        const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
        const int tile_rows = tile[AG_RANK_TILE_ROWS];
        uint16_t value = 0;
        if (local_row < tile_rows)
        {
            const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
            const int token = sorted_gather_index[sorted_start + local_row];
            if (token >= 0 && token < full_m)
            {
                value = full_input[static_cast<int64_t>(token) * k_dim + k];
            }
        }
        packed_input[idx] = value;
    }
}

__global__ void ag_mt256_pack_native_rank_tiles_vec8_kernel(
    const uint4 *__restrict__ full_input,
    const int *__restrict__ sorted_gather_index,
    const int *__restrict__ rank_tile_info,
    uint4 *__restrict__ packed_input,
    int problem_count,
    int full_m,
    int k_vec_dim)
{
    const int64_t rows = static_cast<int64_t>(problem_count) * 256;
    const int64_t total = rows * k_vec_dim;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
         idx < total;
         idx += stride)
    {
        const int row = static_cast<int>(idx / k_vec_dim);
        const int k_vec = static_cast<int>(
            idx - static_cast<int64_t>(row) * k_vec_dim);
        const int problem = row / 256;
        const int local_row = row - problem * 256;
        const int *tile =
            rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
        const int tile_rows = tile[AG_RANK_TILE_ROWS];
        uint4 value{0, 0, 0, 0};
        if (local_row < tile_rows)
        {
            const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
            const int token = sorted_gather_index[sorted_start + local_row];
            if (token >= 0 && token < full_m)
            {
                value = full_input[
                    static_cast<int64_t>(token) * k_vec_dim + k_vec];
            }
        }
        packed_input[idx] = value;
    }
}

__global__ void ag_mt256_pack_native_rank_tiles_tile_wait_kernel(
    const uint16_t *__restrict__ full_input,
    const int *__restrict__ sorted_gather_index,
    const int *__restrict__ rank_tile_info,
    const int *__restrict__ rank_signals,
    uint16_t *__restrict__ packed_input,
    int problem_count,
    int full_m,
    int k_dim,
    int world_size,
    int signal_value,
    int k_tile)
{
    const int problem = blockIdx.x;
    const int k_begin = blockIdx.y * k_tile;
    if (problem >= problem_count || k_begin >= k_dim)
    {
        return;
    }
    const int remaining_k = k_dim - k_begin;
    const int tile_k = remaining_k < k_tile ? remaining_k : k_tile;
    const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
    const int tile_rows = tile[AG_RANK_TILE_ROWS];

    if (rank_signals != nullptr)
    {
        const int segment_start = tile[AG_RANK_TILE_RANK_START];
        const int segment_end = tile[AG_RANK_TILE_RANK_END];
        if (segment_start >= 0 && segment_start <= segment_end &&
            segment_end < world_size)
        {
            for (int src_rank = segment_start + threadIdx.x;
                 src_rank <= segment_end;
                 src_rank += blockDim.x)
            {
                ag_wait_signal_ge(rank_signals, src_rank, signal_value);
            }
        }
        __syncthreads();
    }

    const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
    const int64_t out_row_base = static_cast<int64_t>(problem) * 256;
    const int tile_elems = 256 * tile_k;
    for (int linear = threadIdx.x; linear < tile_elems; linear += blockDim.x)
    {
        const int local_row = linear / tile_k;
        const int k = k_begin + (linear - local_row * tile_k);
        uint16_t value = 0;
        if (local_row < tile_rows)
        {
            const int token = sorted_gather_index[sorted_start + local_row];
            if (token >= 0 && token < full_m)
            {
                value =
                    full_input[static_cast<int64_t>(token) * k_dim + k];
            }
        }
        packed_input[(out_row_base + local_row) * k_dim + k] = value;
    }
}

__global__ void ag_mt256_pack_native_rank_tiles_tile_wait_vec8_kernel(
    const uint4 *__restrict__ full_input,
    const int *__restrict__ sorted_gather_index,
    const int *__restrict__ rank_tile_info,
    const int *__restrict__ rank_signals,
    uint4 *__restrict__ packed_input,
    int problem_count,
    int full_m,
    int k_vec_dim,
    int world_size,
    int signal_value,
    int k_tile_vec)
{
    const int problem = blockIdx.x;
    const int k_vec_begin = blockIdx.y * k_tile_vec;
    if (problem >= problem_count || k_vec_begin >= k_vec_dim)
    {
        return;
    }
    const int remaining_k_vec = k_vec_dim - k_vec_begin;
    const int tile_k_vec =
        remaining_k_vec < k_tile_vec ? remaining_k_vec : k_tile_vec;
    const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
    const int tile_rows = tile[AG_RANK_TILE_ROWS];

    if (rank_signals != nullptr)
    {
        const int segment_start = tile[AG_RANK_TILE_RANK_START];
        const int segment_end = tile[AG_RANK_TILE_RANK_END];
        if (segment_start >= 0 && segment_start <= segment_end &&
            segment_end < world_size)
        {
            for (int src_rank = segment_start + threadIdx.x;
                 src_rank <= segment_end;
                 src_rank += blockDim.x)
            {
                ag_wait_signal_ge(rank_signals, src_rank, signal_value);
            }
        }
        __syncthreads();
    }

    const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
    const int64_t out_row_base = static_cast<int64_t>(problem) * 256;
    const int tile_elems = 256 * tile_k_vec;
    for (int linear = threadIdx.x; linear < tile_elems; linear += blockDim.x)
    {
        const int local_row = linear / tile_k_vec;
        const int k_vec = k_vec_begin + (linear - local_row * tile_k_vec);
        uint4 value{0, 0, 0, 0};
        if (local_row < tile_rows)
        {
            const int token = sorted_gather_index[sorted_start + local_row];
            if (token >= 0 && token < full_m)
            {
                value =
                    full_input[static_cast<int64_t>(token) * k_vec_dim +
                               k_vec];
            }
        }
        packed_input[(out_row_base + local_row) * k_vec_dim + k_vec] = value;
    }
}

__global__ void
ag_mt256_pack_native_rank_tiles_streaming_tile_wait_vec8_kernel(
    const uint4 *__restrict__ full_input,
    const int *__restrict__ sorted_gather_index,
    const int *__restrict__ rank_tile_info,
    const int *__restrict__ rank_signals,
    uint4 *__restrict__ packed_input,
    int *__restrict__ pack_state,
    int problem_begin,
    int problem_count,
    int full_m,
    int k_vec_dim,
    int world_size,
    int signal_value,
    int k_tile_vec)
{
    // Put k_slice in the fastest-changing grid dimension so all slices of an
    // early problem are schedulable together.  The stable packed-input route
    // retains its original (problem, k_slice) grid.
    const int problem = problem_begin + blockIdx.y;
    const int k_vec_begin = blockIdx.x * k_tile_vec;
    if (problem >= problem_count || k_vec_begin >= k_vec_dim)
    {
        return;
    }
    const int remaining_k_vec = k_vec_dim - k_vec_begin;
    const int tile_k_vec =
        remaining_k_vec < k_tile_vec ? remaining_k_vec : k_tile_vec;
    const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
    const int tile_rows = tile[AG_RANK_TILE_ROWS];

    if (rank_signals != nullptr)
    {
        const int segment_start = tile[AG_RANK_TILE_RANK_START];
        const int segment_end = tile[AG_RANK_TILE_RANK_END];
        if (segment_start >= 0 && segment_start <= segment_end &&
            segment_end < world_size)
        {
            for (int src_rank = segment_start + threadIdx.x;
                 src_rank <= segment_end;
                 src_rank += blockDim.x)
            {
                ag_wait_signal_ge(rank_signals, src_rank, signal_value);
            }
        }
        __syncthreads();
    }

    const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
    const int64_t out_row_base = static_cast<int64_t>(problem) * 256;
    const int tile_elems = 256 * tile_k_vec;
    for (int linear = threadIdx.x; linear < tile_elems; linear += blockDim.x)
    {
        const int local_row = linear / tile_k_vec;
        const int k_vec = k_vec_begin + (linear - local_row * tile_k_vec);
        uint4 value{0, 0, 0, 0};
        if (local_row < tile_rows)
        {
            const int token = sorted_gather_index[sorted_start + local_row];
            if (token >= 0 && token < full_m)
            {
                value = full_input[
                    static_cast<int64_t>(token) * k_vec_dim + k_vec];
            }
        }
        packed_input[(out_row_base + local_row) * k_vec_dim + k_vec] = value;
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        // Packed input and its consumer live on this device.  The device fence
        // orders this slice's stores before the atomic publication; the GEMM
        // assembly reads the counter with GLC and invalidates vector L1 before
        // consuming the slab.
        __threadfence();
        atomicAdd(pack_state + problem, 1);
    }
}

__global__ void
ag_mt256_pack_native_rank_tiles_streaming_throttled_vec8_kernel(
    const uint4 *__restrict__ full_input,
    const int *__restrict__ sorted_gather_index,
    const int *__restrict__ rank_tile_info,
    const int *__restrict__ rank_signals,
    uint4 *__restrict__ packed_input,
    int *__restrict__ pack_state,
    int problem_begin,
    int problem_count,
    int full_m,
    int k_vec_dim,
    int world_size,
    int signal_value,
    int k_tile_vec,
    int k_slices)
{
    const int remaining_problems = problem_count - problem_begin;
    const int total_tasks = remaining_problems * k_slices;
    for (int task = blockIdx.x; task < total_tasks; task += gridDim.x)
    {
        const int problem = problem_begin + task / k_slices;
        const int k_slice = task - (task / k_slices) * k_slices;
        const int k_vec_begin = k_slice * k_tile_vec;
        const int remaining_k_vec = k_vec_dim - k_vec_begin;
        const int tile_k_vec =
            remaining_k_vec < k_tile_vec ? remaining_k_vec : k_tile_vec;
        const int *tile =
            rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
        const int tile_rows = tile[AG_RANK_TILE_ROWS];

        if (rank_signals != nullptr)
        {
            const int segment_start = tile[AG_RANK_TILE_RANK_START];
            const int segment_end = tile[AG_RANK_TILE_RANK_END];
            if (segment_start >= 0 && segment_start <= segment_end &&
                segment_end < world_size)
            {
                for (int src_rank = segment_start + threadIdx.x;
                     src_rank <= segment_end;
                     src_rank += blockDim.x)
                {
                    ag_wait_signal_ge(rank_signals, src_rank, signal_value);
                }
            }
            __syncthreads();
        }

        const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
        const int64_t out_row_base = static_cast<int64_t>(problem) * 256;
        const int tile_elems = 256 * tile_k_vec;
        for (int linear = threadIdx.x;
             linear < tile_elems;
             linear += blockDim.x)
        {
            const int local_row = linear / tile_k_vec;
            const int k_vec =
                k_vec_begin + (linear - local_row * tile_k_vec);
            uint4 value{0, 0, 0, 0};
            if (local_row < tile_rows)
            {
                const int token =
                    sorted_gather_index[sorted_start + local_row];
                if (token >= 0 && token < full_m)
                {
                    value = full_input[
                        static_cast<int64_t>(token) * k_vec_dim + k_vec];
                }
            }
            packed_input[(out_row_base + local_row) * k_vec_dim + k_vec] =
                value;
        }

        __syncthreads();
        if (threadIdx.x == 0)
        {
            __threadfence();
            atomicAdd(pack_state + problem, 1);
        }
        __syncthreads();
    }
}

__global__ void ag_mt256_scatter_native_contract_output_kernel(
    const uint16_t *__restrict__ packed_output,
    const int *__restrict__ sorted_token_ids,
    const int *__restrict__ rank_tile_info,
    uint16_t *__restrict__ output,
    int problem_count,
    int output_rows,
    int n_dim)
{
    const int64_t rows = static_cast<int64_t>(problem_count) * 256;
    const int64_t total = rows * n_dim;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride)
    {
        const int row = static_cast<int>(idx / n_dim);
        const int n = static_cast<int>(idx - static_cast<int64_t>(row) * n_dim);
        const int problem = row / 256;
        const int local_row = row - problem * 256;
        const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
        const int tile_rows = tile[AG_RANK_TILE_ROWS];
        if (local_row >= tile_rows)
        {
            continue;
        }
        const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
        const int flat_row = sorted_token_ids[sorted_start + local_row];
        if (flat_row >= 0 && flat_row < output_rows)
        {
            output[static_cast<int64_t>(flat_row) * n_dim + n] =
                packed_output[static_cast<int64_t>(row) * n_dim + n];
        }
    }
}

__global__ void ag_mt256_scatter_native_contract_output_vec8_kernel(
    const uint4 *__restrict__ packed_output,
    const int *__restrict__ sorted_token_ids,
    const int *__restrict__ rank_tile_info,
    uint4 *__restrict__ output,
    int problem_count,
    int output_rows,
    int n_vec)
{
    const int64_t rows = static_cast<int64_t>(problem_count) * 256;
    const int64_t total = rows * n_vec;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride)
    {
        const int row = static_cast<int>(idx / n_vec);
        const int col_vec = static_cast<int>(idx - static_cast<int64_t>(row) * n_vec);
        const int problem = row / 256;
        const int local_row = row - problem * 256;
        const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
        const int tile_rows = tile[AG_RANK_TILE_ROWS];
        if (local_row >= tile_rows)
        {
            continue;
        }
        const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
        const int flat_row = sorted_token_ids[sorted_start + local_row];
        if (flat_row >= 0 && flat_row < output_rows)
        {
            output[static_cast<int64_t>(flat_row) * n_vec + col_vec] =
                packed_output[static_cast<int64_t>(row) * n_vec + col_vec];
        }
    }
}

__global__ void ag_mt256_scatter_native_contract_output_stride_kernel(
    const uint16_t *__restrict__ packed_output,
    const int *__restrict__ sorted_token_ids,
    const int *__restrict__ rank_tile_info,
    uint16_t *__restrict__ output,
    int problem_count,
    int output_rows,
    int n_dim,
    int packed_rows_per_tile)
{
    const int64_t rows =
        static_cast<int64_t>(problem_count) * packed_rows_per_tile;
    const int64_t total = rows * n_dim;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride)
    {
        const int row = static_cast<int>(idx / n_dim);
        const int n = static_cast<int>(idx - static_cast<int64_t>(row) * n_dim);
        const int problem = row / packed_rows_per_tile;
        const int local_row = row - problem * packed_rows_per_tile;
        const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
        const int tile_rows = tile[AG_RANK_TILE_ROWS];
        if (local_row >= tile_rows)
        {
            continue;
        }
        const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
        const int flat_row = sorted_token_ids[sorted_start + local_row];
        if (flat_row >= 0 && flat_row < output_rows)
        {
            output[static_cast<int64_t>(flat_row) * n_dim + n] =
                packed_output[static_cast<int64_t>(row) * n_dim + n];
        }
    }
}

__global__ void ag_mt256_scatter_native_contract_output_vec8_stride_kernel(
    const uint4 *__restrict__ packed_output,
    const int *__restrict__ sorted_token_ids,
    const int *__restrict__ rank_tile_info,
    uint4 *__restrict__ output,
    int problem_count,
    int output_rows,
    int n_vec,
    int packed_rows_per_tile)
{
    const int64_t rows =
        static_cast<int64_t>(problem_count) * packed_rows_per_tile;
    const int64_t total = rows * n_vec;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride)
    {
        const int row = static_cast<int>(idx / n_vec);
        const int col_vec = static_cast<int>(idx - static_cast<int64_t>(row) * n_vec);
        const int problem = row / packed_rows_per_tile;
        const int local_row = row - problem * packed_rows_per_tile;
        const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
        const int tile_rows = tile[AG_RANK_TILE_ROWS];
        if (local_row >= tile_rows)
        {
            continue;
        }
        const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
        const int flat_row = sorted_token_ids[sorted_start + local_row];
        if (flat_row >= 0 && flat_row < output_rows)
        {
            output[static_cast<int64_t>(flat_row) * n_vec + col_vec] =
                packed_output[static_cast<int64_t>(row) * n_vec + col_vec];
        }
    }
}

__global__ void ag_mt256_mark_n_chunk_done_kernel(
    int *__restrict__ n_chunk_done_counts,
    int *__restrict__ n_chunk_done_flags,
    int num_done_groups,
    int done_count,
    int signal_value)
{
    const int group = blockIdx.x;
    if (group >= num_done_groups)
    {
        return;
    }
    if (threadIdx.x == 0)
    {
        n_chunk_done_counts[group] = done_count;
        __threadfence_system();
        n_chunk_done_flags[group] = signal_value;
    }
}

__global__ void ag_mt256_scatter_native_contract_output_nchunk_done_kernel(
    const uint16_t *__restrict__ packed_output,
    const int *__restrict__ sorted_token_ids,
    const int *__restrict__ rank_tile_info,
    uint16_t *__restrict__ output,
    int problem_count,
    int output_rows,
    int n_dim,
    int chunk_size_n,
    int *__restrict__ n_chunk_done_counts,
    int *__restrict__ n_chunk_done_flags,
    int signal_value)
{
    const int problem = blockIdx.x;
    const int group = blockIdx.y;
    if (problem >= problem_count)
    {
        return;
    }

    const int n_start = group * chunk_size_n;
    const int n_end =
        n_start + chunk_size_n < n_dim ? n_start + chunk_size_n : n_dim;
    const int group_cols = n_end - n_start;
    const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
    const int tile_rows = tile[AG_RANK_TILE_ROWS];
    const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
    const int total = tile_rows * group_cols;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x)
    {
        const int local_row = idx / group_cols;
        const int n = n_start + (idx - local_row * group_cols);
        const int flat_row = sorted_token_ids[sorted_start + local_row];
        if (flat_row >= 0 && flat_row < output_rows)
        {
            output[static_cast<int64_t>(flat_row) * n_dim + n] =
                packed_output[(static_cast<int64_t>(problem) * 256 +
                               local_row) *
                                  n_dim +
                              n];
        }
    }

    __syncthreads();
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const int done = atomicAdd(n_chunk_done_counts + group, 1) + 1;
        if (done == problem_count)
        {
            __threadfence();
            n_chunk_done_flags[group] = signal_value;
        }
    }
}

__global__ void ag_mt256_scatter_native_contract_output_vec8_nchunk_done_kernel(
    const uint4 *__restrict__ packed_output,
    const int *__restrict__ sorted_token_ids,
    const int *__restrict__ rank_tile_info,
    uint4 *__restrict__ output,
    int problem_count,
    int output_rows,
    int n_vec,
    int chunk_size_n,
    int *__restrict__ n_chunk_done_counts,
    int *__restrict__ n_chunk_done_flags,
    int signal_value)
{
    const int problem = blockIdx.x;
    const int group = blockIdx.y;
    if (problem >= problem_count)
    {
        return;
    }

    const int chunk_vecs = chunk_size_n / 8;
    const int vec_start = group * chunk_vecs;
    const int vec_end =
        vec_start + chunk_vecs < n_vec ? vec_start + chunk_vecs : n_vec;
    const int group_vecs = vec_end - vec_start;
    const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
    const int tile_rows = tile[AG_RANK_TILE_ROWS];
    const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
    const int total = tile_rows * group_vecs;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x)
    {
        const int local_row = idx / group_vecs;
        const int col_vec = vec_start + (idx - local_row * group_vecs);
        const int flat_row = sorted_token_ids[sorted_start + local_row];
        if (flat_row >= 0 && flat_row < output_rows)
        {
            output[static_cast<int64_t>(flat_row) * n_vec + col_vec] =
                packed_output[(static_cast<int64_t>(problem) * 256 +
                               local_row) *
                                  n_vec +
                              col_vec];
        }
    }

    __syncthreads();
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const int done = atomicAdd(n_chunk_done_counts + group, 1) + 1;
        if (done == problem_count)
        {
            __threadfence();
            n_chunk_done_flags[group] = signal_value;
        }
    }
}

__global__ void ag_mt256_scatter_native_contract_output_group_nchunk_done_kernel(
    const uint16_t *__restrict__ packed_output,
    const int *__restrict__ sorted_token_ids,
    const int *__restrict__ rank_tile_info,
    uint16_t *__restrict__ output,
    int problem_count,
    int output_rows,
    int n_dim,
    int chunk_size_n,
    int group,
    int *__restrict__ n_chunk_done_counts,
    int *__restrict__ n_chunk_done_flags,
    int signal_value)
{
    const int problem = blockIdx.x;
    if (problem >= problem_count)
    {
        return;
    }

    const int n_start = group * chunk_size_n;
    const int n_end = min(n_start + chunk_size_n, n_dim);
    const int group_cols = n_end - n_start;
    const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
    const int tile_rows = tile[AG_RANK_TILE_ROWS];
    const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
    const int total = tile_rows * group_cols;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x)
    {
        const int local_row = idx / group_cols;
        const int n = n_start + idx - local_row * group_cols;
        const int flat_row = sorted_token_ids[sorted_start + local_row];
        if (flat_row >= 0 && flat_row < output_rows)
        {
            output[static_cast<int64_t>(flat_row) * n_dim + n] =
                packed_output[(static_cast<int64_t>(problem) * 256 +
                               local_row) *
                                  n_dim +
                              n];
        }
    }

    __syncthreads();
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const int done = atomicAdd(n_chunk_done_counts + group, 1) + 1;
        if (done == problem_count)
        {
            __threadfence_system();
            n_chunk_done_flags[group] = signal_value;
        }
    }
}

__global__ void ag_mt256_scatter_native_contract_output_vec8_group_nchunk_done_kernel(
    const uint4 *__restrict__ packed_output,
    const int *__restrict__ sorted_token_ids,
    const int *__restrict__ rank_tile_info,
    uint4 *__restrict__ output,
    int problem_count,
    int output_rows,
    int n_vec,
    int chunk_size_n,
    int group,
    int *__restrict__ n_chunk_done_counts,
    int *__restrict__ n_chunk_done_flags,
    int signal_value)
{
    const int problem = blockIdx.x;
    if (problem >= problem_count)
    {
        return;
    }

    const int chunk_vecs = chunk_size_n / 8;
    const int vec_start = group * chunk_vecs;
    const int vec_end = min(vec_start + chunk_vecs, n_vec);
    const int group_vecs = vec_end - vec_start;
    const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
    const int tile_rows = tile[AG_RANK_TILE_ROWS];
    const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
    const int total = tile_rows * group_vecs;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x)
    {
        const int local_row = idx / group_vecs;
        const int col_vec = vec_start + idx - local_row * group_vecs;
        const int flat_row = sorted_token_ids[sorted_start + local_row];
        if (flat_row >= 0 && flat_row < output_rows)
        {
            output[static_cast<int64_t>(flat_row) * n_vec + col_vec] =
                packed_output[(static_cast<int64_t>(problem) * 256 +
                               local_row) *
                                  n_vec +
                              col_vec];
        }
    }

    __syncthreads();
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const int done = atomicAdd(n_chunk_done_counts + group, 1) + 1;
        if (done == problem_count)
        {
            __threadfence_system();
            n_chunk_done_flags[group] = signal_value;
        }
    }
}

struct AgMt256Module
{
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    std::string co_path;
    std::string func_name;
};

AgMt256Module &ag_mt256_native_module_cache(bool mtail)
{
    static AgMt256Module main_cache;
    static AgMt256Module mtail_cache;
    return mtail ? mtail_cache : main_cache;
}

std::mutex &ag_mt256_native_module_mutex()
{
    static std::mutex mutex;
    return mutex;
}

bool ag_mt256_get_native_contract_function(
    hipFunction_t *function,
    bool mtail = false,
    AgMt256AsmVariant variant_override = AgMt256AsmVariant::Auto)
{
    const AgMt256AsmVariant variant =
        variant_override == AgMt256AsmVariant::Auto
            ? get_ag_mt256_asm_variant()
            : variant_override;
    const bool explicit_variant = variant != AgMt256AsmVariant::Auto;
    const bool direct_input =
        explicit_variant
            ? ag_mt256_variant_direct_input(variant)
            : get_env_bool_local(
                  "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_INPUT", false);
    const bool wgm8 =
        explicit_variant
            ? ag_mt256_variant_wgm8(variant)
            : direct_input && get_env_bool_local(
                                  "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_WGM8", false);
    const bool direct_output =
        explicit_variant
            ? ag_mt256_variant_direct_output(variant)
            : direct_input && get_env_bool_local(
                                  "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT",
                                  false);
    const bool direct_output_novalid =
        direct_output &&
        get_env_bool_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT_NOVALID",
            false);
    const bool direct_output_nchunk_signal =
        direct_output &&
        (ag_mt256_variant_nchunk_signal(variant) ||
         (!explicit_variant && get_env_bool_local(
              "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT_N_CHUNK_SIGNAL",
              false)));
    const bool direct_output_oob_store =
        variant == AgMt256AsmVariant::V10DirectOutputOobStoreWgm8 ||
        variant == AgMt256AsmVariant::V12DirectOutputNChunkSignalWgm8 ||
        variant == AgMt256AsmVariant::V13DirectOutputNChunkSignalChunkMajor ||
        variant == AgMt256AsmVariant::V14DirectOutputNChunkSignalGroupedWgm ||
        variant == AgMt256AsmVariant::V15DirectOutputNChunkSignalWideGroupedWgm ||
        variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicTNPackedInputDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicTNLoaderKNDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicTNProducerKNDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalScaleInGemm ||
        variant == AgMt256AsmVariant::V36DynamicWgm8_64BitDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicExpertNMajorWgm1DirectOutputNChunkSignal ||
        (!explicit_variant && wgm8 && direct_output &&
         get_env_bool_local(
             "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT_OOB_STORE", false));
    const char *default_co =
        variant == AgMt256AsmVariant::V36DynamicExpertNMajorWgm1DirectOutputNChunkSignal
            ? "hygon_ag_grouped_gemm_v36_dynamic_expert_nmajor_wgm1_direct_output_nchunk_signal.co"
        : variant == AgMt256AsmVariant::V36DynamicPackedOutputDiagnostic
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_stage_ordered_ready_queue_ag_frontload16_win64_tailmerge_wgm8.co"
        : variant == AgMt256AsmVariant::V36DynamicWgm8_64BitDirectOutputNChunkSignal
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_wgm8_segment_explicit_hi_store_only.co"
        : variant == AgMt256AsmVariant::V36DynamicNNAilkStreamingPackedInputDirectOutputNChunkSignal
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_nn_ailk_streaming_packed_input_direct_output_nchunk_signal.co"
        : variant == AgMt256AsmVariant::V36DynamicNNAilkPackedInputDirectOutputNChunkSignal
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_nn_ailk_packed_input_direct_output_nchunk_signal.co"
        : variant == AgMt256AsmVariant::V36DynamicTNPackedInputDirectOutputNChunkSignal
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_tn_packed_input_direct_output_nchunk_signal.co"
        : variant == AgMt256AsmVariant::V36DynamicTNLoaderKNDirectOutputNChunkSignal
            ? (get_env_bool_local(
                   "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_TNLOADER_MEGATRON", false)
                   ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_tnloader_kn_megatron_direct_output_nchunk_signal.co"
                   : "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_tnloader_kn_direct_output_nchunk_signal.co")
        : variant == AgMt256AsmVariant::V36DynamicTNProducerKNDirectOutputNChunkSignal
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_tnproducer_kn_direct_output_nchunk_signal.co"
        : variant == AgMt256AsmVariant::V36DynamicNNAilkChunkDirectOutputNChunkSignal
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_nn_ailk_chunk_direct_output_nchunk_signal.co"
        : variant == AgMt256AsmVariant::V36DynamicNNAilkCompactDirectOutputNChunkSignal
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_nn_ailk_chunk_direct_output_nchunk_signal.co"
        : variant == AgMt256AsmVariant::V36DynamicNNAilkRawHygonOverlapDirectOutputNChunkSignal
            ? (get_env_bool_local(
                   "FTRAIN_HYGON_AG_GROUP_GEMM_NATIVE_KN_V19_TNLOADER_BODY", false)
                   ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_tnloader_kn_direct_output_nchunk_signal.co"
                   : "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_nn_ailk_raw_hygon_overlap_direct_output_nchunk_signal.co")
        : variant == AgMt256AsmVariant::V36DynamicNNAilkDirectOutputNChunkSignal
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_native_kn_a2a_nn_ailk_direct_output_nchunk_signal.co"
        : variant == AgMt256AsmVariant::V36DynamicNNAilkStaticDirectOutputNChunkSignal
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_nn_ailk_static_direct_output_nchunk_signal.co"
        : variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalFastBuffer
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_direct_output_nchunk_signal_fast_buffer.co"
        : variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalScaleInGemm
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_direct_output_nchunk_signal_scale_in_gemm.co"
        : variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_native_nk_a2a_direct_output_nchunk_signal.co"
        : variant == AgMt256AsmVariant::V15DirectOutputNChunkSignalWideGroupedWgm
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v15_direct_output_nchunk_signal_wide_grouped_wgm.co"
        : variant == AgMt256AsmVariant::V14DirectOutputNChunkSignalGroupedWgm
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v14_direct_output_nchunk_signal_grouped_wgm.co"
        : variant == AgMt256AsmVariant::V13DirectOutputNChunkSignalChunkMajor
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v13_direct_output_nchunk_signal_chunk_major.co"
        : variant == AgMt256AsmVariant::V12DirectOutputNChunkSignalWgm8
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v12_direct_output_nchunk_signal_wgm8.co"
        : direct_output_oob_store
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v10_direct_output_oob_store_wgm8.co"
        : variant == AgMt256AsmVariant::V5DirectInputWgm8
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v5_direct_input_wgm8.co"
        : direct_output_novalid
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v10_no_valid.co"
        : direct_output_nchunk_signal
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v7_direct_output_nchunk_signal.co"
        : direct_output
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v7_direct_output.co"
        : direct_input
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v4_direct_input.co"
            : "hygon_ag_grouped_gemm_contract_rank_tile_x64_v1.co";
    const char *co_env = detail::get_vendor_env(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_CO");
    const char *tail_co_env =
        detail::get_vendor_env(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_MTAIL_CO");
    if (mtail && tail_co_env != nullptr && *tail_co_env != '\0')
    {
        co_env = tail_co_env;
    }
    auto resolvePackagedCodeObject = [](const char *directory,
                                        const char *primaryName) {
        std::string result;
        std::string base(directory);
        if (!base.empty() && base.back() != '/')
        {
            base.push_back('/');
        }
        std::string primaryObject = primaryName;
        std::string legacyObject = primaryObject;
        if (legacyObject.rfind("hygon_", 0) == 0)
        {
            legacyObject = "fl" "ux_" + legacyObject.substr(6);
        }
        for (const std::string &name : {primaryObject, legacyObject})
        {
            const std::string candidate = base + name;
            std::ifstream candidateFile(candidate, std::ios::binary);
            if (candidateFile.good())
            {
                result = candidate;
                break;
            }
        }
        return result;
    };
    std::string co_path;
    if (co_env != nullptr && *co_env != '\0')
    {
        co_path = co_env;
    }
    else
    {
        // MoEReduceRS and AGGroupGemm intentionally package different code
        // objects under a few historical filenames.  Prefer the RS-specific
        // directory so the two operators can coexist in one process, while
        // retaining the generic directory as a compatibility fallback.
        const char *rs_asm_dir =
            detail::get_vendor_env("FTRAIN_HYGON_MOE_RS_ASM_DIR_GFX936");
        if (rs_asm_dir != nullptr && *rs_asm_dir != '\0')
        {
            co_path = resolvePackagedCodeObject(rs_asm_dir, default_co);
        }
        if (co_path.empty())
        {
            const char *asm_dir =
                detail::get_vendor_env("FTRAIN_HYGON_ASM_DIR_GFX936");
            if (asm_dir == nullptr || *asm_dir == '\0')
            {
                return false;
            }
            co_path = resolvePackagedCodeObject(asm_dir, default_co);
        }
    }
    if (co_path.empty())
    {
        return false;
    }
    const char *func_env = detail::get_vendor_env(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_FUNC");
    const char *tail_func_env =
        detail::get_vendor_env(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_MTAIL_FUNC");
    if (mtail && tail_func_env != nullptr && *tail_func_env != '\0')
    {
        func_env = tail_func_env;
    }
    std::string func_name =
        (func_env != nullptr && *func_env != '\0')
              ? std::string(func_env)
              : std::string(
                  variant == AgMt256AsmVariant::V36DynamicNNAilkChunkDirectOutputNChunkSignal
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_nn_ailk_chunk_direct_output_nchunk_signal"
                  : variant == AgMt256AsmVariant::V36DynamicNNAilkCompactDirectOutputNChunkSignal
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_nn_ailk_chunk_direct_output_nchunk_signal"
                  : variant == AgMt256AsmVariant::V36DynamicNNAilkRawHygonOverlapDirectOutputNChunkSignal
                      ? (get_env_bool_local(
                             "FTRAIN_HYGON_AG_GROUP_GEMM_NATIVE_KN_V19_TNLOADER_BODY",
                             false)
                             ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_native_nk_a2a_direct_output_nchunk_signal"
                             : "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_nn_ailk_raw_hygon_overlap_direct_output_nchunk_signal")
                  : variant == AgMt256AsmVariant::V36DynamicNNAilkDirectOutputNChunkSignal
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_native_kn_a2a_nn_ailk_direct_output_nchunk_signal"
                  : variant == AgMt256AsmVariant::V36DynamicNNAilkStaticDirectOutputNChunkSignal
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_nn_ailk_static_direct_output_nchunk_signal"
                  : variant == AgMt256AsmVariant::V36DynamicNNAilkStreamingPackedInputDirectOutputNChunkSignal
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_nn_ailk_streaming_packed_input_direct_output_nchunk_signal"
                  : variant == AgMt256AsmVariant::V36DynamicNNAilkPackedInputDirectOutputNChunkSignal
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_nn_ailk_packed_input_direct_output_nchunk_signal"
                  : variant == AgMt256AsmVariant::V36DynamicTNPackedInputDirectOutputNChunkSignal
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_tn_packed_input_direct_output_nchunk_signal"
                  : variant == AgMt256AsmVariant::V36DynamicTNLoaderKNDirectOutputNChunkSignal
                      ? (get_env_bool_local(
                             "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_TNLOADER_MEGATRON",
                             false)
                             ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_tnloader_kn_megatron_direct_output_nchunk_signal"
                             : "hygon_ag_grouped_gemm_contract_rank_tile_x64_native_nk_a2a_direct_output_nchunk_signal")
                  : variant == AgMt256AsmVariant::V36DynamicTNProducerKNDirectOutputNChunkSignal
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_kn_direct_output_nchunk_signal"
                  : variant == AgMt256AsmVariant::V36DynamicPackedOutputDiagnostic
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_stage_ordered_ready_queue_ag_frontload16_win64_tailmerge_wgm8"
                  : variant == AgMt256AsmVariant::V36DynamicExpertNMajorWgm1DirectOutputNChunkSignal
                      ? "hygon_ag_grouped_gemm_v36_dynamic_expert_nmajor_wgm1_direct_output_nchunk_signal"
                  : variant == AgMt256AsmVariant::V36DynamicWgm8_64BitDirectOutputNChunkSignal
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_wgm8_segment_explicit_hi_store_only"
                  : variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalFastBuffer
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_direct_output_nchunk_signal_fast_buffer"
                  : variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalScaleInGemm
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_direct_output_nchunk_signal_scale_in_gemm"
                  : variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_native_nk_a2a_direct_output_nchunk_signal"
                  : variant == AgMt256AsmVariant::V15DirectOutputNChunkSignalWideGroupedWgm
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v15_direct_output_nchunk_signal_wide_grouped_wgm"
                  : variant == AgMt256AsmVariant::V14DirectOutputNChunkSignalGroupedWgm
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v14_direct_output_nchunk_signal_grouped_wgm"
                  : variant == AgMt256AsmVariant::V13DirectOutputNChunkSignalChunkMajor
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v13_direct_output_nchunk_signal_chunk_major"
                  : variant == AgMt256AsmVariant::V12DirectOutputNChunkSignalWgm8
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v12_direct_output_nchunk_signal_wgm8"
                  : direct_output_oob_store
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v10_direct_output_oob_store_wgm8"
                  : variant == AgMt256AsmVariant::V5DirectInputWgm8
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v5_direct_input_wgm8"
                  : direct_output
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v7_direct_output"
                  : direct_input
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v4_direct_input"
                  : "hygon_ag_grouped_gemm_contract_rank_tile_x64_v1");
    func_name = detail::legacy_kernel_function_name(func_name);
    std::lock_guard<std::mutex> guard(ag_mt256_native_module_mutex());
    AgMt256Module &cache = ag_mt256_native_module_cache(mtail);
    if (cache.function != nullptr && cache.co_path == co_path &&
        cache.func_name == func_name)
    {
        *function = cache.function;
        return true;
    }
    if (cache.module != nullptr)
    {
        hipModuleUnload(cache.module);
        cache = AgMt256Module{};
    }
    hipModule_t module = nullptr;
    hipFunction_t loaded_function = nullptr;
    hipError_t err = hipModuleLoad(&module, co_path.c_str());
    if (err != hipSuccess)
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (logged.compare_exchange_strong(expected, true))
        {
            fprintf(stderr,
                    "[hygon] ASM MT256 hipModuleLoad failed for %s: %s\n",
                    co_path.c_str(),
                    hipGetErrorString(err));
        }
        return false;
    }
    err = hipModuleGetFunction(&loaded_function, module, func_name.c_str());
    const std::string legacy_function_name =
        detail::legacy_kernel_function_name(func_name);
    if (err != hipSuccess && legacy_function_name != func_name)
    {
        err = hipModuleGetFunction(
            &loaded_function, module, legacy_function_name.c_str());
    }
    if (err != hipSuccess)
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (logged.compare_exchange_strong(expected, true))
        {
            fprintf(stderr,
                    "[hygon] ASM MT256 hipModuleGetFunction failed for %s in "
                    "%s: %s\n",
                    func_name.c_str(),
                    co_path.c_str(),
                    hipGetErrorString(err));
        }
        hipModuleUnload(module);
        return false;
    }
    cache.module = module;
    cache.function = loaded_function;
    cache.co_path = co_path;
    cache.func_name = func_name;
    *function = loaded_function;
    return true;
}

bool ag_mt256_launch_native_contract(
    const void *full_input,
    const void *weight,
    const void *weight_kn,
    const int *sorted_token_ids,
    const int *sorted_gather_index,
    const int *rank_tile_info,
    const int *rank_tile_problem_info,
    const int *ntiles,
    const int *rank_signals,
    const void *router_scale,
    void *output,
    int full_m,
    int n_dim,
    int k_dim,
    int topk,
    int world_size,
    int rank,
    bool weight_nk,
    int signal_value,
    int *n_chunk_done_counts,
    int *n_chunk_done_flags,
    int n_chunk_size_n,
    int num_n_chunk_done_groups,
    int *external_v36_task_control,
    int external_v36_task_control_capacity,
    int active_signal_chunk_rows,
    int active_chunks_per_rank,
    bool ragged_m,
    c10::ScalarType dtype,
    hipStream_t stream)
{
    if (!get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_NATIVE_CONTRACT",
                            false))
    {
        return false;
    }
    const int n_tiles = (n_dim + 255) / 256;
    const AgMt256AsmVariant requested_variant = get_ag_mt256_asm_variant();
    const AgMt256AsmVariant variant =
        ag_mt256_resolve_variant(requested_variant,
                                 n_tiles,
                                 full_m,
                                 topk,
                                 n_dim,
                                 weight_nk);
    const bool native_kn_weight = ag_mt256_variant_native_kn(variant);
    const bool v19_tnloader_body =
        variant ==
            AgMt256AsmVariant::V36DynamicNNAilkRawHygonOverlapDirectOutputNChunkSignal &&
        get_env_bool_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_NATIVE_KN_V19_TNLOADER_BODY", false);
    const bool native_kn_packed_d64_body =
        variant ==
            AgMt256AsmVariant::V36DynamicNNAilkPackedInputDirectOutputNChunkSignal &&
        get_env_bool_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_NATIVE_KN_PACKED_D64_BODY", false);
    const bool native_kn_packed_d64_raw_input =
        native_kn_packed_d64_body &&
        get_env_bool_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_NATIVE_KN_PACKED_D64_RAW_INPUT", false);
    const bool native_kn_streaming_pack =
        ag_mt256_variant_native_kn_streaming_pack(variant);
    const bool native_kn_chunk = ag_mt256_variant_native_kn_chunk(variant);
    // Experimental hybrid route: keep the fast variant=17 routed body, but
    // consume the seven-field M-chunk metadata when explicitly requested.
    // The dedicated variant=19 route remains the only default native chunk
    // selector.
    const bool native_kn_direct_chunk =
        variant == AgMt256AsmVariant::V36DynamicNNAilkDirectOutputNChunkSignal &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_NN_CHUNK_SIGNAL", false);
    const bool native_kn_stage_signal_map =
        variant == AgMt256AsmVariant::V36DynamicNNAilkDirectOutputNChunkSignal &&
        native_kn_weight &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_NATIVE_KN_STAGE_SIGNAL_MAP", false);
    const bool native_kn_packed_input =
        (ag_mt256_variant_native_kn_packed(variant) &&
         !native_kn_packed_d64_raw_input) ||
        (v19_tnloader_body &&
         get_env_bool_local(
             "FTRAIN_HYGON_AG_GROUP_GEMM_NATIVE_KN_TNLOADER_MMAC_PACKED_INPUT",
             false)) ||
        (ag_mt256_variant_nn_ailk(variant) &&
         !native_kn_packed_d64_raw_input &&
         get_env_bool_local(
             "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_NN_PACK_INPUT", false));
    const bool v36_tn_packed_input =
        !native_kn_weight &&
        (variant == AgMt256AsmVariant::
                        V36DynamicTNPackedInputDirectOutputNChunkSignal ||
         (variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal &&
          get_env_bool_local(
              "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_TN_PACK_INPUT", false)));
    const bool packed_input_enabled =
        native_kn_packed_input || v36_tn_packed_input;
    const bool weight_layout_matches =
        native_kn_weight ? !weight_nk : weight_nk;
    const bool dynamic_topk_variant =
        variant == AgMt256AsmVariant::V36DynamicPackedOutputDiagnostic ||
        variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicTNPackedInputDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicTNLoaderKNDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicTNProducerKNDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalScaleInGemm ||
        variant == AgMt256AsmVariant::V36DynamicWgm8_64BitDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalFastBuffer ||
        variant == AgMt256AsmVariant::V36DynamicNNAilkDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicNNAilkCompactDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicNNAilkChunkDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicNNAilkRawHygonOverlapDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicNNAilkStaticDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicNNAilkPackedInputDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicNNAilkStreamingPackedInputDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicExpertNMajorWgm1DirectOutputNChunkSignal;
    if (dtype != c10::ScalarType::BFloat16 || !weight_layout_matches ||
        topk <= 0 ||
        (topk != 6 && !dynamic_topk_variant) ||
        full_input == nullptr || weight == nullptr ||
        sorted_token_ids == nullptr || sorted_gather_index == nullptr ||
        rank_tile_info == nullptr || ntiles == nullptr ||
        rank_signals == nullptr || output == nullptr ||
        n_dim <= 0 || k_dim <= 0 || full_m <= 0)
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] ASM MT256 native contract skipped: requires BF16, "
                   "the selected weight layout, rank signals and rank-tile "
                   "metadata; "
                   "runtime topk requires a v36 dynamic variant\n");
        }
        return false;
    }
    if ((native_kn_chunk || native_kn_direct_chunk) &&
        rank_tile_problem_info == nullptr)
    {
        return false;
    }
    const bool v36_scale_in_gemm =
        variant == AgMt256AsmVariant::
                       V36DynamicDirectOutputNChunkSignalScaleInGemm;
    if (v36_scale_in_gemm && router_scale == nullptr)
    {
        return false;
    }
    if ((n_chunk_done_counts == nullptr) !=
        (n_chunk_done_flags == nullptr))
    {
        return false;
    }
    if (n_chunk_done_counts != nullptr &&
        (n_chunk_size_n <= 0 || n_chunk_size_n % 256 != 0 ||
         num_n_chunk_done_groups <= 0 ||
         num_n_chunk_done_groups !=
             (n_dim + n_chunk_size_n - 1) / n_chunk_size_n))
    {
        return false;
    }
    const int active_n_chunk_size =
        n_chunk_done_counts != nullptr ? n_chunk_size_n : 256;
    const int active_n_chunk_groups =
        n_chunk_done_counts != nullptr ? num_n_chunk_done_groups
                                       : (n_dim + 255) / 256;

    const bool explicit_variant = variant != AgMt256AsmVariant::Auto;
    const bool v36_expert_nmajor_wgm1_direct =
        variant == AgMt256AsmVariant::
            V36DynamicExpertNMajorWgm1DirectOutputNChunkSignal;
    if (requested_variant == AgMt256AsmVariant::V36AutoDirect)
    {
        static std::atomic<bool> logged_wgm8{false};
        static std::atomic<bool> logged_wgm1{false};
        std::atomic<bool> &logged = v36_expert_nmajor_wgm1_direct
            ? logged_wgm1
            : logged_wgm8;
        bool expected = false;
        if (rank == 0 && logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] ASM MT256 v36 auto-direct: global M=%d, "
                   "physical N tiles=%d -> %s\n",
                   full_m,
                   n_tiles,
                   v36_expert_nmajor_wgm1_direct
                       ? "expert-N-major WGM1"
                       : "dynamic WGM8");
        }
    }
    if (requested_variant ==
            AgMt256AsmVariant::
                V36DynamicExpertNMajorWgm1DirectOutputNChunkSignal &&
        variant != requested_variant)
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (rank == 0 && logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] ASM MT256 WGM1 falling back to v36 WGM8: "
                   "%d physical N tiles exceed the %d-tile task-map "
                   "encoding\n",
                   n_tiles,
                   kAgMt256Wgm1MaxNTiles);
        }
    }
    const bool v36_dynamic_direct =
        variant == AgMt256AsmVariant::V36DynamicPackedOutputDiagnostic ||
        variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicTNPackedInputDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicTNLoaderKNDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicTNProducerKNDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalScaleInGemm ||
        variant == AgMt256AsmVariant::V36DynamicWgm8_64BitDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalFastBuffer ||
        variant == AgMt256AsmVariant::V36DynamicNNAilkDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicNNAilkCompactDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicNNAilkChunkDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicNNAilkRawHygonOverlapDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicNNAilkStaticDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicNNAilkPackedInputDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicNNAilkStreamingPackedInputDirectOutputNChunkSignal ||
        v36_expert_nmajor_wgm1_direct;
    // Packed Native-KN input can overlap its row-pack with remote A2A by
    // waiting per tile instead of globally before the pack launch.  Keep the
    // opt-out environment override, but make this the default only for the
    // non-streaming packed-input variants; all other routes retain false.
    const bool tile_wait_pack_default =
        native_kn_packed_input && !native_kn_streaming_pack;
    const bool tile_wait_pack = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_TILE_WAIT_PACK",
        tile_wait_pack_default);
    const bool direct_input =
        explicit_variant
            ? ag_mt256_variant_direct_input(variant)
            : get_env_bool_local(
                  "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_INPUT", false);
    const bool wgm8 =
        explicit_variant
            ? ag_mt256_variant_wgm8(variant)
            : direct_input && get_env_bool_local(
                                  "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_WGM8", false);
    const bool grouped_wgm =
        explicit_variant && ag_mt256_variant_grouped_wgm(variant);
    const bool wide_grouped_wgm =
        explicit_variant && ag_mt256_variant_wide_grouped_wgm(variant);
    const bool direct_output_requested =
        explicit_variant
            ? ag_mt256_variant_direct_output(variant)
            : get_env_bool_local(
                  "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT", false);
    if (direct_output_requested && !direct_input)
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (rank == 0 && logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] ASM MT256 direct-output skipped: requires "
                   "direct-input mode\n");
        }
        return false;
    }
    const bool direct_output = direct_input && direct_output_requested;
    const bool v36_group_major =
        v36_dynamic_direct && !v36_expert_nmajor_wgm1_direct && direct_output &&
        n_chunk_done_counts != nullptr && n_chunk_done_flags != nullptr &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_GROUP_MAJOR", true);
    const bool direct_output_oob_store =
        variant == AgMt256AsmVariant::V10DirectOutputOobStoreWgm8 ||
        variant == AgMt256AsmVariant::V12DirectOutputNChunkSignalWgm8 ||
        variant == AgMt256AsmVariant::V13DirectOutputNChunkSignalChunkMajor ||
        variant == AgMt256AsmVariant::V14DirectOutputNChunkSignalGroupedWgm ||
        variant == AgMt256AsmVariant::V15DirectOutputNChunkSignalWideGroupedWgm ||
        variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicTNPackedInputDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicTNLoaderKNDirectOutputNChunkSignal ||
        variant == AgMt256AsmVariant::V36DynamicDirectOutputNChunkSignalScaleInGemm ||
        variant == AgMt256AsmVariant::V36DynamicWgm8_64BitDirectOutputNChunkSignal ||
        v36_expert_nmajor_wgm1_direct ||
        (!explicit_variant && wgm8 && direct_output &&
         get_env_bool_local(
             "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT_OOB_STORE", false));
    if (direct_input && (k_dim % 64 != 0))
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] ASM MT256 direct-input skipped: k_dim must be "
                   "a multiple of 64\n");
        }
        return false;
    }
    // Imported direct-output WGM8 launches can leave a workgroup resident at
    // wider N. V10 stays at its original eight-tile contract. V12 is enabled
    // only through the correctness-validated 16-tile 4096 route; it does not
    // inherit the generic wide-N experiment switch.
    const bool v12_direct_output_signal =
        variant == AgMt256AsmVariant::V12DirectOutputNChunkSignalWgm8;
    const bool allow_wide_n = wgm8 && !direct_output_oob_store &&
        get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_WGM8_ALLOW_WIDE_N",
        variant == AgMt256AsmVariant::V5DirectInputWgm8);
    int max_wgm8_n_tiles =
        v36_dynamic_direct
            ? n_tiles
        : v12_direct_output_signal
            ? 16
        : direct_output_oob_store
            ? 8
        : allow_wide_n
            ? get_env_int_local(
                  "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_WGM8_MAX_N_TILES",
                  64)
            : 8;
    max_wgm8_n_tiles = std::max(
        8,
        std::min(
            max_wgm8_n_tiles,
            v36_dynamic_direct ? n_tiles : 64));
    if (wgm8 && n_tiles > max_wgm8_n_tiles)
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (rank == 0 && logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] ASM MT256 WGM8 skipped: variant supports at most "
                   "%d N tiles in this mode (got %d)%s\n",
                   max_wgm8_n_tiles,
                   n_tiles,
                   v36_dynamic_direct
                       ? "; v36 physical N-tile count is allocated automatically"
                   : v12_direct_output_signal
                       ? "; v12 is limited to the validated 4096 route"
                   : direct_output_oob_store
                       ? "; v10 wide-N is disabled after a device hang"
                       : "");
        }
        return false;
    }
    hipError_t err = hipSuccess;
    if (!tile_wait_pack && !direct_input)
    {
        hipLaunchKernelGGL(wait_rank_signals_kernel,
                           dim3(1),
                           dim3(1),
                           0,
                           stream,
                           rank_signals,
                           world_size,
                           signal_value);
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            return false;
        }
    }

    int problem_count = 0;
    err = hipMemcpyAsync(&problem_count,
                         ntiles,
                         sizeof(int),
                         hipMemcpyDeviceToHost,
                         stream);
    if (err != hipSuccess)
    {
        return false;
    }
    err = hipStreamSynchronize(stream);
    if (err != hipSuccess || problem_count <= 0)
    {
        return false;
    }
    if (v36_expert_nmajor_wgm1_direct &&
        problem_count > kAgMt256Wgm1MaxProblems)
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (rank == 0 && logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] ASM MT256 WGM1 skipped: 32-bit task-map encoding "
                   "supports at most %d problems (got %d)\n",
                   kAgMt256Wgm1MaxProblems,
                   problem_count);
        }
        return false;
    }
    const int grouped_wgm_max_n_tiles =
        wide_grouped_wgm ? 64 : 16;
    if (grouped_wgm &&
        (n_chunk_done_counts == nullptr ||
         (!wide_grouped_wgm && problem_count != 128) ||
         n_tiles > grouped_wgm_max_n_tiles ||
         active_n_chunk_size < 256 || active_n_chunk_size % 256 != 0))
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (rank == 0 && logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] ASM MT256 grouped-WGM skipped: requires N-chunk "
                   "signals, %s problems, <=%d N tiles, and 256-aligned "
                   "logical groups (signals=%d problems=%d tiles=%d "
                   "chunk_n=%d)\n",
                   wide_grouped_wgm ? "dynamic" : "128",
                   grouped_wgm_max_n_tiles,
                   n_chunk_done_counts != nullptr ? 1 : 0,
                   problem_count,
                   n_tiles,
                   active_n_chunk_size);
        }
        return false;
    }

    const bool tile_stats = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_TILE_STATS", false);
    static std::atomic<bool> tile_stats_logged{false};
    bool tile_stats_expected = false;
    if (tile_stats && rank == 0 &&
        tile_stats_logged.compare_exchange_strong(tile_stats_expected, true))
    {
        std::vector<int> host_tile_info(
            static_cast<size_t>(problem_count) * AG_RANK_TILE_INFO_STRIDE);
        err = hipMemcpyAsync(host_tile_info.data(),
                             rank_tile_info,
                             host_tile_info.size() * sizeof(int),
                             hipMemcpyDeviceToHost,
                             stream);
        if (err == hipSuccess)
        {
            err = hipStreamSynchronize(stream);
        }
        if (err == hipSuccess)
        {
            int full_tiles = 0;
            int partial_tiles = 0;
            int empty_tiles = 0;
            int partial_min = 256;
            int partial_max = 0;
            long long valid_rows = 0;
            int partial_bins[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int idx = 0; idx < problem_count; ++idx)
            {
                const int rows =
                    host_tile_info[static_cast<size_t>(idx) *
                                       AG_RANK_TILE_INFO_STRIDE +
                                   AG_RANK_TILE_ROWS];
                valid_rows += rows;
                if (rows >= 256)
                {
                    ++full_tiles;
                }
                else if (rows > 0)
                {
                    ++partial_tiles;
                    partial_min = rows < partial_min ? rows : partial_min;
                    partial_max = rows > partial_max ? rows : partial_max;
                    const int bin = (rows - 1) / 32;
                    partial_bins[bin < 8 ? bin : 7] += 1;
                }
                else
                {
                    ++empty_tiles;
                }
            }
            const long long packed_rows =
                static_cast<long long>(problem_count) * 256LL;
            printf("[hygon] ASM MT256 tile stats: problems=%d full=%d "
                   "partial=%d empty=%d valid_rows=%lld padding_rows=%lld "
                   "partial_min=%d partial_max=%d partial_bins_32="
                   "[%d,%d,%d,%d,%d,%d,%d,%d]\n",
                   problem_count,
                   full_tiles,
                   partial_tiles,
                   empty_tiles,
                   valid_rows,
                   packed_rows - valid_rows,
                   partial_tiles > 0 ? partial_min : 0,
                   partial_max,
                   partial_bins[0],
                   partial_bins[1],
                   partial_bins[2],
                   partial_bins[3],
                   partial_bins[4],
                   partial_bins[5],
                   partial_bins[6],
                   partial_bins[7]);
        }
        else
        {
            printf("[hygon] ASM MT256 tile stats failed: hip error %d\n",
                   static_cast<int>(err));
        }
    }

    const bool vec_scatter = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_VEC_SCATTER", direct_input);
    const bool vec_pack_requested = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_VEC_PACK",
        packed_input_enabled);

    constexpr int kMaxExpertsForContract = 256;
    const int64_t packed_rows = static_cast<int64_t>(problem_count) * 256;
    const int64_t packed_output_rows = direct_output ? 0 : packed_rows;
    const bool ragged_ring_chunk_alloc = ragged_m &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_RAGGED_RING_CHUNK", true);
    const bool ragged_fixed_chunk_alloc = ragged_m &&
        get_env_bool_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_RAGGED_FIXED_CHUNK_ACQUIRE", false);
    const size_t packed_input_bytes =
        direct_input && !packed_input_enabled
            ? 0
            : static_cast<size_t>(packed_rows) * static_cast<size_t>(k_dim) *
                  sizeof(uint16_t);
    const size_t packed_output_bytes =
        direct_output
            ? 0
            : static_cast<size_t>(packed_output_rows) *
                  static_cast<size_t>(n_dim) * sizeof(uint16_t);
    const size_t tile_info_fields =
        (native_kn_chunk || native_kn_direct_chunk || ragged_ring_chunk_alloc ||
         ragged_fixed_chunk_alloc)
            ? 7
            : 5;
    const size_t tile_info_bytes =
        static_cast<size_t>(problem_count) * tile_info_fields * sizeof(int);
    // Keep the experimental scale table adjacent to the direct kernarg tile
    // map.  This preserves the 112-byte ABI and avoids a mutable-pointer
    // indirection in the hand-written epilogue.
    const size_t router_scale_bytes = v36_scale_in_gemm
        ? static_cast<size_t>(full_m) * static_cast<size_t>(topk) *
              sizeof(uint16_t)
        : 0;
    const size_t contract_tile_info_bytes =
        tile_info_bytes + router_scale_bytes;
    const size_t offsets_bytes =
        static_cast<size_t>(kMaxExpertsForContract + 1) * sizeof(int);
    void *packed_input = nullptr;
    void *packed_output = nullptr;
    void *contract_tile_info = nullptr;
    void *sorted_expert_offsets = nullptr;
    void *v36_direct_control = nullptr;
    void *streaming_pack_state = nullptr;
    const size_t v36_control_prefix_words = 0;
    const size_t v36_direct_control_words = v36_dynamic_direct
        ? v36_control_prefix_words +
              static_cast<size_t>(problem_count) *
                  static_cast<size_t>(n_tiles) +
              static_cast<size_t>(active_n_chunk_groups)
        : 0;
    const size_t v36_direct_control_bytes =
        v36_direct_control_words * sizeof(int32_t);
    const size_t streaming_pack_state_bytes = native_kn_streaming_pack
        ? static_cast<size_t>(problem_count) * sizeof(int32_t)
        : 0;
    if (v36_dynamic_direct && external_v36_task_control != nullptr)
    {
        if (external_v36_task_control_capacity < 0 ||
            static_cast<size_t>(external_v36_task_control_capacity) <
                v36_direct_control_words)
        {
            return false;
        }
        v36_direct_control = external_v36_task_control;
    }
    if (((!direct_input || packed_input_enabled) &&
         !ag_mt256_ensure_scratch(&g_mt256_native_packed_input,
                                  packed_input_bytes,
                                  &packed_input)) ||
        (!direct_output &&
         !ag_mt256_ensure_scratch(&g_mt256_native_packed_output,
                                  packed_output_bytes,
                                  &packed_output)) ||
        !ag_mt256_ensure_scratch(&g_mt256_native_tile_info,
                                 contract_tile_info_bytes,
                                 &contract_tile_info) ||
        (v36_dynamic_direct && v36_direct_control == nullptr &&
         !ag_mt256_ensure_scratch(&g_mt256_v36_direct_control,
                                  v36_direct_control_bytes,
                                  &v36_direct_control)) ||
        (native_kn_streaming_pack &&
         !ag_mt256_ensure_scratch(&g_mt256_native_streaming_pack_state,
                                  streaming_pack_state_bytes,
                                  &streaming_pack_state)) ||
        (!direct_input &&
         !ag_mt256_ensure_scratch(&g_mt256_native_expert_offsets,
                                  offsets_bytes,
                                  &sorted_expert_offsets)))
    {
        return false;
    }
    if (v36_dynamic_direct)
    {
        err = hipMemsetAsync(
            v36_direct_control, 0, v36_direct_control_bytes, stream);
        if (err != hipSuccess)
        {
            return false;
        }
    }
    if (native_kn_streaming_pack)
    {
        if (!ag_mt256_ensure_streaming_pack_context(
                &g_mt256_streaming_pack_context))
        {
            return false;
        }
        err = hipMemsetAsync(
            streaming_pack_state, 0, streaming_pack_state_bytes, stream);
        if (err == hipSuccess)
        {
            err = hipEventRecord(
                g_mt256_streaming_pack_context.state_ready, stream);
        }
        if (err == hipSuccess)
        {
            err = hipStreamWaitEvent(
                g_mt256_streaming_pack_context.stream,
                g_mt256_streaming_pack_context.state_ready,
                0);
        }
        if (err != hipSuccess)
        {
            return false;
        }
    }
    const bool native_profile_requested = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_PROFILE", false);
    const int native_profile_limit = get_env_int_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_PROFILE_ITERS", 4);
    static std::atomic<int> native_profile_counter{0};
    int native_profile_iter = -1;
    hipEvent_t native_prof_start = nullptr;
    hipEvent_t native_prof_after_info = nullptr;
    hipEvent_t native_prof_after_asm = nullptr;
    hipEvent_t native_prof_after_scatter = nullptr;
    const bool native_profile_this =
        native_profile_requested && rank == 0 &&
        (native_profile_iter = native_profile_counter.fetch_add(1)) <
            native_profile_limit;
    if (native_profile_this)
    {
        CUDA_CHECK(hipEventCreate(&native_prof_start));
        CUDA_CHECK(hipEventCreate(&native_prof_after_info));
        CUDA_CHECK(hipEventCreate(&native_prof_after_asm));
        CUDA_CHECK(hipEventCreate(&native_prof_after_scatter));
        CUDA_CHECK(hipEventRecord(native_prof_start, stream));
    }

    constexpr int threads = 256;
    auto cleanup_native_profile_events = [&]() {
        if (native_profile_this)
        {
            hipEventDestroy(native_prof_start);
            hipEventDestroy(native_prof_after_info);
            hipEventDestroy(native_prof_after_asm);
            hipEventDestroy(native_prof_after_scatter);
        }
    };

    const bool mtail_env_requested =
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_MTAIL", false);
    const bool mt128_legacy_requested =
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_MTAIL_MT128_LEGACY",
                           false);
    const bool mt128_source_requested =
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_MTAIL_SOURCE_MT128",
                           false);
    const bool mt128_legacy_allow_unsafe_direct_d = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_MTAIL_MT128_ALLOW_UNSAFE_DIRECT_D",
        false);
    const bool mtail_requested =
        !wgm8 && !direct_output && direct_input && mtail_env_requested &&
        (mt128_source_requested ||
         !(mt128_legacy_requested && !mt128_legacy_allow_unsafe_direct_d));
    if (direct_output && mtail_env_requested)
    {
        static std::atomic<bool> mtail_direct_output_logged{false};
        bool expected = false;
        if (rank == 0 &&
            mtail_direct_output_logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] ASM MT256 mtail ignored: direct-output epilogue "
                   "handles rank-tile tails in-kernel\n");
        }
    }
    if (direct_input && mtail_env_requested && mt128_legacy_requested &&
        !mt128_legacy_allow_unsafe_direct_d && !mt128_source_requested)
    {
        static std::atomic<bool> mt128_unsafe_logged{false};
        bool expected = false;
        if (rank == 0 &&
            mt128_unsafe_logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] ASM MT256 mtail MT128 legacy direct-D disabled: "
                   "existing MT128x256 kernel writes 256 output rows and "
                   "is unsafe for compact final output\n");
        }
    }
    if (mtail_requested)
    {
        int tail_packed_rows_per_tile = get_env_int_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_MTAIL_PACK_ROWS", 256);
        int tail_threshold = get_env_int_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_MTAIL_THRESHOLD", 128);
        tail_packed_rows_per_tile =
            std::max(1, std::min(tail_packed_rows_per_tile, 256));
        tail_threshold =
            std::max(1, std::min(std::min(tail_threshold, 255),
                                 tail_packed_rows_per_tile));
        const char *mtail_co_env =
            detail::get_vendor_env(
                "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_MTAIL_CO");
        const char *mtail_func_env =
            detail::get_vendor_env(
                "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_MTAIL_FUNC");
        const bool tail_co_set =
            mtail_co_env != nullptr && mtail_co_env[0] != '\0';
        const bool tail_func_set =
            mtail_func_env != nullptr && mtail_func_env[0] != '\0';
        const bool tail_uses_source_mt128 = mt128_source_requested;
        int tail_source_mt128_bn = get_env_int_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_MTAIL_SOURCE_MT128_BN", 256);
        tail_source_mt128_bn =
            (tail_source_mt128_bn == 128) ? 128 : 256;
        const bool tail_uses_default_v4 =
            !tail_uses_source_mt128 && !tail_co_set && !tail_func_set;
        const bool tail_requested_mt128_legacy = mt128_legacy_requested;
        const bool tail_uses_mt128_legacy =
            !tail_uses_source_mt128 && tail_requested_mt128_legacy &&
            mt128_legacy_allow_unsafe_direct_d;
        const bool tail_wait_all = get_env_bool_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_MTAIL_MT128_WAIT_ALL", false);

        if ((tail_uses_mt128_legacy &&
             (!tail_co_set || !tail_func_set || weight_kn == nullptr)) ||
            (tail_uses_source_mt128 && weight_kn == nullptr))
        {
            cleanup_native_profile_events();
            return false;
        }

        if (!(tail_uses_default_v4 && tail_packed_rows_per_tile != 256))
        {
            void *main_rank_tile_info = nullptr;
            void *tail_rank_tile_info = nullptr;
            void *mtail_counts = nullptr;
            const size_t split_rank_info_bytes =
                static_cast<size_t>(problem_count) *
                AG_RANK_TILE_INFO_STRIDE * sizeof(int);
            if (!ag_mt256_ensure_scratch(&g_mt256_native_mtail_main_rank_info,
                                         split_rank_info_bytes,
                                         &main_rank_tile_info) ||
                !ag_mt256_ensure_scratch(&g_mt256_native_mtail_tail_rank_info,
                                         split_rank_info_bytes,
                                         &tail_rank_tile_info) ||
                !ag_mt256_ensure_scratch(&g_mt256_native_mtail_counts,
                                         2 * sizeof(int),
                                         &mtail_counts))
            {
                cleanup_native_profile_events();
                return false;
            }

            err = hipMemsetAsync(mtail_counts, 0, 2 * sizeof(int), stream);
            if (err != hipSuccess)
            {
                cleanup_native_profile_events();
                return false;
            }
            const int split_blocks =
                std::max((problem_count + threads - 1) / threads, 1);
            hipLaunchKernelGGL(ag_mt256_split_mtail_rank_tile_info_kernel,
                               dim3(split_blocks),
                               dim3(threads),
                               0,
                               stream,
                               rank_tile_info,
                               static_cast<int *>(main_rank_tile_info),
                               static_cast<int *>(tail_rank_tile_info),
                               static_cast<int *>(mtail_counts),
                               problem_count,
                               tail_threshold);
            err = hipGetLastError();
            if (err != hipSuccess)
            {
                cleanup_native_profile_events();
                return false;
            }

            int host_counts[2] = {0, 0};
            err = hipMemcpyAsync(host_counts,
                                 mtail_counts,
                                 sizeof(host_counts),
                                 hipMemcpyDeviceToHost,
                                 stream);
            if (err == hipSuccess)
            {
                err = hipStreamSynchronize(stream);
            }
            if (err != hipSuccess)
            {
                cleanup_native_profile_events();
                return false;
            }

            const int main_problem_count = host_counts[0];
            const int tail_problem_count = host_counts[1];
            if (tail_problem_count > 0 &&
                main_problem_count + tail_problem_count > 0)
            {
                hipFunction_t main_function = nullptr;
                hipFunction_t tail_function = nullptr;
                if ((main_problem_count > 0 &&
                     !ag_mt256_get_native_contract_function(&main_function,
                                                            false)) ||
                    (!tail_uses_source_mt128 && tail_problem_count > 0 &&
                     !ag_mt256_get_native_contract_function(&tail_function,
                                                           true)))
                {
                    cleanup_native_profile_events();
                    return false;
                }

                int *main_contract_tile_info =
                    static_cast<int *>(contract_tile_info);
                int *tail_contract_tile_info =
                    main_contract_tile_info + 5 * main_problem_count;
                if (main_problem_count > 0)
                {
                    const int main_info_blocks = std::max(
                        (main_problem_count + threads - 1) / threads, 1);
                    hipLaunchKernelGGL(
                        ag_mt256_build_native_direct_input_info_kernel,
                        dim3(main_info_blocks),
                        dim3(threads),
                        0,
                        stream,
                        static_cast<const int *>(main_rank_tile_info),
                        main_contract_tile_info,
                        main_problem_count);
                }
                if (tail_problem_count > 0)
                {
                    const int tail_info_blocks = std::max(
                        (tail_problem_count + threads - 1) / threads, 1);
                    hipLaunchKernelGGL(
                        ag_mt256_build_native_direct_input_info_kernel,
                        dim3(tail_info_blocks),
                        dim3(threads),
                        0,
                        stream,
                        static_cast<const int *>(tail_rank_tile_info),
                        tail_contract_tile_info,
                        tail_problem_count);
                }
                err = hipGetLastError();
                if (err != hipSuccess)
                {
                    cleanup_native_profile_events();
                    return false;
                }
                if (native_profile_this)
                {
                    CUDA_CHECK(hipEventRecord(native_prof_after_info, stream));
                }

                auto launch_mtail_asm =
                    [&](hipFunction_t function,
                        int split_problem_count,
                        int64_t packed_row_offset,
                        const int *split_contract_tile_info) -> bool {
                    if (split_problem_count <= 0)
                    {
                        return true;
                    }
                    AgMt256ContractRankTileArgs args{};
                    args.input_buffer =
                        static_cast<const uint16_t *>(full_input);
                    args.weight = static_cast<const uint16_t *>(weight);
                    args.sorted_expert_offsets =
                        static_cast<const int32_t *>(sorted_gather_index);
                    args.output =
                        static_cast<uint16_t *>(packed_output) +
                        packed_row_offset * static_cast<int64_t>(n_dim);
                    args.barrier_ptr = rank_signals;
                    args.tile_info =
                        static_cast<const int32_t *>(split_contract_tile_info);
                    args.problem_count =
                        static_cast<uint32_t>(split_problem_count);
                    args.n_dim = static_cast<uint32_t>(n_dim);
                    args.k_dim = static_cast<uint32_t>(k_dim);
                    args.topk = static_cast<uint32_t>(topk);
                    args.rank = static_cast<uint32_t>(rank);
                    args.wait_signal_value =
                        static_cast<uint32_t>(signal_value);
                    args.epilogue_mode = 0u;
                    args.chunk_size_n = 0u;
                    args.num_done_groups = 0;
                    args.reserved = 0;
                    args.topk_stage = nullptr;
                    args.chunk_done_counters = nullptr;
                    args.chunk_done_flags = nullptr;

                    size_t arg_size = sizeof(args);
                    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                      &args,
                                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                      &arg_size,
                                      HIP_LAUNCH_PARAM_END};
                    err = hipExtModuleLaunchKernel(
                        function,
                        static_cast<uint32_t>(split_problem_count) * 768u,
                        static_cast<uint32_t>((n_dim + 255) / 256),
                        1,
                        768,
                        1,
                        1,
                        0,
                        stream,
                        nullptr,
                        config,
                        0,
                        0,
                        0);
                    return err == hipSuccess;
                };

                auto launch_source_mt128_tail =
                    [&](int split_problem_count,
                        const int *split_rank_tile_info) -> bool {
                    if (split_problem_count <= 0)
                    {
                        return true;
                    }

                    if (tail_source_mt128_bn == 128)
                    {
                        const int n_tiles = (n_dim + 127) / 128;
                        hipLaunchKernelGGL(
                            ag_group_gemm_tiled_128x128x32_mmac_bf16_rank_tile_kernel,
                            dim3(split_problem_count, n_tiles),
                            dim3(512),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight_kn),
                            sorted_token_ids,
                            sorted_gather_index,
                            split_rank_tile_info,
                            nullptr,
                            static_cast<const int *>(mtail_counts) + 1,
                            rank_signals,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            0,
                            0);
                    }
                    else
                    {
                        const int n_tiles = (n_dim + 255) / 256;
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_kernel<
                                false,
                                true,
                                false>),
                            dim3(split_problem_count, n_tiles),
                            dim3(1024),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight_kn),
                            sorted_token_ids,
                            sorted_gather_index,
                            split_rank_tile_info,
                            nullptr,
                            static_cast<const int *>(mtail_counts) + 1,
                            rank_signals,
                            nullptr,
                            nullptr,
                            nullptr,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            0,
                            0);
                    }
                    err = hipGetLastError();
                    return err == hipSuccess;
                };

                auto launch_mt128_legacy_tail =
                    [&](hipFunction_t function,
                        int split_problem_count,
                        const int *split_rank_tile_info) -> bool {
                    if (split_problem_count <= 0)
                    {
                        return true;
                    }

                    void *mt128_args = nullptr;
                    void *mt128_rowmap = nullptr;
                    const size_t mt128_args_bytes =
                        static_cast<size_t>(split_problem_count) *
                        AG_MT128_LEGACY_ARGS_BYTES;
                    const size_t mt128_rowmap_bytes =
                        static_cast<size_t>(split_problem_count) *
                        AG_MT128_LEGACY_ROWS * sizeof(int);
                    if (!ag_mt256_ensure_scratch(
                            &g_mt256_native_mtail_mt128_args,
                            mt128_args_bytes,
                            &mt128_args) ||
                        !ag_mt256_ensure_scratch(
                            &g_mt256_native_mtail_mt128_rowmap,
                            mt128_rowmap_bytes,
                            &mt128_rowmap))
                    {
                        return false;
                    }

                    err = hipMemsetAsync(
                        mt128_args, 0, mt128_args_bytes, stream);
                    if (err != hipSuccess)
                    {
                        return false;
                    }
                    hipLaunchKernelGGL(
                        ag_mt256_build_mtail_mt128_legacy_args_kernel,
                        dim3(split_problem_count),
                        dim3(threads),
                        0,
                        stream,
                        split_rank_tile_info,
                        sorted_token_ids,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight_kn),
                        static_cast<uint16_t *>(output),
                        static_cast<uint8_t *>(mt128_args),
                        static_cast<int *>(mt128_rowmap),
                        split_problem_count,
                        full_m,
                        n_dim,
                        k_dim);
                    err = hipGetLastError();
                    if (err != hipSuccess)
                    {
                        return false;
                    }

                    if (get_env_bool_local(
                            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_MTAIL_MT128_DEBUG",
                            false))
                    {
                        std::vector<int> host_rowmap(
                            static_cast<size_t>(split_problem_count) *
                            AG_MT128_LEGACY_ROWS);
                        std::vector<int> host_info(
                            static_cast<size_t>(split_problem_count) *
                            AG_RANK_TILE_INFO_STRIDE);
                        err = hipMemcpyAsync(
                            host_rowmap.data(),
                            mt128_rowmap,
                            mt128_rowmap_bytes,
                            hipMemcpyDeviceToHost,
                            stream);
                        if (err == hipSuccess)
                        {
                            err = hipMemcpyAsync(
                                host_info.data(),
                                split_rank_tile_info,
                                host_info.size() * sizeof(int),
                                hipMemcpyDeviceToHost,
                                stream);
                        }
                        if (err == hipSuccess)
                        {
                            err = hipStreamSynchronize(stream);
                        }
                        if (err != hipSuccess)
                        {
                            return false;
                        }

                        int min_row = full_m * 6;
                        int max_row = -1;
                        int invalid_count = 0;
                        int first_invalid_problem = -1;
                        int first_invalid_row = -1;
                        int first_invalid_value = 0;
                        for (int p = 0; p < split_problem_count; ++p)
                        {
                            for (int r = 0; r < AG_MT128_LEGACY_ROWS; ++r)
                            {
                                const int value =
                                    host_rowmap[static_cast<size_t>(p) *
                                                    AG_MT128_LEGACY_ROWS +
                                                r];
                                min_row = std::min(min_row, value);
                                max_row = std::max(max_row, value);
                                if (value < 0 || value >= full_m * 6)
                                {
                                    ++invalid_count;
                                    if (first_invalid_problem < 0)
                                    {
                                        first_invalid_problem = p;
                                        first_invalid_row = r;
                                        first_invalid_value = value;
                                    }
                                }
                            }
                        }
                        const int first_rows =
                            split_problem_count > 0
                                ? host_info[AG_RANK_TILE_ROWS]
                                : 0;
                        const int first_sorted =
                            split_problem_count > 0
                                ? host_info[AG_RANK_TILE_SORTED_START]
                                : 0;
                        printf("[hygon] MT128 legacy debug rank=%d problems=%d "
                               "full_m=%d n=%d k=%d rowmap_min=%d "
                               "rowmap_max=%d invalid=%d first_invalid=(%d,%d,%d) "
                               "first_tile=(sorted=%d rows=%d)\n",
                               rank,
                               split_problem_count,
                               full_m,
                               n_dim,
                               k_dim,
                               min_row,
                               max_row,
                               invalid_count,
                               first_invalid_problem,
                               first_invalid_row,
                               first_invalid_value,
                               first_sorted,
                               first_rows);
                    }

                    if (tail_wait_all)
                    {
                        hipLaunchKernelGGL(wait_rank_signals_kernel,
                                           dim3(1),
                                           dim3(1),
                                           0,
                                           stream,
                                           rank_signals,
                                           world_size,
                                           signal_value);
                        err = hipGetLastError();
                        if (err != hipSuccess)
                        {
                            return false;
                        }
                    }

                    AgMt128LegacyGroupedArgs legacy_args{};
                    legacy_args.grouped_sizes =
                        static_cast<uint32_t>(split_problem_count);
                    legacy_args.device_arguments = mt128_args;
                    legacy_args.args_ptr = nullptr;
                    legacy_args.wg_table = 0u;
                    legacy_args.gsu = 1u;
                    legacy_args.reserved = 0u;

                    const int n_tiles = (n_dim + 127) / 128;
                    size_t arg_size = sizeof(legacy_args);
                    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                      &legacy_args,
                                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                      &arg_size,
                                      HIP_LAUNCH_PARAM_END};
                    err = hipExtModuleLaunchKernel(
                        function,
                        static_cast<uint32_t>(split_problem_count) *
                            static_cast<uint32_t>(n_tiles) * 512u,
                        1,
                        1,
                        512,
                        1,
                        1,
                        0,
                        stream,
                        nullptr,
                        config,
                        0,
                        0,
                        0);
                    return err == hipSuccess;
                };

                const int64_t tail_packed_row_offset =
                    static_cast<int64_t>(main_problem_count) * 256;
                const bool asm_launch_ok =
                    launch_mtail_asm(main_function,
                                     main_problem_count,
                                     0,
                                     main_contract_tile_info) &&
                    (tail_uses_mt128_legacy
                         ? launch_mt128_legacy_tail(
                               tail_function,
                               tail_problem_count,
                               static_cast<const int *>(tail_rank_tile_info))
                     : tail_uses_source_mt128
                         ? launch_source_mt128_tail(
                               tail_problem_count,
                               static_cast<const int *>(tail_rank_tile_info))
                         : launch_mtail_asm(tail_function,
                                            tail_problem_count,
                                            tail_packed_row_offset,
                                            tail_contract_tile_info));
                if (!asm_launch_ok)
                {
                    cleanup_native_profile_events();
                    return false;
                }
                if (native_profile_this)
                {
                    CUDA_CHECK(hipEventRecord(native_prof_after_asm, stream));
                }

                auto scatter_mtail_output =
                    [&](const int *split_rank_tile_info,
                        int split_problem_count,
                        int packed_rows_per_tile,
                        int64_t packed_row_offset) -> bool {
                    if (split_problem_count <= 0)
                    {
                        return true;
                    }
                    if (vec_scatter && n_dim % 8 == 0)
                    {
                        const int n_vec = n_dim / 8;
                        const int64_t scatter_total =
                            static_cast<int64_t>(split_problem_count) *
                            packed_rows_per_tile * n_vec;
                        const int scatter_blocks = std::max(
                            1,
                            static_cast<int>(std::min<int64_t>(
                                (scatter_total + threads - 1) / threads,
                                4096)));
                        hipLaunchKernelGGL(
                            ag_mt256_scatter_native_contract_output_vec8_stride_kernel,
                            dim3(scatter_blocks),
                            dim3(threads),
                            0,
                            stream,
                            static_cast<const uint4 *>(packed_output) +
                                packed_row_offset * n_vec,
                            sorted_token_ids,
                            split_rank_tile_info,
                            static_cast<uint4 *>(output),
                            split_problem_count,
                            full_m * topk,
                            n_vec,
                            packed_rows_per_tile);
                    }
                    else
                    {
                        const int64_t scatter_total =
                            static_cast<int64_t>(split_problem_count) *
                            packed_rows_per_tile * n_dim;
                        const int scatter_blocks = std::max(
                            1,
                            static_cast<int>(std::min<int64_t>(
                                (scatter_total + threads - 1) / threads,
                                4096)));
                        hipLaunchKernelGGL(
                            ag_mt256_scatter_native_contract_output_stride_kernel,
                            dim3(scatter_blocks),
                            dim3(threads),
                            0,
                            stream,
                            static_cast<const uint16_t *>(packed_output) +
                                packed_row_offset *
                                    static_cast<int64_t>(n_dim),
                            sorted_token_ids,
                            split_rank_tile_info,
                            static_cast<uint16_t *>(output),
                            split_problem_count,
                            full_m * topk,
                            n_dim,
                            packed_rows_per_tile);
                    }
                    err = hipGetLastError();
                    return err == hipSuccess;
                };

                if (!scatter_mtail_output(
                        static_cast<const int *>(main_rank_tile_info),
                        main_problem_count,
                        256,
                        0) ||
                    (!(tail_uses_mt128_legacy || tail_uses_source_mt128) &&
                     !scatter_mtail_output(
                        static_cast<const int *>(tail_rank_tile_info),
                        tail_problem_count,
                        tail_packed_rows_per_tile,
                        tail_packed_row_offset)))
                {
                    cleanup_native_profile_events();
                    return false;
                }

                if (n_chunk_done_counts != nullptr &&
                    n_chunk_done_flags != nullptr)
                {
                    hipLaunchKernelGGL(
                        ag_mt256_mark_n_chunk_done_kernel,
                        dim3(std::max(active_n_chunk_groups, 1)),
                        dim3(1),
                        0,
                        stream,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        active_n_chunk_groups,
                        problem_count,
                        signal_value);
                    err = hipGetLastError();
                    if (err != hipSuccess)
                    {
                        cleanup_native_profile_events();
                        return false;
                    }
                }

                if (native_profile_this)
                {
                    CUDA_CHECK(
                        hipEventRecord(native_prof_after_scatter, stream));
                    CUDA_CHECK(hipEventSynchronize(native_prof_after_scatter));
                    float info_ms = 0.0f;
                    float asm_ms = 0.0f;
                    float scatter_ms = 0.0f;
                    float total_ms = 0.0f;
                    CUDA_CHECK(hipEventElapsedTime(
                        &info_ms, native_prof_start, native_prof_after_info));
                    CUDA_CHECK(hipEventElapsedTime(
                        &asm_ms, native_prof_after_info,
                        native_prof_after_asm));
                    CUDA_CHECK(hipEventElapsedTime(
                        &scatter_ms, native_prof_after_asm,
                        native_prof_after_scatter));
                    CUDA_CHECK(hipEventElapsedTime(
                        &total_ms, native_prof_start,
                        native_prof_after_scatter));
                    printf("[hygon] ASM MT256 mtail profile iter=%d "
                           "info_ms=%.3f asm_ms=%.3f scatter_ms=%.3f "
                           "total_ms=%.3f\n",
                           native_profile_iter,
                           info_ms,
                           asm_ms,
                           scatter_ms,
                           total_ms);
                    cleanup_native_profile_events();
                }

                static std::atomic<bool> mtail_logged{false};
                bool expected = false;
                if (rank == 0 &&
                    mtail_logged.compare_exchange_strong(expected, true))
                {
                    const int64_t mtail_packed_output_rows =
                        static_cast<int64_t>(main_problem_count) * 256 +
                        (tail_uses_source_mt128
                             ? 0
                             : static_cast<int64_t>(tail_problem_count) *
                                   (tail_uses_mt128_legacy
                                        ? AG_MT128_LEGACY_ROWS
                                        : tail_packed_rows_per_tile));
                    printf("[hygon] ASM MT256 mtail enabled: main_problems=%d "
                           "tail_problems=%d original_problems=%d "
                           "tail_threshold=%d tail_pack_rows=%d "
                           "packed_output_rows=%lld n=%d k=%d vec_scatter=%d "
                           "tail_default_v4=%d tail_mt128_legacy=%d "
                           "tail_source_mt128=%d tail_source_bn=%d\n",
                           main_problem_count,
                           tail_problem_count,
                           problem_count,
                           tail_threshold,
                           tail_uses_mt128_legacy
                               ? AG_MT128_LEGACY_ROWS
                               : (tail_uses_source_mt128
                                      ? 128
                                      : tail_packed_rows_per_tile),
                           static_cast<long long>(mtail_packed_output_rows),
                           n_dim,
                           k_dim,
                           (vec_scatter && n_dim % 8 == 0) ? 1 : 0,
                           tail_uses_default_v4 ? 1 : 0,
                           tail_uses_mt128_legacy ? 1 : 0,
                           tail_uses_source_mt128 ? 1 : 0,
                           tail_uses_source_mt128 ? tail_source_mt128_bn : 0);
                }
                return true;
            }
        }
    }

    const int info_count = std::max(problem_count, kMaxExpertsForContract + 1);
    const int info_blocks = std::max((info_count + threads - 1) / threads, 1);
    const bool compact_routed_rows =
        (variant == AgMt256AsmVariant::V36DynamicNNAilkDirectOutputNChunkSignal ||
         variant == AgMt256AsmVariant::V36DynamicNNAilkCompactDirectOutputNChunkSignal ||
         variant == AgMt256AsmVariant::V36DynamicNNAilkChunkDirectOutputNChunkSignal) &&
        direct_input && direct_output && !packed_input_enabled &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_NN_ROUTED_INPUT", false) &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_NN_COMPACT_ROWS", false);
    const int rank_signal_chunk_rows = active_signal_chunk_rows > 0
        ? active_signal_chunk_rows
        : get_env_int_local("FTRAIN_HYGON_AG_GROUP_GEMM_SIGNAL_CHUNK_ROWS", 0);
    const int local_rows = world_size > 0 ? full_m / world_size : 0;
    const int rank_signal_chunks_per_rank = active_chunks_per_rank > 0
        ? active_chunks_per_rank
        : (rank_signal_chunk_rows > 0
               ? (local_rows + rank_signal_chunk_rows - 1) /
                     rank_signal_chunk_rows
               : 0);
    const bool v36_rank_chunk_signal =
        direct_input && native_kn_weight && rank_tile_problem_info != nullptr &&
        rank_signal_chunk_rows > 0 && rank_signal_chunks_per_rank > 0 &&
        (get_env_bool_local(
             "FTRAIN_HYGON_AG_GROUP_GEMM_V40_PARTIAL_READY_RANK_CHUNKS", false) ||
         get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_RAGGED_RING_CHUNK", true));
    const bool ragged_ring_chunk = v36_rank_chunk_signal && ragged_m &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_RAGGED_RING_CHUNK", true);
    const bool compact_payload_fenced = compact_routed_rows;
    if (direct_input)
    {
        if (v36_rank_chunk_signal)
        {
            if (ragged_ring_chunk)
            {
                hipLaunchKernelGGL(
                    ag_mt256_build_native_direct_input_chunk_info_kernel,
                    dim3(info_blocks),
                    dim3(threads),
                    0,
                    stream,
                    rank_tile_problem_info,
                    static_cast<int *>(contract_tile_info),
                    problem_count);
            }
            else
            {
                hipLaunchKernelGGL(
                    ag_mt256_build_native_direct_input_rank_chunk_info_kernel,
                    dim3(info_blocks),
                    dim3(threads),
                    0,
                    stream,
                    rank_tile_problem_info,
                    static_cast<int *>(contract_tile_info),
                    problem_count,
                    rank_signal_chunks_per_rank,
                    rank,
                    world_size,
                    get_env_int_local(
                        "FTRAIN_HYGON_AG_GROUP_GEMM_RAGGED_CHUNK_SLOT_BASE", 0));
            }
        }
        else if (native_kn_chunk || native_kn_direct_chunk)
        {
            hipLaunchKernelGGL(
                ag_mt256_build_native_direct_input_chunk_info_kernel,
                dim3(info_blocks),
                dim3(threads),
                0,
                stream,
                rank_tile_problem_info,
                static_cast<int *>(contract_tile_info),
                problem_count);
        }
        else if (native_kn_stage_signal_map)
        {
            hipLaunchKernelGGL(
                ag_mt256_build_native_direct_input_stage_info_kernel,
                dim3(info_blocks),
                dim3(threads),
                0,
                stream,
                rank_tile_info,
                               static_cast<int *>(contract_tile_info),
                               problem_count,
                               rank,
                               world_size,
                               get_env_int_local(
                                   "FTRAIN_HYGON_AG_GROUP_GEMM_RAGGED_FIXED_BODY_SLOT_BASE",
                                   0));
        }
        else if (compact_routed_rows)
        {
            hipLaunchKernelGGL(
                ag_mt256_build_native_direct_input_compact_info_kernel,
                dim3(info_blocks),
                dim3(threads),
                0,
                stream,
                rank_tile_info,
                sorted_gather_index,
                static_cast<int *>(contract_tile_info),
                problem_count,
                full_m,
                world_size,
                std::max(1,
                         std::min(get_env_int_local(
                                      "FTRAIN_HYGON_AG_GROUP_GEMM_NN_COMPACT_MAX_RANK_SEGMENTS",
                                      2),
                                  2)));
        }
        else
        {
            hipLaunchKernelGGL(ag_mt256_build_native_direct_input_info_kernel,
                               dim3(info_blocks),
                               dim3(threads),
                               0,
                               stream,
                               rank_tile_info,
                               static_cast<int *>(contract_tile_info),
                               problem_count);
        }
    }
    else
    {
        hipLaunchKernelGGL(ag_mt256_build_native_contract_info_kernel,
                           dim3(info_blocks),
                           dim3(threads),
                           0,
                           stream,
                           rank_tile_info,
                           static_cast<int *>(contract_tile_info),
                           static_cast<int *>(sorted_expert_offsets),
                           problem_count,
                           kMaxExpertsForContract);
    }
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        return false;
    }
    if (get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_DEBUG_RANK_CHUNK_INFO", false))
    {
        printf("[hygon] legacy rank-chunk state rank=%d direct=%d "
               "native_kn=%d problem_info=%p rows=%d chunks=%d enabled=%d\n",
               rank,
               direct_input ? 1 : 0,
               native_kn_weight ? 1 : 0,
               rank_tile_problem_info,
               rank_signal_chunk_rows,
               rank_signal_chunks_per_rank,
               v36_rank_chunk_signal ? 1 : 0);
        fflush(stdout);
    }
    if (compact_routed_rows && rank == 0 &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_NN_COMPACT_DEBUG", false))
    {
        static std::atomic<bool> compact_metadata_logged{false};
        bool expected = false;
        if (compact_metadata_logged.compare_exchange_strong(expected, true))
        {
            std::vector<int> host_tile_info(
                static_cast<size_t>(problem_count) * 5);
            std::vector<int> host_sorted_gather_index;
            err = hipMemcpyAsync(host_tile_info.data(),
                                 contract_tile_info,
                                 host_tile_info.size() * sizeof(int),
                                 hipMemcpyDeviceToHost,
                                 stream);
            if (err == hipSuccess)
            {
                err = hipStreamSynchronize(stream);
            }
            if (err == hipSuccess)
            {
                int max_sorted_end = full_m;
                for (int tile = 0; tile < problem_count; ++tile)
                {
                    const int sorted_start =
                        host_tile_info[problem_count + tile];
                    const int rows = host_tile_info[4 * problem_count + tile] &
                                     0xffff;
                    if (sorted_start >= 0 && rows > 0)
                    {
                        max_sorted_end = std::max(max_sorted_end,
                                                  sorted_start + rows);
                    }
                }
                host_sorted_gather_index.resize(
                    static_cast<size_t>(max_sorted_end));
                err = hipMemcpyAsync(host_sorted_gather_index.data(),
                                     sorted_gather_index,
                                     host_sorted_gather_index.size() *
                                         sizeof(int),
                                     hipMemcpyDeviceToHost,
                                     stream);
            }
            if (err == hipSuccess)
            {
                err = hipStreamSynchronize(stream);
            }
            if (err != hipSuccess)
            {
                return false;
            }
            int one_rank = 0;
            int two_rank = 0;
            int rowmap_fallback = 0;
            for (int tile = 0; tile < problem_count; ++tile)
            {
                const int encoded_rows = host_tile_info[
                    4 * problem_count + tile];
                const int rows = encoded_rows & 0xffff;
                const int split = static_cast<unsigned int>(encoded_rows) >> 16;
                if (split == 0)
                {
                    ++rowmap_fallback;
                }
                else if (split >= rows)
                {
                    ++one_rank;
                }
                else
                {
                    ++two_rank;
                }
            }
            printf("[hygon] NN compact metadata: problems=%d one_rank=%d "
                   "two_rank=%d rowmap_fallback=%d\\n",
                   problem_count,
                   one_rank,
                   two_rank,
                   rowmap_fallback);
            for (int tile = 0; tile < std::min(problem_count, 3); ++tile)
            {
                const int sorted_start = host_tile_info[problem_count + tile];
                const int rows = host_tile_info[4 * problem_count + tile] &
                                 0xffff;
                const int sample_count = std::min(rows, 6);
                printf("[hygon] NN compact tile=%d start=%d rows=%d gather=",
                       tile,
                       sorted_start,
                       rows);
                for (int row = 0; row < sample_count; ++row)
                {
                    printf("%s%d", row == 0 ? "" : ",",
                           host_sorted_gather_index[sorted_start + row]);
                }
                printf("\\n");
            }
            if (get_env_int_local("FTRAIN_HYGON_AG_GROUP_GEMM_NN_COMPACT_DEBUG", 0) >= 2)
            {
                for (int tile = 0; tile < problem_count; ++tile)
                {
                    const int encoded_rows = host_tile_info[
                        4 * problem_count + tile];
                    const int rows = encoded_rows & 0xffff;
                    const int split = static_cast<unsigned int>(encoded_rows) >> 16;
                    printf("[hygon] NN compact detail tile=%d start=%d rows=%d "
                           "split=%d rank=%d..%d gather0=%d gather_last=%d\\n",
                           tile,
                           host_tile_info[problem_count + tile],
                           rows,
                           split,
                           host_tile_info[2 * problem_count + tile],
                           host_tile_info[3 * problem_count + tile],
                           host_sorted_gather_index[
                               host_tile_info[problem_count + tile]],
                           host_sorted_gather_index[
                               host_tile_info[problem_count + tile] + rows - 1]);
                }
            }
        }
    }
    if (v36_scale_in_gemm)
    {
        err = hipMemcpyAsync(static_cast<uint8_t *>(contract_tile_info) +
                                 tile_info_bytes,
                             router_scale,
                             router_scale_bytes,
                             hipMemcpyDeviceToDevice,
                             stream);
        if (err != hipSuccess)
        {
            return false;
        }
    }
    if (v36_expert_nmajor_wgm1_direct)
    {
        const bool default_chunk_major_task_map =
            n_tiles >= 32 && n_chunk_done_counts != nullptr &&
            !get_env_bool_local("FTRAIN_HYGON_MOE_RS_UPSTREAM_ONLY", false);
        const bool chunk_major_task_map = get_env_bool_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_WGM1_CHUNK_MAJOR_TASK_MAP",
            default_chunk_major_task_map);
        const int logical_chunk_n_tiles =
            std::max(1, active_n_chunk_size / 256);
        int auto_task_group_n_tiles = n_tiles;
        if (chunk_major_task_map)
        {
            // Grow the locality window smoothly with N, align it to complete
            // logical chunks, and retain at least four publication windows.
            const int locality_target = std::max(
                logical_chunk_n_tiles,
                (n_tiles * std::max(n_tiles - 48, 0)) / 512);
            const int max_group_n_tiles = std::max(
                logical_chunk_n_tiles,
                (n_tiles / 4 / logical_chunk_n_tiles) *
                    logical_chunk_n_tiles);
            auto_task_group_n_tiles = std::min(
                max_group_n_tiles,
                std::max(
                    logical_chunk_n_tiles,
                    (locality_target / logical_chunk_n_tiles) *
                        logical_chunk_n_tiles));
        }
        const int task_group_n_tiles = std::max(
            1,
            std::min(
                get_env_int_local(
                    "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_WGM1_TASK_GROUP_N_TILES",
                    auto_task_group_n_tiles),
                n_tiles));
        int readiness_window_stages = get_env_int_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_WGM1_READINESS_WINDOW_STAGES", 0);
        readiness_window_stages = std::max(
            0, std::min(readiness_window_stages, std::max(world_size, 1)));
        hipLaunchKernelGGL(
            ag_mt256_build_expert_n_major_wgm1_task_map_kernel,
            dim3(info_blocks),
            dim3(threads),
            0,
            stream,
            rank_tile_info,
            static_cast<int32_t *>(v36_direct_control) +
                v36_control_prefix_words,
            problem_count,
            n_tiles,
            task_group_n_tiles,
            readiness_window_stages);
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            return false;
        }
    }
    if (native_profile_this)
    {
        CUDA_CHECK(hipEventRecord(native_prof_after_info, stream));
    }
    bool vec_pack_active = false;
    const int64_t pack_input_rows64 = packed_input_enabled
        ? static_cast<int64_t>(full_m) * static_cast<int64_t>(topk)
        : static_cast<int64_t>(full_m);
    if (pack_input_rows64 > std::numeric_limits<int>::max())
    {
        return false;
    }
    const int pack_input_rows = static_cast<int>(pack_input_rows64);
    if (problem_count > 0)
    {
        hipFunction_t function = nullptr;
        if (!ag_mt256_get_native_contract_function(
            &function, false, variant))
        {
            return false;
        }

        AgMt256ContractRankTileArgs args{};
        const int requested_pack_k_tile = get_env_int_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_TILE_WAIT_PACK_K", 512);
        const int active_pack_k_tile =
            requested_pack_k_tile > 0 ? requested_pack_k_tile : 512;
        const int streaming_pack_k_slices = native_kn_streaming_pack
            ? (k_dim + active_pack_k_tile - 1) / active_pack_k_tile
            : 0;
        args.input_buffer = direct_input && !packed_input_enabled
                                ? static_cast<const uint16_t *>(full_input)
                                : static_cast<const uint16_t *>(packed_input);
        /* TN variants consume [expert][n_dim][k_dim].  The isolated
         * NN/Ailk variant consumes native [expert][k_dim][n_dim]. */
        args.weight = static_cast<const uint16_t *>(weight);
        args.sorted_expert_offsets = direct_input
            ? static_cast<const int32_t *>(sorted_gather_index)
            : static_cast<const int32_t *>(sorted_expert_offsets);
        args.output = direct_output ? static_cast<uint16_t *>(output)
                                    : static_cast<uint16_t *>(packed_output);
        args.barrier_ptr = native_kn_streaming_pack
            ? static_cast<const int32_t *>(streaming_pack_state)
            : rank_signals;
        args.tile_info = static_cast<const int32_t *>(contract_tile_info);
        args.problem_count = static_cast<uint32_t>(problem_count);
        args.n_dim = static_cast<uint32_t>(n_dim);
        args.k_dim = static_cast<uint32_t>(k_dim);
        args.topk = static_cast<uint32_t>(topk);
        // The streaming experiment removes owner-rank polling from the GEMM
        // scheduler.  Its otherwise-unused rank slot transports the exact
        // pack-slice threshold while preserving the 112-byte ABI.
        args.rank = static_cast<uint32_t>(
            native_kn_streaming_pack ? streaming_pack_k_slices : rank);
        args.wait_signal_value = static_cast<uint32_t>(signal_value);
        args.epilogue_mode = direct_output ? 1u : 0u;
        const bool direct_output_nchunk_signal =
            direct_output && n_chunk_done_counts != nullptr &&
            n_chunk_done_flags != nullptr &&
            (ag_mt256_variant_nchunk_signal(variant) ||
             (!explicit_variant && get_env_bool_local(
                  "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT_N_CHUNK_SIGNAL",
                  false)));
        args.chunk_size_n = direct_output_nchunk_signal
            ? static_cast<uint32_t>(active_n_chunk_size)
            : 0u;
        args.num_done_groups = direct_output_nchunk_signal
            ? static_cast<uint32_t>(active_n_chunk_groups)
            : 0u;
        // Bit 0 selects v36 group-major launch.  Bit 4 is the proven strict
        // Native-KN routed-input ABI. Bit 6 is the isolated compact variant:
        // merged tiles use at most two uniform physical row bases, with a
        // per-row fallback for three-or-more-rank tiles.  Bit 6 intentionally
        // avoids bit 5, which belongs to the static-rank scheduler.
        const bool routed_input =
            (ag_mt256_variant_nn_ailk(variant) ||
             get_env_bool_local(
                 "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_TNLOADER_MEGATRON", false)) &&
            (get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_NN_ROUTED_INPUT",
                                false) ||
             get_env_bool_local(
                 "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_TNLOADER_MEGATRON", false)) &&
            (get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_STRICT_RANK_TILE", false) ||
             compact_routed_rows);
        const bool static_rank_schedule =
            variant == AgMt256AsmVariant::V36DynamicNNAilkStaticDirectOutputNChunkSignal;
        const int native_chunk_rows =
            std::max(1, get_env_int_local("FTRAIN_HYGON_AG_GROUP_GEMM_SIGNAL_CHUNK_ROWS", 128));
        const int native_local_rows = world_size > 0 ? full_m / world_size : 0;
        const int native_chunks_per_rank =
            native_local_rows > 0
                ? (native_local_rows + native_chunk_rows - 1) / native_chunk_rows
                : 0;
        // Bit 8 tells the compact assembly that the host launcher has already
        // fenced all peer payloads with remote_ready_event_.  In that mode
        // owner-signal polling and the dynamic claim are redundant: the
        // launch coordinates provide a unique problem/tile owner, so leaving
        // both out avoids global loads and the per-tile CAS.  Keep it clear for
        // the no-fence experiment path.
        const bool native_chunk_metadata =
            native_kn_chunk || native_kn_direct_chunk;
        args.reserved = (v36_group_major ? 1u : 0u) |
                        (routed_input ? (1u << 4) : 0u) |
                        (compact_routed_rows ? (1u << 6) : 0u) |
                        (static_rank_schedule ? (1u << 5) : 0u) |
                        ((native_chunk_metadata || ragged_ring_chunk ||
                          v36_rank_chunk_signal)
                             ? (1u << 7)
                             : 0u) |
                        (compact_payload_fenced ? (1u << 8) : 0u) |
                        (get_env_bool_local(
                             "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_READY_MASK",
                             false)
                             ? (1u << 9)
                             : 0u) |
                        (native_chunk_metadata
                             ? (static_cast<uint32_t>(native_chunks_per_rank)
                                << 16)
                             : 0u);
        if (rank == 0 &&
            detail::get_vendor_bool_env(
                "FTRAIN_HYGON_AG_GROUP_GEMM_DEBUG_ROUTE", false))
        {
            static std::atomic<bool> routed_logged{false};
            bool expected = false;
            if (routed_logged.compare_exchange_strong(expected, true))
            {
                printf("[hygon] NN routed-input ABI=%d compact=%d strict_env=%d\\n",
                       routed_input ? 1 : 0,
                       compact_routed_rows ? 1 : 0,
                       get_env_bool_local(
                           "FTRAIN_HYGON_AG_GROUP_GEMM_STRICT_RANK_TILE", false)
                           ? 1
                           : 0);
            }
        }
        args.topk_stage =
            direct_output ? const_cast<int *>(sorted_token_ids) : nullptr;
        args.chunk_done_counters = v36_dynamic_direct
            ? static_cast<int32_t *>(v36_direct_control) +
                  v36_control_prefix_words
            : (direct_output_nchunk_signal ? n_chunk_done_counts : nullptr);
        args.chunk_done_flags =
            direct_output_nchunk_signal ? n_chunk_done_flags : nullptr;

        if (native_kn_streaming_pack)
        {
            const bool use_vec_pack =
                (k_dim % 8 == 0) && (active_pack_k_tile % 8 == 0);
            if (!use_vec_pack || streaming_pack_k_slices <= 0)
            {
                return false;
            }
            vec_pack_active = true;
            const int k_vec_dim = k_dim / 8;
            const int k_tile_vec = active_pack_k_tile / 8;
            const int seed_problems = std::max(
                1,
                std::min(
                    problem_count,
                    get_env_int_local(
                        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_STREAM_PACK_SEED_PROBLEMS",
                        16)));
            hipLaunchKernelGGL(
                ag_mt256_pack_native_rank_tiles_streaming_tile_wait_vec8_kernel,
                dim3(streaming_pack_k_slices, seed_problems),
                dim3(threads),
                0,
                g_mt256_streaming_pack_context.stream,
                static_cast<const uint4 *>(full_input),
                sorted_gather_index,
                rank_tile_info,
                rank_signals,
                static_cast<uint4 *>(packed_input),
                static_cast<int *>(streaming_pack_state),
                0,
                problem_count,
                pack_input_rows,
                k_vec_dim,
                world_size,
                signal_value,
                k_tile_vec);
            if (hipGetLastError() != hipSuccess)
            {
                return false;
            }
            err = hipEventRecord(
                g_mt256_streaming_pack_context.seed_ready,
                g_mt256_streaming_pack_context.stream);
            if (err != hipSuccess)
            {
                return false;
            }
            const int tail_problem_count = problem_count - seed_problems;
            if (tail_problem_count > 0)
            {
                const int tail_tasks =
                    tail_problem_count * streaming_pack_k_slices;
                const int tail_blocks = std::max(
                    1,
                    std::min(
                        tail_tasks,
                        get_env_int_local(
                            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_STREAM_PACK_BLOCKS",
                            24)));
                hipLaunchKernelGGL(
                    ag_mt256_pack_native_rank_tiles_streaming_throttled_vec8_kernel,
                    dim3(tail_blocks),
                    dim3(threads),
                    0,
                    g_mt256_streaming_pack_context.stream,
                    static_cast<const uint4 *>(full_input),
                    sorted_gather_index,
                    rank_tile_info,
                    rank_signals,
                    static_cast<uint4 *>(packed_input),
                    static_cast<int *>(streaming_pack_state),
                    seed_problems,
                    problem_count,
                    pack_input_rows,
                    k_vec_dim,
                    world_size,
                    signal_value,
                    k_tile_vec,
                    streaming_pack_k_slices);
                if (hipGetLastError() != hipSuccess)
                {
                    return false;
                }
            }
            // Ensure the scheduler starts with claimable work.  It then polls
            // only per-problem pack counters while the rest of the independent
            // stream continues packing.
            err = hipStreamWaitEvent(
                stream, g_mt256_streaming_pack_context.seed_ready, 0);
            if (err != hipSuccess)
            {
                return false;
            }
        }
        else if (direct_input && !packed_input_enabled)
        {
            vec_pack_active = false;
        }
        else if (tile_wait_pack || packed_input_enabled)
        {
            const int active_k_tile = active_pack_k_tile;
            const bool use_vec_pack =
                vec_pack_requested && (k_dim % 8 == 0) &&
                (active_k_tile % 8 == 0);
            vec_pack_active = use_vec_pack;
            if (use_vec_pack)
            {
                const int k_vec_dim = k_dim / 8;
                const int k_tile_vec = active_k_tile / 8;
                const int k_slices =
                    (k_vec_dim + k_tile_vec - 1) / k_tile_vec;
                hipLaunchKernelGGL(
                    ag_mt256_pack_native_rank_tiles_tile_wait_vec8_kernel,
                    dim3(problem_count, k_slices),
                    dim3(threads),
                    0,
                    stream,
                    static_cast<const uint4 *>(full_input),
                    sorted_gather_index,
                    rank_tile_info,
                    rank_signals,
                    static_cast<uint4 *>(packed_input),
                    problem_count,
                    pack_input_rows,
                    k_vec_dim,
                    world_size,
                    signal_value,
                    k_tile_vec);
            }
            else
            {
                const int k_slices =
                    (k_dim + active_k_tile - 1) / active_k_tile;
                hipLaunchKernelGGL(
                    ag_mt256_pack_native_rank_tiles_tile_wait_kernel,
                    dim3(problem_count, k_slices),
                    dim3(threads),
                    0,
                    stream,
                    static_cast<const uint16_t *>(full_input),
                    sorted_gather_index,
                    rank_tile_info,
                    rank_signals,
                    static_cast<uint16_t *>(packed_input),
                    problem_count,
                    pack_input_rows,
                    k_dim,
                    world_size,
                    signal_value,
                    active_k_tile);
            }
        }
        else
        {
            const int64_t pack_total = packed_rows * k_dim;
            const int pack_blocks = std::max(
                1,
                static_cast<int>(std::min<int64_t>(
                    (pack_total + threads - 1) / threads, 4096)));
            hipLaunchKernelGGL(ag_mt256_pack_native_rank_tiles_kernel,
                               dim3(pack_blocks),
                               dim3(threads),
                               0,
                               stream,
                               static_cast<const uint16_t *>(full_input),
                               sorted_gather_index,
                               rank_tile_info,
                               static_cast<uint16_t *>(packed_input),
                               problem_count,
                               pack_input_rows,
                               k_dim);
        }
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            return false;
        }

        size_t arg_size = sizeof(args);
        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                          &args,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &arg_size,
                          HIP_LAUNCH_PARAM_END};
        const uint32_t n_tiles_per_group =
            static_cast<uint32_t>(active_n_chunk_size / 256);
        /* Layout and compute-body geometry are independent: native KN/Ailk
         * direct siblings use eight waves, while the TN-derived loader keeps
         * the TN body's twelve-wave (four producer + eight MMAC) workgroup. */
        const uint32_t asm_workgroup_size =
            ag_mt256_variant_nn_ailk(variant) && !v19_tnloader_body &&
                    !native_kn_packed_d64_body
                ? 512u
                : 768u;
        const uint64_t launch_groups_x64 =
            v36_expert_nmajor_wgm1_direct
                ? static_cast<uint64_t>(problem_count) *
                      static_cast<uint64_t>(n_tiles)
            : v36_group_major ? n_tiles_per_group
            : grouped_wgm     ? n_tiles_per_group
            : wgm8            ? static_cast<uint32_t>(n_tiles)
                              : static_cast<uint32_t>(problem_count);
        const uint64_t launch_groups_y64 =
            v36_expert_nmajor_wgm1_direct
                ? 1u
            : v36_group_major
                ? static_cast<uint64_t>(problem_count) *
                      static_cast<uint64_t>(active_n_chunk_groups)
            : grouped_wgm
                ? static_cast<uint64_t>(problem_count) *
                      static_cast<uint64_t>(active_n_chunk_groups)
            : wgm8 ? static_cast<uint32_t>(problem_count)
                   : static_cast<uint32_t>(n_tiles);
        if (launch_groups_x64 == 0 || launch_groups_y64 == 0 ||
            launch_groups_x64 >
                static_cast<uint64_t>(UINT32_MAX) / asm_workgroup_size ||
            launch_groups_y64 > UINT32_MAX)
        {
            static std::atomic<bool> logged{false};
            bool expected = false;
            if (rank == 0 && logged.compare_exchange_strong(expected, true))
            {
                printf("[hygon] ASM MT256 launch skipped: grid exceeds HIP "
                       "32-bit dimensions (groups=%llu x %llu, wg=%u)\n",
                       static_cast<unsigned long long>(launch_groups_x64),
                       static_cast<unsigned long long>(launch_groups_y64),
                       asm_workgroup_size);
            }
            return false;
        }
        const uint32_t launch_groups_x =
            static_cast<uint32_t>(launch_groups_x64);
        const uint32_t launch_groups_y =
            static_cast<uint32_t>(launch_groups_y64);
        err = hipExtModuleLaunchKernel(
            function,
            launch_groups_x * asm_workgroup_size,
            launch_groups_y,
            1,
            asm_workgroup_size,
            1,
            1,
            0,
            stream,
            nullptr,
            config,
            0,
            0,
            0);
        if (err != hipSuccess)
        {
            return false;
        }
        if (v36_dynamic_direct &&
            get_env_bool_local(
                "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_DEBUG_CONTROL", false))
        {
            const size_t claim_count = static_cast<size_t>(problem_count) *
                                       static_cast<size_t>(n_tiles);
            const size_t control_count =
                claim_count + static_cast<size_t>(active_n_chunk_groups);
            std::vector<int32_t> host_control(control_count);
            err = hipMemcpyAsync(host_control.data(),
                                 static_cast<int32_t *>(v36_direct_control) +
                                     v36_control_prefix_words,
                                 control_count * sizeof(int32_t),
                                 hipMemcpyDeviceToHost,
                                 stream);
            if (err == hipSuccess)
            {
                err = hipStreamSynchronize(stream);
            }
            if (err != hipSuccess)
            {
                return false;
            }

            if (v36_expert_nmajor_wgm1_direct)
            {
                std::vector<uint8_t> seen(claim_count, 0);
                size_t invalid_tasks = 0;
                size_t duplicate_tasks = 0;
                for (size_t i = 0; i < claim_count; ++i)
                {
                    const uint32_t encoded =
                        static_cast<uint32_t>(host_control[i]);
                    const uint32_t task_n_tile =
                        encoded & (kAgMt256Wgm1MaxNTiles - 1);
                    const uint32_t task_problem =
                        encoded >> kAgMt256Wgm1TaskNTileBits;
                    if (task_problem >= static_cast<uint32_t>(problem_count) ||
                        task_n_tile >= static_cast<uint32_t>(n_tiles))
                    {
                        ++invalid_tasks;
                        continue;
                    }
                    const size_t logical_task =
                        static_cast<size_t>(task_problem) * n_tiles +
                        task_n_tile;
                    duplicate_tasks += seen[logical_task] != 0 ? 1 : 0;
                    seen[logical_task] = 1;
                }
                size_t missing_tasks = 0;
                for (uint8_t present : seen)
                {
                    missing_tasks += present == 0 ? 1 : 0;
                }
                printf("[hygon] ASM MT256 v36 WGM1 control: tasks=%zu "
                       "invalid=%zu duplicate=%zu missing=%zu counters=",
                       claim_count,
                       invalid_tasks,
                       duplicate_tasks,
                       missing_tasks);
            }
            else
            {
                size_t zero_claims = 0;
                size_t one_claims = 0;
                size_t other_claims = 0;
                for (size_t i = 0; i < claim_count; ++i)
                {
                    if (host_control[i] == 0)
                    {
                        ++zero_claims;
                    }
                    else if (host_control[i] == 1)
                    {
                        ++one_claims;
                    }
                    else
                    {
                        ++other_claims;
                    }
                }
                printf("[hygon] ASM MT256 v36 control: claims=%zu ones=%zu "
                       "zeros=%zu other=%zu counters=",
                       claim_count,
                       one_claims,
                       zero_claims,
                       other_claims);
            }
            for (int group = 0; group < active_n_chunk_groups; ++group)
            {
                printf("%s%d",
                       group == 0 ? "[" : ",",
                       host_control[claim_count + static_cast<size_t>(group)]);
            }
            printf("]\n");
        }
        if (native_profile_this)
        {
            CUDA_CHECK(hipEventRecord(native_prof_after_asm, stream));
        }
        bool n_chunk_marked_by_scatter = direct_output_nchunk_signal;
        if (direct_output)
        {
            err = hipSuccess;
        }
        else if (vec_scatter && n_dim % 8 == 0)
        {
            const int n_vec = n_dim / 8;
            if (n_chunk_done_counts != nullptr &&
                n_chunk_done_flags != nullptr)
            {
                hipLaunchKernelGGL(
                    ag_mt256_scatter_native_contract_output_vec8_nchunk_done_kernel,
                    dim3(problem_count, active_n_chunk_groups),
                    dim3(threads),
                    0,
                    stream,
                    static_cast<const uint4 *>(packed_output),
                    sorted_token_ids,
                    rank_tile_info,
                    static_cast<uint4 *>(output),
                    problem_count,
                    full_m * topk,
                    n_vec,
                    active_n_chunk_size,
                    n_chunk_done_counts,
                    n_chunk_done_flags,
                    signal_value);
                n_chunk_marked_by_scatter = true;
            }
            else
            {
                const int64_t scatter_total = packed_rows * n_vec;
                const int scatter_blocks = std::max(
                    1,
                    static_cast<int>(std::min<int64_t>(
                        (scatter_total + threads - 1) / threads, 4096)));
                hipLaunchKernelGGL(
                    ag_mt256_scatter_native_contract_output_vec8_kernel,
                    dim3(scatter_blocks),
                    dim3(threads),
                    0,
                    stream,
                    static_cast<const uint4 *>(packed_output),
                    sorted_token_ids,
                    rank_tile_info,
                    static_cast<uint4 *>(output),
                    problem_count,
                    full_m * topk,
                    n_vec);
            }
        }
        else
        {
            if (n_chunk_done_counts != nullptr &&
                n_chunk_done_flags != nullptr)
            {
                hipLaunchKernelGGL(
                    ag_mt256_scatter_native_contract_output_nchunk_done_kernel,
                    dim3(problem_count, active_n_chunk_groups),
                    dim3(threads),
                    0,
                    stream,
                    static_cast<const uint16_t *>(packed_output),
                    sorted_token_ids,
                    rank_tile_info,
                    static_cast<uint16_t *>(output),
                    problem_count,
                    full_m * topk,
                    n_dim,
                    active_n_chunk_size,
                    n_chunk_done_counts,
                    n_chunk_done_flags,
                    signal_value);
                n_chunk_marked_by_scatter = true;
            }
            else
            {
                const int64_t scatter_total = packed_rows * n_dim;
                const int scatter_blocks = std::max(
                    1,
                    static_cast<int>(std::min<int64_t>(
                        (scatter_total + threads - 1) / threads, 4096)));
                hipLaunchKernelGGL(
                    ag_mt256_scatter_native_contract_output_kernel,
                    dim3(scatter_blocks),
                    dim3(threads),
                    0,
                    stream,
                    static_cast<const uint16_t *>(packed_output),
                    sorted_token_ids,
                    rank_tile_info,
                    static_cast<uint16_t *>(output),
                    problem_count,
                    full_m * topk,
                    n_dim);
            }
        }
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            return false;
        }

        if (!n_chunk_marked_by_scatter &&
            n_chunk_done_counts != nullptr && n_chunk_done_flags != nullptr)
        {
            hipLaunchKernelGGL(ag_mt256_mark_n_chunk_done_kernel,
                               dim3(std::max(active_n_chunk_groups, 1)),
                               dim3(1),
                               0,
                               stream,
                               n_chunk_done_counts,
                               n_chunk_done_flags,
                               active_n_chunk_groups,
                               problem_count,
                               signal_value);
            err = hipGetLastError();
            if (err != hipSuccess)
            {
                return false;
            }
        }
    }
    else if (native_profile_this)
    {
        CUDA_CHECK(hipEventRecord(native_prof_after_asm, stream));
    }

    if (native_profile_this)
    {
        CUDA_CHECK(hipEventRecord(native_prof_after_scatter, stream));
        CUDA_CHECK(hipEventSynchronize(native_prof_after_scatter));
        float info_ms = 0.0f;
        float asm_ms = 0.0f;
        float scatter_ms = 0.0f;
        float total_ms = 0.0f;
        CUDA_CHECK(hipEventElapsedTime(
            &info_ms, native_prof_start, native_prof_after_info));
        CUDA_CHECK(hipEventElapsedTime(
            &asm_ms, native_prof_after_info, native_prof_after_asm));
        CUDA_CHECK(hipEventElapsedTime(
            &scatter_ms, native_prof_after_asm, native_prof_after_scatter));
        CUDA_CHECK(hipEventElapsedTime(
            &total_ms, native_prof_start, native_prof_after_scatter));
        printf("[hygon] ASM MT256 native profile iter=%d info_ms=%.3f "
               "asm_ms=%.3f scatter_ms=%.3f total_ms=%.3f\n",
               native_profile_iter,
               info_ms,
               asm_ms,
               scatter_ms,
               total_ms);
        hipEventDestroy(native_prof_start);
        hipEventDestroy(native_prof_after_info);
        hipEventDestroy(native_prof_after_asm);
        hipEventDestroy(native_prof_after_scatter);
    }

    static std::atomic<bool> logged{false};
    bool expected = false;
    if (rank == 0 && logged.compare_exchange_strong(expected, true))
    {
        printf("[hygon] ASM MT256 native contract enabled: problems=%d "
               "packed_rows=%lld packed_output_rows=%lld n=%d k=%d "
               "direct_input=%d direct_output=%d tile_wait_pack=%d vec_pack=%d "
               "vec_scatter=%d direct_output_nchunk_signal=%d wgm8=%d "
               "grouped_wgm=%d wide_grouped_wgm=%d v36_group_major=%d "
               "v36_expert_nmajor_wgm1=%d native_kn_weight=%d "
               "packed_input=%d d64_body=%d "
               "variant=%d\n",
               problem_count,
               static_cast<long long>(packed_rows),
               static_cast<long long>(packed_output_rows),
               n_dim,
               k_dim,
               direct_input ? 1 : 0,
               direct_output ? 1 : 0,
               tile_wait_pack ? 1 : 0,
               vec_pack_active ? 1 : 0,
               (!direct_output && vec_scatter && n_dim % 8 == 0) ? 1 : 0,
               (direct_output &&
                (ag_mt256_variant_nchunk_signal(variant) ||
                 (!explicit_variant && get_env_bool_local(
                      "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT_N_CHUNK_SIGNAL",
                      false))))
                   ? 1
                   : 0,
               wgm8 ? 1 : 0,
               grouped_wgm ? 1 : 0,
               wide_grouped_wgm ? 1 : 0,
               v36_group_major ? 1 : 0,
               v36_expert_nmajor_wgm1_direct ? 1 : 0,
               native_kn_weight ? 1 : 0,
               packed_input_enabled ? 1 : 0,
               native_kn_packed_d64_body ? 1 : 0,
               static_cast<int>(variant));
    }
    return true;
}

void ag_group_gemm_flat_input_forward_impl(
    const void *flat_input,
    const void *weight_nk,
    const void *weight_kn,
    const int *sorted_token_ids,
    const int *sorted_gather_index,
    const int *rank_tile_info,
    const int *ntiles,
    const int *rank_signals,
    int *n_chunk_done_counts,
    int *n_chunk_done_flags,
    int *v36_task_control,
    int v36_task_control_capacity,
    void *output,
    int full_m,
    int n_dim,
    int k_dim,
    int topk,
    int max_tiles,
    int signal_value,
    int chunk_size_n,
    int num_done_groups,
    c10::ScalarType dtype,
    hipStream_t stream,
    const void *router_scale)
{
    (void)max_tiles;
    FTRAIN_HYGON_CHECK(get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_NATIVE_CONTRACT", false))
        << "MoEReduceRS source MT256 route requires "
           "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_NATIVE_CONTRACT=1";
    const AgMt256AsmVariant requested_variant = get_ag_mt256_asm_variant();
    const bool layout_auto =
        requested_variant == AgMt256AsmVariant::V36LayoutAuto;
    // `weight_nk` names the legacy [E,N,K_local] contract here. In
    // layout-auto mode a null legacy pointer marks native [E,K_local,N].
    // Explicit selectors retain their old layout semantics.
    const bool input_weight_nk = layout_auto
        ? weight_nk != nullptr
        : ag_mt256_variant_native_kn(requested_variant);
    const AgMt256AsmVariant variant = ag_mt256_resolve_variant(
        requested_variant,
        (n_dim + 255) / 256,
        full_m,
        topk,
        n_dim,
        input_weight_nk);
    const bool explicit_variant = variant != AgMt256AsmVariant::Auto;
    const bool direct_input =
        explicit_variant
            ? ag_mt256_variant_direct_input(variant)
            : get_env_bool_local(
                  "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_INPUT", false);
    FTRAIN_HYGON_CHECK(direct_input)
        << "MoEReduceRS source MT256 route requires "
           "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_INPUT=1";
    const bool direct_output =
        explicit_variant
            ? ag_mt256_variant_direct_output(variant)
            : get_env_bool_local(
                  "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT", false);
    const bool direct_output_nchunk_signal =
        ag_mt256_variant_nchunk_signal(variant) ||
        (!explicit_variant && get_env_bool_local(
             "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT_N_CHUNK_SIGNAL",
             false));
    FTRAIN_HYGON_CHECK(!direct_output_nchunk_signal || direct_output)
        << "MT256 direct-output N-chunk signal requires "
           "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT=1";

    const bool native_kn_weight = ag_mt256_variant_native_kn(variant);
    const void *selected_weight = native_kn_weight ? weight_kn : weight_nk;
    FTRAIN_HYGON_CHECK(selected_weight != nullptr)
        << (native_kn_weight
                ? "v36 NN/Ailk requires a contiguous [E,K,N] weight"
                : "MT256 TN requires a contiguous [E,N,K] weight");

    const bool launched = ag_mt256_launch_native_contract(
        flat_input,
        selected_weight,
        weight_kn,
        sorted_token_ids,
        sorted_gather_index,
        rank_tile_info,
        nullptr,
        ntiles,
        rank_signals,
        router_scale,
        output,
        full_m,
        n_dim,
        k_dim,
        topk,
        1,
        0,
        !native_kn_weight,
        signal_value,
        n_chunk_done_counts,
        n_chunk_done_flags,
        chunk_size_n,
        num_done_groups,
        v36_task_control,
        v36_task_control_capacity,
        0,
        0,
        false,
        dtype,
        stream);
    FTRAIN_HYGON_CHECK(launched)
        << "failed to launch reconstructed AGGroupGemm MT256 contract; "
           "check the selected variant's shape limit and "
           "FTRAIN_HYGON_ASM_DIR_GFX936/FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_CO";
}

bool ag_group_gemm_mt256_uses_v36_wgm1_direct(int full_m,
                                               int n_dim,
                                               bool native_kn_weight)
{
    if (full_m <= 0 || n_dim <= 0)
    {
        return false;
    }
    const AgMt256AsmVariant variant = ag_mt256_resolve_variant(
        get_ag_mt256_asm_variant(),
        (n_dim + 255) / 256,
        full_m,
        6,
        n_dim,
        !native_kn_weight);
    return variant == AgMt256AsmVariant::
        V36DynamicExpertNMajorWgm1DirectOutputNChunkSignal;
}

bool ag_group_gemm_mt256_uses_native_kn_weight(int full_m, int n_dim)
{
    if (full_m <= 0 || n_dim <= 0)
    {
        return false;
    }
    // AGGroupGemm's public ABI is [E,K,N], so layout-auto is resolved with
    // legacy_weight_nk=false. Explicit TN selectors therefore retain the
    // cached KN->NK pack path, while NN/Ailk selectors consume the public
    // weight directly.
    const AgMt256AsmVariant variant = ag_mt256_resolve_variant(
        get_ag_mt256_asm_variant(),
        (n_dim + 255) / 256,
        full_m,
        6,
        n_dim,
        false);
    return ag_mt256_variant_native_kn(variant);
}

bool ag_group_gemm_mt256_uses_native_kn_chunk(int full_m, int n_dim)
{
    if (full_m <= 0 || n_dim <= 0)
    {
        return false;
    }
    return ag_mt256_variant_native_kn_chunk(get_ag_mt256_asm_variant());
}

void ag_group_gemm_v4_flat_input_forward_impl(
    const void *flat_input,
    const void *weight_nk,
    const void *weight_kn,
    const int *sorted_token_ids,
    const int *sorted_gather_index,
    const int *rank_tile_info,
    const int *ntiles,
    const int *rank_signals,
    int *n_chunk_done_counts,
    int *n_chunk_done_flags,
    void *output,
    int full_m,
    int n_dim,
    int k_dim,
    int topk,
    int max_tiles,
    int signal_value,
    int chunk_size_n,
    int num_done_groups,
    c10::ScalarType dtype,
    hipStream_t stream)
{
    ag_group_gemm_flat_input_forward_impl(flat_input,
                                          weight_nk,
                                          weight_kn,
                                          sorted_token_ids,
                                          sorted_gather_index,
                                          rank_tile_info,
                                          ntiles,
                                          rank_signals,
                                          n_chunk_done_counts,
                                          n_chunk_done_flags,
                                          nullptr,
                                          0,
                                          output,
                                          full_m,
                                          n_dim,
                                          k_dim,
                                          topk,
                                          max_tiles,
                                          signal_value,
                                          chunk_size_n,
                                          num_done_groups,
                                          dtype,
                                          stream,
                                          nullptr);
}

void ag_group_gemm_tiled_forward_impl(const void *full_input,
                                      const void *weight,
                                      const void *weight_kn,
                                      const int *sorted_token_ids,
                                      const int *sorted_gather_index,
                                      const int *rank_tile_info,
                                      const int *tile_experts,
                                      const int *tile_starts,
                                      const int *tile_rank_starts,
                                      const int *tile_rank_ends,
                                      const int *rank_tile_problem_info,
                                      const int *ntiles,
                                      const int *rank_signals,
                                      int *rank_tile_ready_flags,
                                      int *n_chunk_done_counts,
                                      int *n_chunk_done_flags,
                                      void *output,
                                      int full_m,
                                      int n_dim,
                                      int k_dim,
                                      int topk,
                                      int world_size,
                                      int rank,
                                      int max_tiles,
                                      bool weight_nk,
    int signal_value,
    int signal_chunk_rows,
    int chunks_per_rank,
    bool ragged_m,
    c10::ScalarType dtype,
    hipStream_t stream)
{
    if (ag_mt256_launch_native_contract(full_input,
                                        weight,
                                        weight_kn,
                                        sorted_token_ids,
                                        sorted_gather_index,
                                        rank_tile_info,
                                        rank_tile_problem_info,
                                        ntiles,
                                        rank_signals,
                                        nullptr,
                                        output,
                                        full_m,
                                        n_dim,
                                        k_dim,
                                        topk,
                                        world_size,
                                        rank,
                                        weight_nk,
                                        signal_value,
                                        n_chunk_done_counts,
                                        n_chunk_done_flags,
                                        get_group_n_chunk_size_local(),
                                        (n_dim +
                                         get_group_n_chunk_size_local() - 1) /
                                            get_group_n_chunk_size_local(),
                                        nullptr,
                                        0,
                                        signal_chunk_rows,
                                        chunks_per_rank,
                                        ragged_m,
                                        dtype,
                                        stream))
    {
        return;
    }

    constexpr int BN = 256;
    constexpr int SUB_BN = 128;
    constexpr int MMAC_BN = 128;
    const bool use_mmac_direct =
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_DIRECT_MMAC", true);
    const bool use_lds_mmac =
        (dtype == c10::ScalarType::Half ||
         dtype == c10::ScalarType::BFloat16) &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_LDS_MMAC", true);
    const bool use_smem_tile =
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_REBUILT_SMEM_TILE", true);
    const bool use_mmac_shmem_b =
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_MMAC_SHMEM_B", true);
    const int direct_bn =
        (use_mmac_direct && use_lds_mmac && dtype == c10::ScalarType::BFloat16)
            ? 256
            : MMAC_BN;
    const int n_tiles = use_mmac_direct
                            ? ((n_dim + direct_bn - 1) / direct_bn)
                            : (use_smem_tile ? ((n_dim + SUB_BN - 1) / SUB_BN)
                                             : ((n_dim + BN - 1) / BN));
    const dim3 grid(use_mmac_direct ? max_tiles
                                    : (use_smem_tile ? max_tiles * 2
                                                     : max_tiles),
                    n_tiles);

    if (dtype == c10::ScalarType::BFloat16)
    {
        if (use_mmac_direct && use_lds_mmac)
        {
            const bool use_source_dsread_n128 = get_env_bool_local(
                "FTRAIN_HYGON_AG_GROUP_GEMM_SOURCE_DSREAD_N128",
                true);
            const bool use_dsread_fragment_layout = get_env_bool_local(
                "FTRAIN_HYGON_AG_GROUP_GEMM_DSREAD_FRAGMENT_LAYOUT",
                use_source_dsread_n128);
            const bool use_dsread_swizzle =
                use_dsread_fragment_layout &&
                get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_DSREAD_SWIZZLE",
                                   false);
            const bool use_wg8_fragment =
                use_dsread_fragment_layout &&
                get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_WG8_FRAGMENT", false);
            const bool use_wg8_m32 =
                !use_dsread_fragment_layout &&
                get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_WG8_M32", false);
            const bool prefer_wg16_m32_n64 =
                !use_source_dsread_n128 && !use_dsread_fragment_layout &&
                !use_wg8_m32 &&
                n_dim >= 512 && k_dim >= 512;
            const bool use_wg16_m32_n64 =
                !use_dsread_fragment_layout && !use_wg8_m32 &&
                get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_WG16_M32_N64",
                                   prefer_wg16_m32_n64);
            const bool use_wg16_pair_b =
                use_wg16_m32_n64 &&
                get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_WG16_M32_N64_PAIRB",
                                   false);
            const bool prefer_tile_ready =
                use_source_dsread_n128 && world_size > 1 && n_dim > 256 &&
                get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_SOURCE_TILE_READY",
                                   false);
            const bool use_tile_ready =
                rank_signals != nullptr && rank_tile_ready_flags != nullptr &&
                get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_TILE_READY_FLAG",
                                   prefer_tile_ready);
            if (use_wg8_fragment)
            {
                if (use_tile_ready)
                {
                    if (use_dsread_swizzle)
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_fragment_kernel<
                                true,
                                true>),
                            grid,
                            dim3(512),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            rank_tile_ready_flags,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                    else
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_fragment_kernel<
                                true,
                                false>),
                            grid,
                            dim3(512),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            rank_tile_ready_flags,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                }
                else
                {
                    if (use_dsread_swizzle)
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_fragment_kernel<
                                false,
                                true>),
                            grid,
                            dim3(512),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            nullptr,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                    else
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_fragment_kernel<
                                false,
                                false>),
                            grid,
                            dim3(512),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            nullptr,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                }
                return;
            }
            if (use_wg8_m32)
            {
                if (use_tile_ready)
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_m32_kernel<
                            true>),
                        grid,
                        dim3(512),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        rank_tile_ready_flags,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
                else
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_m32_kernel<
                            false>),
                        grid,
                        dim3(512),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        nullptr,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
            }
            else if (use_wg16_m32_n64)
            {
                if (use_tile_ready && use_wg16_pair_b)
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg16_m32_n64_kernel<
                            true,
                            true>),
                        grid,
                        dim3(1024),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        rank_tile_ready_flags,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
                else if (use_tile_ready)
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg16_m32_n64_kernel<
                            true,
                            false>),
                        grid,
                        dim3(1024),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        rank_tile_ready_flags,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
                else if (use_wg16_pair_b)
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg16_m32_n64_kernel<
                            false,
                            true>),
                        grid,
                        dim3(1024),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        nullptr,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
                else
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg16_m32_n64_kernel<
                            false,
                            false>),
                        grid,
                        dim3(1024),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        nullptr,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
            }
            else if (use_tile_ready)
            {
                if (use_dsread_fragment_layout)
                {
                    if (use_dsread_swizzle)
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_kernel<
                                true,
                                true,
                                true>),
                            grid,
                            dim3(1024),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            rank_tile_ready_flags,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                    else
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_kernel<
                                true,
                                true>),
                            grid,
                            dim3(1024),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            rank_tile_ready_flags,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                }
                else
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_kernel<
                            true,
                            false>),
                        grid,
                        dim3(1024),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        rank_tile_ready_flags,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
            }
            else
            {
                if (use_dsread_fragment_layout)
                {
                    if (use_dsread_swizzle)
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_kernel<
                                false,
                                true,
                                true>),
                            grid,
                            dim3(1024),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            nullptr,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                    else
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_kernel<
                                false,
                                true>),
                            grid,
                            dim3(1024),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            nullptr,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                }
                else
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_kernel<
                            false,
                            false>),
                        grid,
                        dim3(1024),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        nullptr,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
            }
        }
        else if (use_mmac_direct)
        {
            hipLaunchKernelGGL(ag_group_gemm_tiled_128x128x32_mmac_bf16_kernel,
                               grid,
                               dim3(512),
                               0,
                               stream,
                               static_cast<const uint16_t *>(full_input),
                               static_cast<const uint16_t *>(weight),
                               sorted_token_ids,
                               tile_experts,
                               tile_starts,
                               tile_rank_starts,
                               tile_rank_ends,
                               ntiles,
                               rank_signals,
                               static_cast<uint16_t *>(output),
                               full_m,
                               n_dim,
                               k_dim,
                               topk,
                               world_size,
                               weight_nk,
                               signal_value);
        }
        else if (use_smem_tile)
        {
            hipLaunchKernelGGL((ag_group_gemm_tiled_128x256x32_smem64x128_kernel<
                                   __hip_bfloat16>),
                               grid,
                               dim3(256),
                               0,
                               stream,
                               static_cast<const __hip_bfloat16 *>(full_input),
                               static_cast<const __hip_bfloat16 *>(weight),
                               sorted_token_ids,
                               tile_experts,
                               tile_starts,
                               tile_rank_starts,
                               tile_rank_ends,
                               ntiles,
                               rank_signals,
                               static_cast<__hip_bfloat16 *>(output),
                               full_m,
                               n_dim,
                               k_dim,
                               topk,
                               world_size,
                               signal_value);
        }
        else
        {
            hipLaunchKernelGGL((ag_group_gemm_tiled_128x256x32_kernel<
                                   __hip_bfloat16>),
                               grid,
                               dim3(256),
                               0,
                               stream,
                               static_cast<const __hip_bfloat16 *>(full_input),
                               static_cast<const __hip_bfloat16 *>(weight),
                               sorted_token_ids,
                               tile_experts,
                               tile_starts,
                               tile_rank_starts,
                               tile_rank_ends,
                               ntiles,
                               rank_signals,
                               static_cast<__hip_bfloat16 *>(output),
                               full_m,
                               n_dim,
                               k_dim,
                               topk,
                               world_size,
                               signal_value);
        }
    }
    else if (dtype == c10::ScalarType::Half)
    {
        if (use_mmac_direct && use_lds_mmac)
        {
            hipLaunchKernelGGL(
                ag_group_gemm_tiled_128x128x16_lds_mmac_f16_kernel,
                grid,
                dim3(1024),
                0,
                stream,
                static_cast<const _Float16 *>(full_input),
                static_cast<const _Float16 *>(weight),
                sorted_token_ids,
                tile_experts,
                tile_starts,
                tile_rank_starts,
                tile_rank_ends,
                ntiles,
                rank_signals,
                static_cast<_Float16 *>(output),
                full_m,
                n_dim,
                k_dim,
                topk,
                world_size,
                signal_value);
        }
        else if (use_mmac_direct)
        {
            hipLaunchKernelGGL(ag_group_gemm_tiled_128x128x32_mmac_f16_kernel,
                               grid,
                               dim3(512),
                               0,
                               stream,
                               static_cast<const _Float16 *>(full_input),
                               static_cast<const _Float16 *>(weight),
                               sorted_token_ids,
                               tile_experts,
                               tile_starts,
                               tile_rank_starts,
                               tile_rank_ends,
                               ntiles,
                               rank_signals,
                               static_cast<_Float16 *>(output),
                               full_m,
                               n_dim,
                               k_dim,
                               topk,
                               world_size,
                               weight_nk,
                               signal_value,
                               use_mmac_shmem_b);
        }
        else if (use_smem_tile)
        {
            hipLaunchKernelGGL((ag_group_gemm_tiled_128x256x32_smem64x128_kernel<
                                   __half>),
                               grid,
                               dim3(256),
                               0,
                               stream,
                               static_cast<const __half *>(full_input),
                               static_cast<const __half *>(weight),
                               sorted_token_ids,
                               tile_experts,
                               tile_starts,
                               tile_rank_starts,
                               tile_rank_ends,
                               ntiles,
                               rank_signals,
                               static_cast<__half *>(output),
                               full_m,
                               n_dim,
                               k_dim,
                               topk,
                               world_size,
                               signal_value);
        }
        else
        {
            hipLaunchKernelGGL((ag_group_gemm_tiled_128x256x32_kernel<__half>),
                               grid,
                               dim3(256),
                               0,
                               stream,
                               static_cast<const __half *>(full_input),
                               static_cast<const __half *>(weight),
                               sorted_token_ids,
                               tile_experts,
                               tile_starts,
                               tile_rank_starts,
                               tile_rank_ends,
                               ntiles,
                               rank_signals,
                               static_cast<__half *>(output),
                               full_m,
                               n_dim,
                               k_dim,
                               topk,
                               world_size,
                               signal_value);
        }
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "AGGroupGemm tiled path supports FP16/BF16";
    }
}

void ag_group_gemm_pack_weight_impl(const void *weight,
                                    void *packed_weight,
                                    int num_experts,
                                    int k_dim,
                                    int n_dim,
                                    c10::ScalarType dtype,
                                    hipStream_t stream)
{
    const int threads = 256;
    const int64_t total =
        static_cast<int64_t>(num_experts) * k_dim * n_dim;
    const dim3 grid((total + threads - 1) / threads);
    if (dtype == c10::ScalarType::BFloat16)
    {
        hipLaunchKernelGGL((pack_weight_kn_to_nk_kernel<uint16_t>),
                           grid,
                           dim3(threads),
                           0,
                           stream,
                           static_cast<const uint16_t *>(weight),
                           static_cast<uint16_t *>(packed_weight),
                           num_experts,
                           k_dim,
                           n_dim);
    }
    else if (dtype == c10::ScalarType::Half)
    {
        hipLaunchKernelGGL((pack_weight_kn_to_nk_kernel<_Float16>),
                           grid,
                           dim3(threads),
                           0,
                           stream,
                           static_cast<const _Float16 *>(weight),
                           static_cast<_Float16 *>(packed_weight),
                           num_experts,
                           k_dim,
                           n_dim);
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "AGGroupGemm weight pack supports FP16/BF16";
    }
}

void ag_group_gemm_wait_rank_signals_impl(const int *rank_signals,
                                          int world_size,
                                          int signal_value,
                                          hipStream_t stream)
{
    hipLaunchKernelGGL(wait_rank_signals_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       rank_signals,
                       world_size,
                       signal_value);
}

void ag_group_gemm_publish_signal_release_impl(int *remote_signal,
                                               uint32_t signal_value,
                                               hipStream_t stream)
{
    hipLaunchKernelGGL(publish_signal_release_kernel,
                       dim3(1),
                       dim3(64),
                       0,
                       stream,
                       remote_signal,
                       signal_value);
}

void ag_group_gemm_system_fence_impl(hipStream_t stream)
{
    hipLaunchKernelGGL(system_fence_kernel,
                       dim3(1),
                       dim3(64),
                       0,
                       stream);
}

void ag_group_gemm_reference_forward_impl(const void *full_input,
                                          const void *weight,
                                          const void *topk_ids,
                                          void *output,
                                          int full_m,
                                          int n_dim,
                                          int k_dim,
                                          int num_experts,
                                          int topk,
                                          c10::ScalarType dtype,
                                          c10::ScalarType index_dtype,
                                          hipStream_t stream)
{
    if (dtype == c10::ScalarType::BFloat16)
    {
        launch_reference_forward<__hip_bfloat16>(full_input,
                                                 weight,
                                                 topk_ids,
                                                 output,
                                                 full_m,
                                                 n_dim,
                                                 k_dim,
                                                 num_experts,
                                                 topk,
                                                 index_dtype,
                                                 stream);
    }
    else if (dtype == c10::ScalarType::Half)
    {
        launch_reference_forward<__half>(full_input,
                                         weight,
                                         topk_ids,
                                         output,
                                         full_m,
                                         n_dim,
                                         k_dim,
                                         num_experts,
                                         topk,
                                         index_dtype,
                                         stream);
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "AGGroupGemm reference path supports FP16/BF16";
    }
}

} // namespace hygon_vendor_impl
} // namespace bytedance


namespace bytedance
{
namespace hygon_vendor
{

void ag_group_gemm_publish_signal_release_impl(int *remote_signal,
                                               uint32_t signal_value,
                                               hipStream_t stream)
{
    hygon_vendor_impl::ag_group_gemm_publish_signal_release_impl(
        remote_signal, signal_value, stream);
}

void ag_group_gemm_system_fence_impl(hipStream_t stream)
{
    hygon_vendor_impl::ag_group_gemm_system_fence_impl(stream);
}

void ag_group_gemm_prepare_metadata_flat_input_impl(
    const void *topk_ids,
    int *sorted_token_ids,
    int *sorted_gather_index,
    int *expert_rank_counts,
    int *expert_rank_counters,
    int *expert_rank_offsets,
    int *global_expert_offsets,
    int *rank_cursors,
    int *rank_tile_info,
    int *tile_experts,
    int *tile_starts,
    int *tile_rank_starts,
    int *tile_rank_ends,
    int *rank_tile_problem_info,
    int *ntiles,
    int full_m,
    int topk,
    int num_experts,
    int world_size,
    int rank,
    int sorted_capacity,
    int rank_tile_capacity,
    int signal_chunk_rows,
    int chunks_per_rank,
    bool strict_rank_tile,
    bool build_problem_info,
    c10::ScalarType index_dtype,
    hipStream_t stream)
{
    hygon_vendor_impl::ag_group_gemm_prepare_metadata_flat_input_impl(
        topk_ids,
        sorted_token_ids,
        sorted_gather_index,
        expert_rank_counts,
        expert_rank_counters,
        expert_rank_offsets,
        global_expert_offsets,
        rank_cursors,
        rank_tile_info,
        tile_experts,
        tile_starts,
        tile_rank_starts,
        tile_rank_ends,
        rank_tile_problem_info,
        ntiles,
        full_m,
        topk,
        num_experts,
        world_size,
        rank,
        sorted_capacity,
        rank_tile_capacity,
        signal_chunk_rows,
        chunks_per_rank,
        strict_rank_tile,
        build_problem_info,
        index_dtype,
        stream);
}

void ag_group_gemm_flat_input_forward_impl(
    const void *flat_input,
    const void *weight_nk,
    const void *weight_kn,
    const int *sorted_token_ids,
    const int *sorted_gather_index,
    const int *rank_tile_info,
    const int *ntiles,
    const int *rank_signals,
    int *n_chunk_done_counts,
    int *n_chunk_done_flags,
    int *v36_task_control,
    int v36_task_control_capacity,
    void *output,
    int full_m,
    int n_dim,
    int k_dim,
    int topk,
    int max_tiles,
    int signal_value,
    int chunk_size_n,
    int num_done_groups,
    c10::ScalarType dtype,
    hipStream_t stream,
    const void *router_scale)
{
    hygon_vendor_impl::ag_group_gemm_flat_input_forward_impl(
        flat_input,
        weight_nk,
        weight_kn,
        sorted_token_ids,
        sorted_gather_index,
        rank_tile_info,
        ntiles,
        rank_signals,
        n_chunk_done_counts,
        n_chunk_done_flags,
        v36_task_control,
        v36_task_control_capacity,
        output,
        full_m,
        n_dim,
        k_dim,
        topk,
        max_tiles,
        signal_value,
        chunk_size_n,
        num_done_groups,
        dtype,
        stream,
        router_scale);
}

bool ag_group_gemm_mt256_uses_v36_wgm1_direct(int full_m,
                                               int n_dim,
                                               bool native_kn_weight)
{
    return hygon_vendor_impl::ag_group_gemm_mt256_uses_v36_wgm1_direct(full_m,
                                                                 n_dim,
                                                                 native_kn_weight);
}

bool ag_group_gemm_mt256_uses_native_kn_weight(int full_m, int n_dim)
{
    return hygon_vendor_impl::ag_group_gemm_mt256_uses_native_kn_weight(full_m,
                                                                  n_dim);
}

bool ag_group_gemm_mt256_uses_native_kn_chunk(int full_m, int n_dim)
{
    return hygon_vendor_impl::ag_group_gemm_mt256_uses_native_kn_chunk(full_m,
                                                                 n_dim);
}

void ag_group_gemm_native_kn_tiled_forward_impl(
    const void *full_input,
    const void *weight_kn,
    const int *sorted_token_ids,
    const int *sorted_gather_index,
    const int *rank_tile_info,
    const int *tile_experts,
    const int *tile_starts,
    const int *tile_rank_starts,
    const int *tile_rank_ends,
    const int *rank_tile_problem_info,
    const int *ntiles,
    const int *rank_signals,
    int *rank_tile_ready_flags,
    int *n_chunk_done_counts,
    int *n_chunk_done_flags,
    void *output,
    int full_m,
    int n_dim,
    int k_dim,
    int topk,
    int world_size,
    int rank,
    int max_tiles,
    int signal_value,
    int signal_chunk_rows,
    int chunks_per_rank,
    bool ragged_m,
    c10::ScalarType dtype,
    hipStream_t stream)
{
    // Pass the public KN pointer through both slots so the native route and
    // its correctness fallback see the same storage. weight_nk=false is the
    // layout discriminator consumed by the source-only selector.
    hygon_vendor_impl::ag_group_gemm_tiled_forward_impl(
        full_input,
        weight_kn,
        weight_kn,
        sorted_token_ids,
        sorted_gather_index,
        rank_tile_info,
        tile_experts,
        tile_starts,
        tile_rank_starts,
        tile_rank_ends,
        rank_tile_problem_info,
        ntiles,
        rank_signals,
        rank_tile_ready_flags,
        n_chunk_done_counts,
        n_chunk_done_flags,
        output,
        full_m,
        n_dim,
        k_dim,
        topk,
        world_size,
        rank,
        max_tiles,
        false,
        signal_value,
        signal_chunk_rows,
        chunks_per_rank,
        ragged_m,
        dtype,
        stream);
}

void ag_group_gemm_legacy_tn_tiled_forward_impl(
    const void *full_input,
    const void *weight_nk,
    const int *sorted_token_ids,
    const int *sorted_gather_index,
    const int *rank_tile_info,
    const int *tile_experts,
    const int *tile_starts,
    const int *tile_rank_starts,
    const int *tile_rank_ends,
    const int *ntiles,
    const int *rank_signals,
    int *rank_tile_ready_flags,
    int *n_chunk_done_counts,
    int *n_chunk_done_flags,
    void *output,
    int full_m,
    int n_dim,
    int k_dim,
    int topk,
    int world_size,
    int rank,
    int max_tiles,
    int signal_value,
    int signal_chunk_rows,
    int chunks_per_rank,
    bool ragged_m,
    c10::ScalarType dtype,
    hipStream_t stream)
{
    // The caller owns the public KN->NK cache.  The reconstructed TN launcher
    // therefore receives only the selected [E,N,K] pointer and uses
    // weight_nk=true as its layout discriminator.
    hygon_vendor_impl::ag_group_gemm_tiled_forward_impl(
        full_input,
        weight_nk,
        nullptr,
        sorted_token_ids,
        sorted_gather_index,
        rank_tile_info,
        tile_experts,
        tile_starts,
        tile_rank_starts,
        tile_rank_ends,
        nullptr,
        ntiles,
        rank_signals,
        rank_tile_ready_flags,
        n_chunk_done_counts,
        n_chunk_done_flags,
        output,
        full_m,
        n_dim,
        k_dim,
        topk,
        world_size,
        rank,
        max_tiles,
        true,
        signal_value,
        signal_chunk_rows,
        chunks_per_rank,
        ragged_m,
        dtype,
        stream);
}

void ag_group_gemm_v4_flat_input_forward_impl(
    const void *flat_input,
    const void *weight_nk,
    const void *weight_kn,
    const int *sorted_token_ids,
    const int *sorted_gather_index,
    const int *rank_tile_info,
    const int *ntiles,
    const int *rank_signals,
    int *n_chunk_done_counts,
    int *n_chunk_done_flags,
    void *output,
    int full_m,
    int n_dim,
    int k_dim,
    int topk,
    int max_tiles,
    int signal_value,
    int chunk_size_n,
    int num_done_groups,
    c10::ScalarType dtype,
    hipStream_t stream)
{
    ag_group_gemm_flat_input_forward_impl(flat_input,
                                          weight_nk,
                                          weight_kn,
                                          sorted_token_ids,
                                          sorted_gather_index,
                                          rank_tile_info,
                                          ntiles,
                                          rank_signals,
                                          n_chunk_done_counts,
                                          n_chunk_done_flags,
                                          nullptr,
                                          0,
                                          output,
                                          full_m,
                                          n_dim,
                                          k_dim,
                                          topk,
                                          max_tiles,
                                          signal_value,
                                          chunk_size_n,
                                          num_done_groups,
                                          dtype,
                                          stream);
}

} // namespace hygon_vendor
} // namespace bytedance
