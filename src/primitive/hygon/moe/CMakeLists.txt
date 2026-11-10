set(ftrain_moe_sources
    assets.cpp
    runtime.cpp
    ag_group_gemm.cpp
    moe_reduce_rs.cpp
    kernels/cudaipc_barrier_all.hip
    kernels/ag_copy_remote.hip
    kernels/ag_group_gemm_ck.hip
    kernels/ag_group_gemm_legacy_compat_ck.hip
    kernels/moe_reduce_rs_kernels.hip
    kernels/moe_fc2_gemm.hip)
list(TRANSFORM ftrain_moe_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
target_sources(flash_train_impl PRIVATE ${ftrain_moe_sources})
target_include_directories(flash_train_impl PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
set_source_files_properties(
  ${ftrain_moe_sources}
  TARGET_DIRECTORY flash_train_impl
  PROPERTIES LANGUAGE HIP
             COMPILE_OPTIONS "-mno-fma;-mllvm;-enable-num-vgprs-512=true;-ffile-prefix-map=${PROJECT_SOURCE_DIR}=."
             COMPILE_DEFINITIONS
             "CUDA_HAS_FP16=1;USE_ROCM;BF162FLOAT_USE_ASM;BWDSEPARATE;HIPBLAS_V2;HIPBLASLT_OUTER_VEC;__HIP_NO_HALF_OPERATORS__=1;__HIP_NO_HALF_CONVERSIONS__=1;DCU_ASM;HIP_ENABLE_WARP_SYNC_BUILTINS")

set(ftrain_moe_assembly_directory "${CMAKE_CURRENT_BINARY_DIR}/code_objects/gfx936")
file(MAKE_DIRECTORY "${ftrain_moe_assembly_directory}")
file(GLOB ftrain_moe_assembly_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/asm/gfx936/*.s")

foreach(route_suffix IN ITEMS "" "_32k")
  set(source_symbol "ftrain_ag_grouped_gemm_contract_rank_tile_x64_ag_nt_a2a_chunk${route_suffix}_direct_output_nchunk_signal")
  set(destination_symbol "ftrain_ag_grouped_gemm_contract_rank_tile_x64_ag_nt_ring_chunk${route_suffix}_direct_output_nchunk_signal")
  set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/asm/gfx936/${source_symbol}.s")
  set(destination_file "${CMAKE_CURRENT_BINARY_DIR}/${destination_symbol}.s")
  add_custom_command(
    OUTPUT "${destination_file}"
    COMMAND ${CMAKE_COMMAND} "-DINPUT=${source_file}" "-DOUTPUT=${destination_file}"
            "-DSOURCE_SYMBOL=${source_symbol}" "-DDESTINATION_SYMBOL=${destination_symbol}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MoeAssemblyAlias.cmake"
    DEPENDS "${source_file}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MoeAssemblyAlias.cmake"
    VERBATIM)
  list(APPEND ftrain_moe_assembly_sources "${destination_file}")
endforeach()

set(ftrain_moe_code_objects)
foreach(assembly_source IN LISTS ftrain_moe_assembly_sources)
  get_filename_component(kernel_name "${assembly_source}" NAME_WE)
  set(assembly_object "${CMAKE_CURRENT_BINARY_DIR}/${kernel_name}.o")
  set(code_object "${ftrain_moe_assembly_directory}/${kernel_name}.co")
  add_custom_command(
    OUTPUT "${code_object}"
    BYPRODUCTS "${assembly_object}"
    COMMAND "${CMAKE_HIP_COMPILER}" -x assembler -target amdgcn-amd-amdhsa
            -mcode-object-version=4 -mcpu=gfx936 -c "${assembly_source}" -o "${assembly_object}"
    COMMAND "${CMAKE_HIP_COMPILER}" -target amdgcn-amd-amdhsa "${assembly_object}" -o "${code_object}"
    DEPENDS "${assembly_source}"
    VERBATIM)
  list(APPEND ftrain_moe_code_objects "${code_object}")
endforeach()
add_custom_target(ftrain_moe_assembly DEPENDS ${ftrain_moe_code_objects})
add_dependencies(flash_train_impl ftrain_moe_assembly)

set(ftrain_moe_embedded_source "${CMAKE_CURRENT_BINARY_DIR}/embedded_code_objects.cpp")
add_custom_command(
  OUTPUT "${ftrain_moe_embedded_source}"
  COMMAND ${CMAKE_COMMAND} "-DINPUTS=${ftrain_moe_code_objects}" "-DOUTPUT=${ftrain_moe_embedded_source}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MoeEmbedCodeObjects.cmake"
  DEPENDS ${ftrain_moe_code_objects} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MoeEmbedCodeObjects.cmake"
  VERBATIM)
add_custom_target(ftrain_moe_embedded DEPENDS "${ftrain_moe_embedded_source}")
add_dependencies(flash_train_impl ftrain_moe_embedded)
target_sources(flash_train_impl PRIVATE "${ftrain_moe_embedded_source}")
