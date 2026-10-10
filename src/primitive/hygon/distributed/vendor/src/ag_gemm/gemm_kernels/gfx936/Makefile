# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Rebuild the gfx936 grouped-GEMM code objects from the vendored assembly.
#
# The assembly intentionally retains the historical flux_* ELF symbols.  Those
# symbols are the loader ABI; packaged asset filenames use the Hygon brand.

DTK_ROOT ?= /opt/dtk-25.04.2
CLANG ?= $(DTK_ROOT)/llvm/bin/clang
SFLAGS = -x assembler -target amdgcn-amd-amdhsa \
         -mcode-object-version=4 -mcpu=gfx936

SRC_DIR := grouped_kernels
BUILD_DIR := build
FLASH_TRAIN_ROOT ?= $(shell git rev-parse --show-toplevel 2>/dev/null \
                     || printf '%s' '$(CURDIR)/../../../../../../../../../..')
PACKAGE_DIR ?= $(FLASH_TRAIN_ROOT)/python/flash_train/torch/hygon_assets/gfx936

NATIVE_NK_A2A_SYMBOL := flux_ag_grouped_gemm_contract_rank_tile_x64_native_nk_a2a_direct_output_nchunk_signal
NATIVE_KN_A2A_SYMBOL := flux_ag_grouped_gemm_contract_rank_tile_x64_native_kn_a2a_nn_ailk_direct_output_nchunk_signal
NATIVE_KN_RING_FIXED_SYMBOL := flux_ag_grouped_gemm_contract_rank_tile_x64_native_kn_ring_fixed_routed_direct_output_nchunk_signal
NATIVE_KN_RING_READYMASK_SYMBOL := flux_ag_grouped_gemm_contract_rank_tile_x64_native_kn_ring_readymask_routed_direct_output_nchunk_signal

NATIVE_NK_A2A_ASSET := $(subst flux_,hygon_,$(NATIVE_NK_A2A_SYMBOL))
NATIVE_KN_A2A_ASSET := $(subst flux_,hygon_,$(NATIVE_KN_A2A_SYMBOL))
NATIVE_KN_RING_FIXED_ASSET := $(subst flux_,hygon_,$(NATIVE_KN_RING_FIXED_SYMBOL))
NATIVE_KN_RING_READYMASK_ASSET := $(subst flux_,hygon_,$(NATIVE_KN_RING_READYMASK_SYMBOL))

$(BUILD_DIR)/$(NATIVE_NK_A2A_ASSET).o: $(SRC_DIR)/$(NATIVE_NK_A2A_ASSET).s
	@mkdir -p $(BUILD_DIR)
	$(CLANG) $(SFLAGS) -c -o $@ $<
$(BUILD_DIR)/$(NATIVE_NK_A2A_ASSET).co: $(BUILD_DIR)/$(NATIVE_NK_A2A_ASSET).o
	$(CLANG) -target amdgcn-amd-amdhsa $< -o $@

$(BUILD_DIR)/$(NATIVE_KN_A2A_ASSET).o: $(SRC_DIR)/$(NATIVE_KN_A2A_ASSET).s
	@mkdir -p $(BUILD_DIR)
	$(CLANG) $(SFLAGS) -c -o $@ $<
$(BUILD_DIR)/$(NATIVE_KN_A2A_ASSET).co: $(BUILD_DIR)/$(NATIVE_KN_A2A_ASSET).o
	$(CLANG) -target amdgcn-amd-amdhsa $< -o $@

$(BUILD_DIR)/$(NATIVE_KN_RING_FIXED_ASSET).o: $(SRC_DIR)/$(NATIVE_KN_RING_FIXED_ASSET).s
	@mkdir -p $(BUILD_DIR)
	$(CLANG) $(SFLAGS) -c -o $@ $<
$(BUILD_DIR)/$(NATIVE_KN_RING_FIXED_ASSET).co: $(BUILD_DIR)/$(NATIVE_KN_RING_FIXED_ASSET).o
	$(CLANG) -target amdgcn-amd-amdhsa $< -o $@

$(BUILD_DIR)/$(NATIVE_KN_RING_READYMASK_ASSET).o: $(SRC_DIR)/$(NATIVE_KN_RING_READYMASK_ASSET).s
	@mkdir -p $(BUILD_DIR)
	$(CLANG) $(SFLAGS) -c -o $@ $<
$(BUILD_DIR)/$(NATIVE_KN_RING_READYMASK_ASSET).co: $(BUILD_DIR)/$(NATIVE_KN_RING_READYMASK_ASSET).o
	$(CLANG) -target amdgcn-amd-amdhsa $< -o $@

ALL_CO := $(BUILD_DIR)/$(NATIVE_NK_A2A_ASSET).co \
          $(BUILD_DIR)/$(NATIVE_KN_A2A_ASSET).co \
          $(BUILD_DIR)/$(NATIVE_KN_RING_FIXED_ASSET).co \
          $(BUILD_DIR)/$(NATIVE_KN_RING_READYMASK_ASSET).co

.PHONY: all check install clean
all: $(ALL_CO)

check: all
	@set -e; \
	cmp "$(BUILD_DIR)/$(NATIVE_NK_A2A_ASSET).co" "$(PACKAGE_DIR)/$(NATIVE_NK_A2A_ASSET).co"; \
	echo "verified $(NATIVE_NK_A2A_ASSET).co"; \
	cmp "$(BUILD_DIR)/$(NATIVE_KN_A2A_ASSET).co" "$(PACKAGE_DIR)/$(NATIVE_KN_A2A_ASSET).co"; \
	echo "verified $(NATIVE_KN_A2A_ASSET).co"; \
	cmp "$(BUILD_DIR)/$(NATIVE_KN_RING_FIXED_ASSET).co" "$(PACKAGE_DIR)/$(NATIVE_KN_RING_FIXED_ASSET).co"; \
	echo "verified $(NATIVE_KN_RING_FIXED_ASSET).co"; \
	cmp "$(BUILD_DIR)/$(NATIVE_KN_RING_READYMASK_ASSET).co" "$(PACKAGE_DIR)/$(NATIVE_KN_RING_READYMASK_ASSET).co"; \
	echo "verified $(NATIVE_KN_RING_READYMASK_ASSET).co"; \
	echo "verified 4 production code objects"

install: check
	@set -e; \
	cp "$(BUILD_DIR)/$(NATIVE_NK_A2A_ASSET).co" "$(PACKAGE_DIR)/$(NATIVE_NK_A2A_ASSET).co"; \
	cp "$(BUILD_DIR)/$(NATIVE_KN_A2A_ASSET).co" "$(PACKAGE_DIR)/$(NATIVE_KN_A2A_ASSET).co"; \
	cp "$(BUILD_DIR)/$(NATIVE_KN_RING_FIXED_ASSET).co" "$(PACKAGE_DIR)/$(NATIVE_KN_RING_FIXED_ASSET).co"; \
	cp "$(BUILD_DIR)/$(NATIVE_KN_RING_READYMASK_ASSET).co" "$(PACKAGE_DIR)/$(NATIVE_KN_RING_READYMASK_ASSET).co"; \
	echo "installed 4 production code objects"

clean:
	rm -rf $(BUILD_DIR)
