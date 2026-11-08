# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flash_train.torch
import flash_train.torch as ft

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no visible HCU device"
)

SHAPE_GRID = [(3, 17, 2), (7, 33, 4), (33, 64, 8)]


@pytest.mark.parametrize(
    "columns,top_k",
    [(63, 63), (64, 64), (65, 64), (511, 129), (512, 129), (513, 129), (2048, 513), (8192, 2048)],
)
@pytest.mark.parametrize("ordered", [False, True])
def test_wave_boundaries_preserve_keys_indices_and_nan_payloads(columns, top_k, ordered):
    generator = torch.Generator(device="cuda").manual_seed(719)
    values = torch.stack(
        [torch.randperm(columns, generator=generator, device="cuda").float() for _ in range(5)]
    )
    # Include both NaN signs/payloads, infinities and signed zero. Sorting keys
    # must preserve the original value bits at each returned index.
    bits = [0x7FC01234, 0xFFC05678, 0x7F800000, 0xFF800000, 0, 0x80000000]
    signed_bits = [b if b < 2**31 else b - 2**32 for b in bits]
    values.view(torch.int32)[:, :6] = torch.tensor(signed_bits, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0, 1, columns // 2, columns - 1, columns], device="cuda", dtype=torch.int32)
    ids, selected = ft.indexer_topk(values, lengths, top_k, order_values=ordered)
    valid = torch.arange(top_k, device="cuda")[None, :] < lengths[:, None]
    torch.testing.assert_close(ids >= 0, valid)
    assert torch.all(ids[valid] < lengths[:, None].expand_as(ids)[valid])
    for row in range(5):
        chosen = ids[row][valid[row]]
        assert chosen.unique().numel() == chosen.numel()
    gathered = values.gather(1, ids.clamp_min(0).long()).masked_fill(~valid, -float("inf"))
    masked = values.masked_fill(
        torch.arange(columns, device="cuda")[None, :] >= lengths[:, None], -float("inf")
    )
    # The device radix-sort reference ranks negative NaNs below -infinity.
    # Use the CPU reference to rank both NaN signs above all finite values.
    expected = masked.cpu().topk(top_k, sorted=True).values
    actual = selected.cpu() if ordered else gathered.cpu().sort(descending=True).values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    if ordered:
        original_bits = values.view(torch.int32).gather(1, ids.clamp_min(0).long())
        torch.testing.assert_close(selected.view(torch.int32)[valid], original_bits[valid], rtol=0, atol=0)


@pytest.mark.parametrize("columns,top_k", [(17, 8), (1024, 8), (4096, 2048)])
@pytest.mark.parametrize("ordered", [False, True])
def test_nonfinite_selection_does_not_depend_on_order_values(columns, top_k, ordered):
    values = torch.arange(columns, device="cuda", dtype=torch.float32).unsqueeze(0)
    values[0, :3] = torch.tensor(
        [float("inf"), float("nan"), -float("inf")], device="cuda"
    )
    lengths = torch.tensor([columns], device="cuda", dtype=torch.int32)
    ids, selected = ft.indexer_topk(values, lengths, top_k, order_values=ordered)
    expected_values, expected_ids = values.topk(top_k)
    assert (ids >= 0).sum().item() == top_k
    torch.testing.assert_close(ids.sort().values, expected_ids.int().sort().values)
    if ordered:
        torch.testing.assert_close(selected, expected_values, equal_nan=True)


@pytest.mark.parametrize("shape", SHAPE_GRID)
def test_matches_torch(shape):
    rows, columns, top_k = shape
    generator = torch.Generator(device="cuda").manual_seed(7)
    # Distinct values avoid imposing a tie-breaking rule on equal elements.
    values = torch.stack(
        [
            torch.randperm(columns, device="cuda", generator=generator).float()
            for _ in range(rows)
        ]
    )
    lengths = torch.full((rows,), columns, device="cuda", dtype=torch.int32)
    expected_values, expected_indices = values.topk(top_k, sorted=True)
    actual_indices, actual_values = flash_train.torch.indexer_topk(
        values, lengths, top_k, next_n=1, order_values=True
    )
    expected_indices = expected_indices.to(torch.int32)
    torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)
