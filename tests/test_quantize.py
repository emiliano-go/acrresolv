"""Tests for quantize.py module."""

import pytest
import torch
import torch.nn as nn

from acrresolv.quantize import (
    fake_quant,
    fake_quant_act,
    quantize_params,
    QATHook,
    disable_quant,
    is_quant_enabled,
    get_model_size,
)


class TestFakeQuant:
    def test_basic_quantization(self):
        w = torch.randn(128, 128)
        q = fake_quant(w, bits=4, group_size=128)
        assert q.shape == w.shape
        assert q.dtype == w.dtype

    def test_preserves_shape(self):
        w = torch.randn(256, 64)
        q = fake_quant(w, bits=4, group_size=128)
        assert q.shape == w.shape

    def test_different_bits(self):
        w = torch.randn(128, 128)
        for bits in [2, 3, 4]:
            q = fake_quant(w, bits=bits, group_size=128)
            assert q.shape == w.shape

    def test_different_group_sizes(self):
        w = torch.randn(128, 128)
        for group_size in [32, 64, 128]:
            q = fake_quant(w, bits=4, group_size=group_size)
            assert q.shape == w.shape


class TestDisableQuant:
    def test_disable_quant(self):
        assert is_quant_enabled() == True
        with disable_quant():
            assert is_quant_enabled() == False
        assert is_quant_enabled() == True

    def test_nested_disable(self):
        assert is_quant_enabled() == True
        with disable_quant():
            assert is_quant_enabled() == False
            with disable_quant():
                assert is_quant_enabled() == False
            assert is_quant_enabled() == False
        assert is_quant_enabled() == True


class TestQATHook:
    def test_register_hook(self):
        model = nn.Linear(128, 64)
        hook = QATHook(model, bits=4, group_size=128)
        hook.register()
        assert len(hook._hooks) == 1
        hook.remove()
        assert len(hook._hooks) == 0

    def test_multiple_layers(self):
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        hook = QATHook(model, bits=4, group_size=128)
        hook.register()
        assert len(hook._hooks) == 2
        hook.remove()
        assert len(hook._hooks) == 0


class TestQuantizeParams:
    def test_quantize_linear(self):
        model = nn.Linear(128, 64)
        quantized = quantize_params(model, bits=4, group_size=128)
        assert quantized.weight.shape == model.weight.shape

    def test_quantize_sequential(self):
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        quantized = quantize_params(model, bits=4, group_size=128)
        assert len(list(quantized.children())) == 3


class TestGetModelSize:
    def test_linear_model(self):
        model = nn.Linear(128, 64)
        size_info = get_model_size(model)
        assert 'total_params' in size_info
        assert 'quantized_params' in size_info
        assert 'size_mb' in size_info
        assert size_info['total_params'] == 128 * 64 + 64

    def test_sequential_model(self):
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64, 32),
        )
        size_info = get_model_size(model)
        assert size_info['total_params'] == 128 * 64 + 64 + 64 * 32 + 32
