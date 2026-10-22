"""Arrays created by aljabr must live on the device of their inputs.

`array_api_strict` emulates several devices and refuses mixed-device
operations, which lets these tests run without a GPU.
"""

import array_api_compat as arr_api
import numpy as np
import pytest
from conftest import RNG

from aljabr.concrete import Conv, Diag, Diff, Identity, Slice
from aljabr.linop import Dense, asmatrix
from aljabr.utils import dottest, fwadjtest

xp = pytest.importorskip("array_api_strict")

DEVICE = xp.Device("device1")


def on_device(arr):
    return xp.asarray(arr, device=DEVICE)


def like():
    return xp.zeros(1, dtype=xp.float64, device=DEVICE)


class TestAdjointDevice:
    def test_slice(self):
        op = Slice((8, 8), np.index_exp[::2, 1])
        out = op.adjoint(on_device(RNG.standard_normal(op.oshape)))
        assert arr_api.device(out) == DEVICE

    @pytest.mark.xfail(strict=True, reason="udft.ir2fr ignores the ir device")
    def test_conv(self):
        op = Conv(on_device(np.ones((3, 3)) / 9), (8, 8), dim=2)
        out = op.adjoint(on_device(RNG.standard_normal(op.oshape)))
        assert arr_api.device(out) == DEVICE


class TestAsmatrixDevice:
    def test_linop(self):
        assert arr_api.device(Diff(0, (4,)).asmatrix(like=like())) == DEVICE

    def test_identity(self):
        assert arr_api.device(Identity((3,)).asmatrix(like=like())) == DEVICE

    def test_diag(self):
        diag = RNG.standard_normal(3)
        mat = Diag(diag).asmatrix(like=like())
        assert arr_api.device(mat) == DEVICE
        np.testing.assert_array_equal(
            np.asarray(xp.asarray(mat, device=xp.Device("CPU_DEVICE"))), np.diag(diag)
        )

    def test_dense(self):
        assert arr_api.device(Dense(np.ones((3, 4))).asmatrix(like=like())) == DEVICE

    def test_array(self):
        assert arr_api.device(asmatrix(np.eye(2), like=like())) == DEVICE


class TestUtilsDevice:
    def test_dottest(self):
        assert dottest(Diag(on_device(RNG.standard_normal(5))), xp=xp, device=DEVICE)

    def test_fwadjtest(self):
        assert fwadjtest(Diag(on_device(RNG.standard_normal(5))), xp=xp, device=DEVICE)
