"""NNModelManager + 归一化函数 单元测试。

覆盖：
1. 归一化函数（_normalize_symptoms / _normalize_dx / _normalize_setup / _encode_track）
2. build_input_vector 维度与内容
3. _denormalize_delta 反归一化
4. NNModelManager 降级行为（PyTorch 不可用时 available=False, predict 返回 None）
5. 单例管理（get_nn_manager / reset_nn_manager）
6. PyTorch 可用时的模型构建/前向传播/权重保存加载（skipif 不可用）
7. is_torch_available 探测
8. 边界条件：未知症状、未知赛道、空输入
"""

from __future__ import annotations

from pathlib import Path

import pytest

from setup_tuner.domain.setup import ALL_SETUP_FIELDS, CarSetup
from setup_tuner.domain.symptoms import Symptom
from setup_tuner.domain.track import ALL_TRACKS
from setup_tuner.engine import nn_model
from setup_tuner.engine.nn_model import (
    F1SetupNet,
    NNModelManager,
    _denormalize_delta,
    _encode_track,
    _normalize_dx,
    _normalize_setup,
    _normalize_symptoms,
    build_input_vector,
    get_nn_manager,
    is_torch_available,
    reset_nn_manager,
)

_TORCH_AVAILABLE = is_torch_available()


# ===========================================================================
# 1. _normalize_symptoms
# ===========================================================================
class TestNormalizeSymptoms:
    """症状强度归一化为 12 维向量。"""

    def test_empty_symptoms(self) -> None:
        """空症状列表返回全零 12 维向量。"""
        vec = _normalize_symptoms([])
        assert len(vec) == nn_model._NUM_SYMPTOMS
        assert all(v == 0.0 for v in vec)

    def test_known_symptom(self) -> None:
        """已知症状归一化到 [0, 1]。"""
        # 用前 12 个症状（索引 0-11，避免越界）
        symptom_keys = [s.value for s in Symptom][:12]
        vec = _normalize_symptoms([(symptom_keys[0], 5)])
        assert vec[0] == pytest.approx(1.0)  # 5/5
        # 其余为 0
        assert all(vec[i] == 0.0 for i in range(1, 12))

    def test_intensity_normalization(self) -> None:
        """强度 0-5 归一化到 0-1。"""
        symptom_keys = [s.value for s in Symptom][:12]
        vec = _normalize_symptoms([(symptom_keys[0], 3)])
        assert vec[0] == pytest.approx(0.6)  # 3/5

    def test_unknown_symptom_ignored(self) -> None:
        """未知症状 key 被忽略。"""
        vec = _normalize_symptoms([("nonexistent", 5)])
        assert all(v == 0.0 for v in vec)

    def test_multiple_symptoms(self) -> None:
        """多症状叠加到各自位置。"""
        symptom_keys = [s.value for s in Symptom][:12]
        vec = _normalize_symptoms([
            (symptom_keys[0], 5), (symptom_keys[1], 2), (symptom_keys[5], 4),
        ])
        assert vec[0] == pytest.approx(1.0)
        assert vec[1] == pytest.approx(0.4)
        assert vec[5] == pytest.approx(0.8)


# ===========================================================================
# 2. _normalize_dx
# ===========================================================================
class TestNormalizeDx:
    """Dx 诊断向量归一化为 9 维向量。"""

    def test_empty_dx(self) -> None:
        """空 Dx 返回全零（tanh(0)=0）。"""
        vec = _normalize_dx({})
        assert len(vec) == nn_model._NUM_DIAG_DIMS
        assert all(v == 0.0 for v in vec)

    def test_dx_within_range(self) -> None:
        """Dx 归一化到 [-1, 1]（tanh 压缩）。"""
        vec = _normalize_dx({"front_grip_req": 10.0, "rear_grip_req": -10.0})
        assert len(vec) == 9
        assert -1.0 < vec[0] < 1.0
        assert -1.0 < vec[1] < 1.0
        assert vec[0] > 0  # 正值
        assert vec[1] < 0  # 负值

    def test_dx_unknown_key_ignored(self) -> None:
        """未知 Dx key 被忽略（对应位置为 0）。"""
        vec = _normalize_dx({"unknown_dim": 100.0})
        assert all(v == 0.0 for v in vec)

    def test_dx_large_value_saturates(self) -> None:
        """大值被 tanh 压缩到接近 ±1。"""
        vec = _normalize_dx({"front_grip_req": 1000.0})
        assert vec[0] > 0.99  # tanh(200) ≈ 1


# ===========================================================================
# 3. _normalize_setup
# ===========================================================================
class TestNormalizeSetup:
    """调教参数归一化为 20 维向量。"""

    def test_default_setup(self) -> None:
        """缺省调教归一化后每维在 [0, 1]。"""
        setup = CarSetup.default().to_dict()
        vec = _normalize_setup(setup)
        assert len(vec) == len(ALL_SETUP_FIELDS)
        assert all(0.0 <= v <= 1.0 for v in vec)

    def test_empty_setup_uses_defaults(self) -> None:
        """空 dict 用 spec.default 填充。"""
        vec = _normalize_setup({})
        assert len(vec) == len(ALL_SETUP_FIELDS)
        assert all(0.0 <= v <= 1.0 for v in vec)

    def test_min_max_values(self) -> None:
        """最小值归一化为 0，最大值归一化为 1。"""
        spec = ALL_SETUP_FIELDS[0]
        vec_min = _normalize_setup({spec.name: spec.min_val})
        assert vec_min[0] == pytest.approx(0.0)
        vec_max = _normalize_setup({spec.name: spec.max_val})
        assert vec_max[0] == pytest.approx(1.0)

    def test_out_of_range_clamped(self) -> None:
        """越界值被裁剪到 [0, 1]。"""
        spec = ALL_SETUP_FIELDS[0]
        vec = _normalize_setup({spec.name: spec.max_val * 10})
        assert vec[0] == 1.0  # 裁剪到上限
        vec2 = _normalize_setup({spec.name: spec.min_val - 100})
        assert vec2[0] == 0.0  # 裁剪到下限


# ===========================================================================
# 4. _encode_track
# ===========================================================================
class TestEncodeTrack:
    """赛道 one-hot 编码为 24 维向量。"""

    def test_known_track(self) -> None:
        """已知赛道返回 one-hot 向量。"""
        vec = _encode_track("suzuka")
        assert len(vec) == nn_model._NUM_TRACKS
        assert sum(vec) == 1.0  # 只有一个 1
        assert all(v in (0.0, 1.0) for v in vec)

    def test_unknown_track_returns_zeros(self) -> None:
        """未知赛道返回全零向量。"""
        vec = _encode_track("nonexistent_track")
        assert len(vec) == nn_model._NUM_TRACKS
        assert all(v == 0.0 for v in vec)

    def test_different_tracks_different_positions(self) -> None:
        """不同赛道的 one-hot 位置不同。"""
        v1 = _encode_track("suzuka")
        v2 = _encode_track("monaco")
        assert v1 != v2

    def test_all_tracks_encodable(self) -> None:
        """24 条赛道都能编码。"""
        for track in ALL_TRACKS:
            vec = _encode_track(track.track_id)
            assert sum(vec) == 1.0


# ===========================================================================
# 5. build_input_vector
# ===========================================================================
class TestBuildInputVector:
    """build_input_vector 构建完整输入特征向量。"""

    @staticmethod
    def _sample_input() -> tuple:
        """构造样本输入。"""
        symptom_keys = [s.value for s in Symptom][:12]
        symptoms = [(symptom_keys[0], 3), (symptom_keys[5], 4)]
        dx = {"front_grip_req": 5.0, "rear_grip_req": -3.0}
        setup = CarSetup.default().to_dict()
        track_id = "suzuka"
        return symptoms, dx, setup, track_id

    def test_input_vector_length(self) -> None:
        """输入向量长度 = 12 + 9 + 20 + 24 = 65。"""
        symptoms, dx, setup, track_id = self._sample_input()
        vec = build_input_vector(symptoms, dx, setup, track_id)
        # 实际长度：12 症状 + 9 Dx + 20 参数 + 24 赛道 = 65
        assert len(vec) == 12 + 9 + len(ALL_SETUP_FIELDS) + 24

    def test_input_vector_components(self) -> None:
        """输入向量含症状、Dx、调教、赛道四段。"""
        symptoms, dx, setup, track_id = self._sample_input()
        vec = build_input_vector(symptoms, dx, setup, track_id)
        # 症状段（0-11）
        assert vec[0] == pytest.approx(0.6)  # 3/5
        # Dx 段（12-20）
        assert vec[12] != 0.0  # front_grip_req
        # 赛道段（最后 24 维）含一个 1.0
        track_part = vec[-24:]
        assert sum(track_part) == 1.0

    def test_input_vector_empty_inputs(self) -> None:
        """空输入返回合法向量（全默认值）。"""
        vec = build_input_vector([], {}, {}, "suzuka")
        assert len(vec) == 12 + 9 + len(ALL_SETUP_FIELDS) + 24


# ===========================================================================
# 6. _denormalize_delta
# ===========================================================================
class TestDenormalizeDelta:
    """网络输出反归一化为参数 delta。"""

    def test_zero_delta(self) -> None:
        """全零输出反归一化为全零 delta。"""
        vec = [0.0] * len(ALL_SETUP_FIELDS)
        delta = _denormalize_delta(vec)
        assert len(delta) == len(ALL_SETUP_FIELDS)
        assert all(v == 0.0 for v in delta.values())

    def test_max_delta(self) -> None:
        """输出 1.0 反归一化为 +max_delta。"""
        spec = ALL_SETUP_FIELDS[0]
        vec = [0.0] * len(ALL_SETUP_FIELDS)
        vec[0] = 1.0
        delta = _denormalize_delta(vec)
        assert delta[spec.name] == pytest.approx(spec.max_delta)

    def test_neg_max_delta(self) -> None:
        """输出 -1.0 反归一化为 -max_delta。"""
        spec = ALL_SETUP_FIELDS[0]
        vec = [0.0] * len(ALL_SETUP_FIELDS)
        vec[0] = -1.0
        delta = _denormalize_delta(vec)
        assert delta[spec.name] == pytest.approx(-spec.max_delta)

    def test_all_fields_covered(self) -> None:
        """反归一化覆盖全部 20 参数。"""
        vec = [0.5] * len(ALL_SETUP_FIELDS)
        delta = _denormalize_delta(vec)
        for spec in ALL_SETUP_FIELDS:
            assert spec.name in delta


# ===========================================================================
# 7. NNModelManager 降级行为
# ===========================================================================
class TestNNModelManagerDegradation:
    """PyTorch 不可用时的降级行为；可用时验证基本构造。"""

    def test_is_torch_available_returns_bool(self) -> None:
        """is_torch_available 返回布尔值。"""
        assert isinstance(is_torch_available(), bool)

    @pytest.mark.skipif(_TORCH_AVAILABLE, reason="PyTorch 可用，跳过降级测试")
    def test_manager_unavailable_when_no_torch(self, tmp_path: Path) -> None:
        """PyTorch 不可用时 NNModelManager.available=False。"""
        mgr = NNModelManager(weights_path=tmp_path / "w.pt")
        assert mgr.available is False
        assert mgr.model is None

    @pytest.mark.skipif(_TORCH_AVAILABLE, reason="PyTorch 可用，跳过降级测试")
    def test_predict_returns_none_when_unavailable(self, tmp_path: Path) -> None:
        """不可用时 predict 返回 None。"""
        mgr = NNModelManager(weights_path=tmp_path / "w.pt")
        result = mgr.predict([], {}, {}, "suzuka")
        assert result is None

    @pytest.mark.skipif(_TORCH_AVAILABLE, reason="PyTorch 可用，跳过降级测试")
    def test_save_weights_returns_false_when_unavailable(self, tmp_path: Path) -> None:
        """不可用时 save_weights 返回 False。"""
        mgr = NNModelManager(weights_path=tmp_path / "w.pt")
        assert mgr.save_weights() is False

    @pytest.mark.skipif(_TORCH_AVAILABLE, reason="PyTorch 可用，跳过降级测试")
    def test_f1setupnet_is_dummy_when_no_torch(self) -> None:
        """PyTorch 不可用时 F1SetupNet 是占位类。"""
        net = F1SetupNet()
        # 占位类方法存在且不报错
        net.eval()
        result = net.forward("x")
        assert result == "x"

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch 不可用")
    def test_manager_available_with_weights(self, tmp_path: Path) -> None:
        """PyTorch 可用且权重存在时 available=True（先保存再加载）。"""
        import torch  # type: ignore[import-not-found]

        wpath = tmp_path / "w.pt"
        # 先用随机初始化模型保存权重
        mgr1 = NNModelManager(weights_path=wpath)
        # 权重文件不存在时 available=False，但模型可构造
        if mgr1.available:
            assert mgr1.save_weights(wpath) is True
        else:
            # 手动保存随机权重
            net = F1SetupNet()
            net.eval()
            torch.save(net.state_dict(), str(wpath))
        # 重新加载
        mgr2 = NNModelManager(weights_path=wpath)
        assert mgr2.available is True
        assert mgr2.model is not None


# ===========================================================================
# 8. NNModelManager 推理（PyTorch 可用时）
# ===========================================================================
class TestNNModelManagerInference:
    """PyTorch 可用时的推理测试。"""

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch 不可用")
    def test_predict_returns_delta_dict(self, tmp_path: Path) -> None:
        """predict 返回含全部 20 参数的 delta 字典。"""
        import torch  # type: ignore[import-not-found]

        wpath = tmp_path / "w.pt"
        net = F1SetupNet()
        net.eval()
        torch.save(net.state_dict(), str(wpath))
        mgr = NNModelManager(weights_path=wpath)
        symptom_keys = [s.value for s in Symptom][:12]
        delta = mgr.predict(
            [(symptom_keys[0], 3)], {"front_grip_req": 5.0},
            CarSetup.default().to_dict(), "suzuka",
        )
        assert delta is not None
        assert len(delta) == len(ALL_SETUP_FIELDS)
        for spec in ALL_SETUP_FIELDS:
            assert spec.name in delta
            assert -spec.max_delta <= delta[spec.name] <= spec.max_delta

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch 不可用")
    def test_save_and_load_weights_consistency(self, tmp_path: Path) -> None:
        """保存后重新加载，推理结果一致。"""
        import torch  # type: ignore[import-not-found]

        wpath = tmp_path / "w.pt"
        net = F1SetupNet()
        net.eval()
        torch.save(net.state_dict(), str(wpath))
        mgr1 = NNModelManager(weights_path=wpath)
        delta1 = mgr1.predict(
            [], {}, CarSetup.default().to_dict(), "monaco",
        )
        # 保存到新路径再加载
        wpath2 = tmp_path / "w2.pt"
        assert mgr1.save_weights(wpath2) is True
        mgr2 = NNModelManager(weights_path=wpath2)
        delta2 = mgr2.predict(
            [], {}, CarSetup.default().to_dict(), "monaco",
        )
        assert delta1 is not None and delta2 is not None
        for k in delta1:
            assert delta1[k] == pytest.approx(delta2[k], rel=1e-5)

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch 不可用")
    def test_f1setupnet_forward_shape(self) -> None:
        """F1SetupNet 前向传播：输入 68 维 → 输出 23 维。"""
        import torch  # type: ignore[import-not-found]

        net = F1SetupNet()
        net.eval()
        x = torch.zeros(1, nn_model._INPUT_SIZE)
        with torch.no_grad():
            y = net(x)
        assert y.shape[0] == 1
        assert y.shape[1] == nn_model._OUTPUT_SIZE
        # tanh 输出在 [-1, 1]
        assert (y >= -1.0).all() and (y <= 1.0).all()


# ===========================================================================
# 9. 单例管理
# ===========================================================================
class TestSingletonManager:
    """get_nn_manager / reset_nn_manager 单例。"""

    def test_get_nn_manager_returns_instance(self, tmp_path: Path) -> None:
        """get_nn_manager 返回 NNModelManager 实例。"""
        reset_nn_manager()
        mgr = get_nn_manager(weights_path=tmp_path / "w.pt")
        assert isinstance(mgr, NNModelManager)
        reset_nn_manager()

    def test_get_nn_manager_singleton(self, tmp_path: Path) -> None:
        """多次调用返回同一实例。"""
        reset_nn_manager()
        mgr1 = get_nn_manager(weights_path=tmp_path / "w1.pt")
        mgr2 = get_nn_manager(weights_path=tmp_path / "w2.pt")
        assert mgr1 is mgr2  # 同一实例
        reset_nn_manager()

    def test_reset_nn_manager(self, tmp_path: Path) -> None:
        """reset 后再次 get 返回新实例。"""
        reset_nn_manager()
        mgr1 = get_nn_manager(weights_path=tmp_path / "w.pt")
        reset_nn_manager()
        mgr2 = get_nn_manager(weights_path=tmp_path / "w.pt")
        assert mgr1 is not mgr2
        reset_nn_manager()


# ===========================================================================
# 10. 模块常量
# ===========================================================================
class TestModuleConstants:
    """模块级常量正确性。"""

    def test_input_output_sizes(self) -> None:
        """输入 68 维，输出 23 维。"""
        assert nn_model._INPUT_SIZE == 68
        assert nn_model._OUTPUT_SIZE == 23

    def test_component_sizes(self) -> None:
        """分量维度：12 + 9 + 23 + 24 = 68。"""
        assert nn_model._NUM_SYMPTOMS == 12
        assert nn_model._NUM_DIAG_DIMS == 9
        assert nn_model._NUM_SETUP_PARAMS == 23
        assert nn_model._NUM_TRACKS == 24

    def test_normalization_constants(self) -> None:
        """归一化常量。"""
        assert nn_model._SYMPTOM_INTENSITY_MAX == 5.0
        assert nn_model._DX_NORMALIZE_SCALE == 5.0

    def test_track_index_built(self) -> None:
        """赛道索引映射含 24 条赛道。"""
        idx = nn_model._build_track_index()
        assert len(idx) == 24
        assert "suzuka" in idx
        assert "monaco" in idx


# ===========================================================================
# 11. Mock PyTorch 路径覆盖（PyTorch 不可用时用 mock 覆盖降级分支）
# ===========================================================================
class _MockSqueezed:
    """模拟 tensor.squeeze().tolist() 链。"""

    def tolist(self) -> list[float]:
        """返回 20 维零 delta（反归一化后仍为 0）。"""
        return [0.0] * len(ALL_SETUP_FIELDS)


class _MockOutput:
    """模拟 model(x) 输出 tensor。"""

    def squeeze(self) -> _MockSqueezed:
        """模拟 tensor.squeeze()。"""
        return _MockSqueezed()


class _MockNet:
    """模拟 F1SetupNet（PyTorch 可用时的行为）。"""

    def eval(self) -> _MockNet:
        """切换到 eval 模式。"""
        return self

    def forward(self, x: object) -> _MockOutput:
        """前向传播返回 mock 输出。"""
        return _MockOutput()

    def load_state_dict(self, sd: object) -> None:
        """加载权重（no-op）。"""

    def state_dict(self) -> dict:
        """返回 mock state_dict。"""
        return {"mock": True}

    def __call__(self, x: object) -> _MockOutput:
        """可调用（与 forward 一致）。"""
        return self.forward(x)


class _MockNoGrad:
    """模拟 torch.no_grad() 上下文管理器。"""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class _MockTorch:
    """模拟 torch 模块（仅覆盖 NNModelManager 用到的 API）。"""

    float32 = "float32"

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> dict:
        """模拟 torch.load 返回 state_dict。"""
        return {"mock": True}

    @staticmethod
    def no_grad() -> _MockNoGrad:
        """模拟 torch.no_grad 上下文管理器。"""
        return _MockNoGrad()

    @staticmethod
    def tensor(data: list, dtype: object = None) -> object:
        """模拟 torch.tensor。"""
        return data

    @staticmethod
    def save(obj: object, path: str) -> None:
        """模拟 torch.save（写入空文件以使 exists() 为 True）。"""
        Path(path).write_bytes(b"")


@pytest.mark.skipif(_TORCH_AVAILABLE, reason="PyTorch 可用，跳过 mock 测试")
class TestMockPyTorchPaths:
    """用 mock 模拟 PyTorch 可用，覆盖 NNModelManager 的加载/推理/保存路径。

    这些测试在 PyTorch 不可用时运行，用 mock 覆盖源码中
    ``if _TORCH_AVAILABLE:`` 分支的代码路径，提升覆盖率。
    """

    @staticmethod
    def _patch_torch():
        """返回 mock patch 上下文（patch _TORCH_AVAILABLE/torch/F1SetupNet）。"""
        from unittest.mock import patch

        return (
            patch.object(nn_model, "_TORCH_AVAILABLE", True),
            patch.object(nn_model, "torch", _MockTorch),
            patch.object(nn_model, "F1SetupNet", _MockNet),
        )

    def test_manager_available_with_mock_weights(self, tmp_path: Path) -> None:
        """mock PyTorch + 权重文件存在 → available=True。"""
        wpath = tmp_path / "w.pt"
        wpath.write_bytes(b"mock weights")
        p1, p2, p3 = self._patch_torch()
        with p1, p2, p3:
            mgr = NNModelManager(weights_path=wpath)
            assert mgr.available is True
            assert mgr.model is not None

    def test_manager_unavailable_without_weights(self, tmp_path: Path) -> None:
        """mock PyTorch + 权重文件不存在 → available=False, model=None。"""
        wpath = tmp_path / "nonexistent.pt"
        p1, p2, p3 = self._patch_torch()
        with p1, p2, p3:
            mgr = NNModelManager(weights_path=wpath)
            assert mgr.available is False
            assert mgr.model is None

    def test_manager_load_exception_degrades(self, tmp_path: Path) -> None:
        """mock PyTorch + 加载异常 → 降级 available=False。"""
        wpath = tmp_path / "w.pt"
        wpath.write_bytes(b"bad")

        class _BoomNet:
            def __init__(self) -> None:
                raise RuntimeError("init boom")

        from unittest.mock import patch

        with (
            patch.object(nn_model, "_TORCH_AVAILABLE", True),
            patch.object(nn_model, "torch", _MockTorch),
            patch.object(nn_model, "F1SetupNet", _BoomNet),
        ):
            mgr = NNModelManager(weights_path=wpath)
            assert mgr.available is False
            assert mgr.model is None

    def test_predict_with_mock_returns_delta(self, tmp_path: Path) -> None:
        """mock PyTorch + 可用模型 → predict 返回 delta 字典。"""
        wpath = tmp_path / "w.pt"
        wpath.write_bytes(b"mock")
        p1, p2, p3 = self._patch_torch()
        with p1, p2, p3:
            mgr = NNModelManager(weights_path=wpath)
            delta = mgr.predict(
                [], {}, CarSetup.default().to_dict(), "suzuka",
            )
            assert delta is not None
            assert len(delta) == len(ALL_SETUP_FIELDS)
            # mock 输出全 0 → delta 全 0
            assert all(v == 0.0 for v in delta.values())

    def test_predict_unavailable_returns_none(self, tmp_path: Path) -> None:
        """mock PyTorch + 不可用模型 → predict 返回 None。"""
        wpath = tmp_path / "nonexistent.pt"
        p1, p2, p3 = self._patch_torch()
        with p1, p2, p3:
            mgr = NNModelManager(weights_path=wpath)
            assert mgr.predict([], {}, {}, "suzuka") is None

    def test_save_weights_with_mock(self, tmp_path: Path) -> None:
        """mock PyTorch + 可用模型 → save_weights 成功。"""
        wpath = tmp_path / "w.pt"
        wpath.write_bytes(b"mock")
        p1, p2, p3 = self._patch_torch()
        with p1, p2, p3:
            mgr = NNModelManager(weights_path=wpath)
            save_path = tmp_path / "saved.pt"
            assert mgr.save_weights(save_path) is True
            assert save_path.exists()

    def test_save_weights_unavailable_returns_false(self, tmp_path: Path) -> None:
        """mock PyTorch + 不可用模型 → save_weights 返回 False。"""
        wpath = tmp_path / "nonexistent.pt"
        p1, p2, p3 = self._patch_torch()
        with p1, p2, p3:
            mgr = NNModelManager(weights_path=wpath)
            assert mgr.save_weights() is False

    def test_save_weights_exception_returns_false(self, tmp_path: Path) -> None:
        """mock PyTorch + 保存异常 → save_weights 返回 False。"""
        wpath = tmp_path / "w.pt"
        wpath.write_bytes(b"mock")

        class _BadSaveNet(_MockNet):
            def state_dict(self) -> dict:
                raise RuntimeError("state_dict boom")

        from unittest.mock import patch

        with (
            patch.object(nn_model, "_TORCH_AVAILABLE", True),
            patch.object(nn_model, "torch", _MockTorch),
            patch.object(nn_model, "F1SetupNet", _BadSaveNet),
        ):
            mgr = NNModelManager(weights_path=wpath)
            # 加载成功（load_state_dict 不调 state_dict）
            assert mgr.available is True
            assert mgr.save_weights(tmp_path / "out.pt") is False

    def test_predict_exception_returns_none(self, tmp_path: Path) -> None:
        """mock PyTorch + 推理异常 → predict 返回 None。"""
        wpath = tmp_path / "w.pt"
        wpath.write_bytes(b"mock")

        class _BoomForwardNet(_MockNet):
            def forward(self, x: object) -> _MockOutput:
                raise RuntimeError("forward boom")

            def __call__(self, x: object) -> _MockOutput:
                return self.forward(x)

        from unittest.mock import patch

        with (
            patch.object(nn_model, "_TORCH_AVAILABLE", True),
            patch.object(nn_model, "torch", _MockTorch),
            patch.object(nn_model, "F1SetupNet", _BoomForwardNet),
        ):
            mgr = NNModelManager(weights_path=wpath)
            assert mgr.available is True
            assert mgr.predict([], {}, {}, "suzuka") is None

    def test_predict_squeeze_scalar_branch(self, tmp_path: Path) -> None:
        """mock 输出 squeeze().tolist() 返回标量时转列表。"""

        class _ScalarSqueezed:
            def tolist(self) -> float:
                return 0.0  # 标量

        class _ScalarOutput:
            def squeeze(self) -> _ScalarSqueezed:
                return _ScalarSqueezed()

        class _ScalarNet(_MockNet):
            def forward(self, x: object) -> _ScalarOutput:
                return _ScalarOutput()

            def __call__(self, x: object) -> _ScalarOutput:
                return self.forward(x)

        wpath = tmp_path / "w.pt"
        wpath.write_bytes(b"mock")
        from unittest.mock import patch

        with (
            patch.object(nn_model, "_TORCH_AVAILABLE", True),
            patch.object(nn_model, "torch", _MockTorch),
            patch.object(nn_model, "F1SetupNet", _ScalarNet),
        ):
            mgr = NNModelManager(weights_path=wpath)
            # 标量 tolist() 触发 isinstance(float) 分支
            # 但 _denormalize_delta 需要长度 20 的列表，标量会 IndexError → 异常 → None
            result = mgr.predict([], {}, {}, "suzuka")
            # 因标量无法索引 20 次，触发异常返回 None
            assert result is None


# ===========================================================================
# 12. _DummyModule 方法覆盖（PyTorch 不可用时）
# ===========================================================================
@pytest.mark.skipif(_TORCH_AVAILABLE, reason="PyTorch 可用，跳过占位类测试")
class TestDummyModuleCoverage:
    """覆盖 _DummyModule 的所有方法（PyTorch 不可用时 F1SetupNet 继承它）。"""

    def test_dummy_module_all_methods(self) -> None:
        """_DummyModule 所有方法可调用且不报错。"""
        from setup_tuner.engine.nn_model import _DummyModule

        dummy = _DummyModule()
        assert dummy.eval() is dummy
        assert dummy.forward("x") == "x"
        dummy.load_state_dict({})
        assert dummy.state_dict() == {}
        assert dummy.parameters() == []
        assert dummy.train() is dummy
        dummy.zero_grad()  # no-op

    def test_f1setupnet_dummy_methods(self) -> None:
        """F1SetupNet（占位）继承 _DummyModule 方法。"""
        net = F1SetupNet()
        assert net.eval() is net
        assert net.forward("test") == "test"
        net.load_state_dict({"a": 1})
        assert net.state_dict() == {}
        assert net.parameters() == []
        assert net.train() is net
        net.zero_grad()


# ===========================================================================
# 13. _normalize_setup span=0 分支覆盖
# ===========================================================================
class TestNormalizeSetupSpanZero:
    """覆盖 _normalize_setup 中 span=0 的分支（用 mock SetupField）。"""

    def test_span_zero_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """构造 min_val==max_val 的字段触发 span=0 分支。"""
        from setup_tuner.domain.setup import SetupField

        # 构造一个 span=0 的字段（min_val == max_val）
        zero_span_field = SetupField(
            name="zero_span", group="Test", label="zero",
            min_val=5.0, max_val=5.0, step=1.0, default=5.0,
            unit="x", max_delta=1.0, source="test",
        )
        # 临时替换 ALL_SETUP_FIELDS

        monkeypatch.setattr(
            "setup_tuner.engine.nn_model.ALL_SETUP_FIELDS", [zero_span_field],
        )
        try:
            vec = _normalize_setup({"zero_span": 5.0})
            assert vec == [0.0]  # span=0 → normalized=0.0
        finally:
            # monkeypatch 会自动恢复，但 _normalize_setup 引用的是模块属性
            pass


# ===========================================================================
# 14. F1SetupNet PyTorch 版本类定义覆盖（用 importlib.reload 模拟 PyTorch）
# ===========================================================================
class TestF1SetupNetPyTorchClassDef:
    """用 importlib.reload 在 mock torch 环境下重新加载 nn_model，
    覆盖 ``if _TORCH_AVAILABLE:`` 分支中 F1SetupNet(nn.Module) 类定义代码。

    这部分代码在 PyTorch 不可用时不会执行，通过 mock torch + nn 模块
    并重新加载 nn_model 来覆盖。
    """

    @pytest.mark.skipif(_TORCH_AVAILABLE, reason="PyTorch 可用，跳过 mock 重载测试")
    def test_reload_with_mock_torch_covers_class_def(self) -> None:
        """mock torch/nn 后 reload nn_model，覆盖 F1SetupNet PyTorch 版本类定义。"""
        import importlib
        import sys
        import types

        # 构造 mock torch 和 torch.nn 模块
        mock_torch = types.ModuleType("torch")
        mock_nn = types.ModuleType("torch.nn")

        class _MockLayer:
            """模拟 nn.Linear / nn.Dropout / nn.ReLU / nn.Tanh。"""

            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def __call__(self, x: object) -> object:
                return x

        class _MockModule:
            """模拟 nn.Module 基类。"""

            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def eval(self) -> _MockModule:
                return self

            def __call__(self, x: object) -> object:
                return x

        mock_nn.Module = _MockModule
        mock_nn.Linear = _MockLayer
        mock_nn.Dropout = _MockLayer
        mock_nn.ReLU = _MockLayer
        mock_nn.Tanh = _MockLayer
        mock_torch.nn = mock_nn
        mock_torch.__version__ = "mock"

        # 保存原始 sys.modules 状态
        original_torch = sys.modules.get("torch")
        original_nn = sys.modules.get("torch.nn")
        original_nn_model = sys.modules.get("setup_tuner.engine.nn_model")

        try:
            sys.modules["torch"] = mock_torch
            sys.modules["torch.nn"] = mock_nn
            # 重新加载 nn_model，触发 if _TORCH_AVAILABLE 分支
            reloaded = importlib.reload(original_nn_model)
            # 验证 F1SetupNet 是 PyTorch 版本（继承 _MockModule 而非 _DummyModule）
            net = reloaded.F1SetupNet()
            assert net.eval() is net
            # 验证层属性存在
            assert hasattr(net, "fc1")
            assert hasattr(net, "fc2")
            assert hasattr(net, "fc3")
            assert hasattr(net, "fc4")
            assert hasattr(net, "dropout")
            assert hasattr(net, "relu")
            assert hasattr(net, "tanh")
            # 验证 forward 可调用
            result = net.forward("input")
            assert result is not None
        finally:
            # 恢复 sys.modules 并重新加载原始模块
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            else:
                sys.modules.pop("torch", None)
            if original_nn is not None:
                sys.modules["torch.nn"] = original_nn
            else:
                sys.modules.pop("torch.nn", None)
            importlib.reload(original_nn_model)