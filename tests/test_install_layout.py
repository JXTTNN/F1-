"""安装形态与数据落点契约（用户要求：安装后即完整可用、数据只在安装文件夹内）。

锁两件事：

1. **数据只在安装文件夹内**：默认数据目录 = `<安装根>/data`，且**与当前工作
   目录无关**（从任意 cwd 启动，录制/数据库/派生数据都落在安装文件夹里）。
   安装根 = 含 `setup_tuner/` 包的那一层（仓库开发时是仓库根，`pip install`
   后是 `Lib/site-packages/`）。
2. **安装即完整**：运行时需要的资源全部随包分发 —— 训练好的模型（两个）、
   UI（HTML/JS/CSS + 全部赛道 SVG）、SQLite schema、24 条赛道数据。
   `pip install` 之后不应出现"少了文件 → 静默降级"。

注意：本文件只**解析路径**，不真的在安装目录里建文件（避免污染工作区）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from setup_tuner.config import Config, default_data_dir, install_root, load_config

_PKG = install_root() / "setup_tuner"


class TestDataLocation:
    """数据目录固定在安装文件夹内（不跟随 cwd）。"""

    def test_default_data_dir_is_inside_install_root(self) -> None:
        data_dir = Path(default_data_dir()).resolve()
        root = install_root().resolve()
        assert data_dir == root / "data"
        assert data_dir.is_absolute(), "数据目录必须是绝对路径（否则会跟随 cwd）"
        assert root in data_dir.parents or data_dir.parent == root

    def test_resolution_ignores_cwd(self, tmp_path: Path, monkeypatch) -> None:
        """换到任意工作目录，解析结果不变 —— 这是"数据只在安装文件夹"的实质。"""
        expected = Path(default_data_dir()).resolve()
        for cwd in (tmp_path, tmp_path / "deep" / "nested", Path.home()):
            if not cwd.exists():
                cwd.mkdir(parents=True, exist_ok=True)
            monkeypatch.chdir(cwd)
            assert Path(load_config(env_path=tmp_path / "none.env")
                        .resolved_data_dir()).resolve() == expected
            assert Path(Config().resolved_data_dir()).resolve() == expected

    def test_explicit_data_dir_wins(self, tmp_path: Path, monkeypatch) -> None:
        """显式设置 F1OPT_DATA_DIR 仍可覆盖（高级用法）。"""
        target = tmp_path / "custom"
        monkeypatch.setenv("F1OPT_DATA_DIR", str(target))
        cfg = load_config(env_path=tmp_path / "none.env")
        assert Path(cfg.resolved_data_dir()) == target

    def test_env_file_is_read_from_cwd(self, tmp_path: Path, monkeypatch) -> None:
        """默认读 cwd 的 .env（配置来源不受数据目录口径影响）。"""
        (tmp_path / ".env").write_text("API_PORT=12345\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        for key in ("API_PORT", "F1OPT_API_PORT"):
            monkeypatch.delenv(key, raising=False)
        assert load_config().api_port == 12345


class TestInstalledPackageIsComplete:
    """运行时资源必须全部在包内（安装即完整，和本地一样）。"""

    def test_trained_models_shipped(self) -> None:
        models = _PKG / "resources" / "models"
        assert models.is_dir(), f"包内缺少模型目录：{models}"
        for name in ("setup_sim_nn.json", "telemetry_surrogate.json"):
            path = models / name
            assert path.is_file(), f"包内缺少模型：{name}"
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload, f"{name} 内容为空"
        # 调教性能模型必须真的能用（不只是文件存在）
        from setup_tuner.engine.setup_sim import SetupSimModel

        sim = SetupSimModel(models / "setup_sim_nn.json")
        assert sim.available is True, f"随包分发的模型无法加载：{sim.reason}"
        assert len(sim.track_ids) >= 13, f"模型覆盖赛道过少：{len(sim.track_ids)}"

    def test_models_resolvable_from_any_cwd(self, tmp_path: Path, monkeypatch) -> None:
        """从任意 cwd 解析模型，最终必须落到包内资源（而不是找不到）。"""
        from setup_tuner.engine import model_io, setup_sim, surrogate

        monkeypatch.chdir(tmp_path)
        for module, name in (
            (setup_sim, setup_sim.MODEL_NAME),
            (surrogate, surrogate.MODEL_NAME),
        ):
            resolved = module._resolve_path(Path("data") / "models" / name)
            assert resolved.exists(), f"{name} 从 {tmp_path} 解析失败：{resolved}"
            assert resolved.resolve() == (
                model_io.PACKAGE_MODELS_DIR / name
            ).resolve()

    def test_ui_assets_shipped(self) -> None:
        ui = _PKG / "ui"
        for name in ("index.html", "app.js", "style.css"):
            assert (ui / name).is_file(), f"包内缺少 UI 资源：{name}"
        tracks = sorted((ui / "tracks").glob("*.svg"))
        assert len(tracks) >= 24, f"赛道 SVG 数量不足：{len(tracks)}"

    def test_db_schema_shipped(self) -> None:
        sql = sorted((_PKG / "db").glob("*.sql"))
        assert sql, "包内缺少 SQLite schema（db/*.sql）"

    def test_track_data_complete(self) -> None:
        """24 条赛道（含 2026 新赛道）必须在包内可用。"""
        from setup_tuner.domain.track import get_all_tracks

        tracks = get_all_tracks()
        assert len(tracks) >= 24, f"赛道数量不足：{len(tracks)}"
        ids = {t.track_id for t in tracks}
        for required in ("madrid", "suzuka", "monaco", "spa", "monza"):
            assert required in ids, f"缺少赛道：{required}"

    def test_no_runtime_dependency_on_repo_root(self, tmp_path: Path, monkeypatch) -> None:
        """运行时不得依赖仓库根的 data/（安装形态下它不存在）。

        做法：把 cwd 换到空目录、并断言模型解析与调教建议仍可用。
        """
        monkeypatch.chdir(tmp_path)
        from setup_tuner.domain.setup import CarSetup
        from setup_tuner.engine.engine import generate_suggestion

        res = generate_suggestion(
            [("understeer", 2)], CarSetup.default().to_dict(), "suzuka", None,
            model_type="hybrid",
        )
        assert res["model_type"] == "nn", (
            "脱离仓库根运行时应仍能用模型（否则安装后等于降级）："
            f"{res['holistic']['simulation']['reason']}"
        )


@pytest.mark.parametrize("key", ["DATA_DIR", "F1OPT_DATA_DIR"])
def test_data_dir_env_both_spellings(key: str, tmp_path: Path, monkeypatch) -> None:
    """两种写法都认（README 文档化 F1OPT_*，代码曾只读无前缀）。"""
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("F1OPT_DATA_DIR", raising=False)
    monkeypatch.setenv(key, str(tmp_path / "x"))
    assert Path(load_config(
        env_path=tmp_path / "none.env",
    ).resolved_data_dir()) == tmp_path / "x"
    os.environ.pop(key, None)
