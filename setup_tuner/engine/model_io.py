"""模型产物路径解析（工作目录覆盖 → 仓库根 → **随包资源**）。

为什么要单独一层
--------------
模型文件必须"仓库开发时能改、pip 安装后能用"：

| 安装形态 | 模型位置 |
|----------|----------|
| 仓库开发（`python -m setup_tuner.cli`） | 优先 `<cwd>/data/models/`（训练脚本默认输出，便于覆盖/实验），再仓库根，最后包内 |
| `pip install`（含从 GitHub 安装） | 仓库根的 `data/` 不存在 → 落到**包内资源** `setup_tuner/resources/models/` |

历史问题：模型只放在仓库根 `data/models/`，`pip install` 后找不到文件 →
`available=False` → silently 降级为纯规则（"装好了却没模型"这一真实故障）。
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PACKAGE_MODELS_DIR", "resolve_model_path", "model_search_paths"]

#: 包内随分发的模型目录（本模块位于 setup_tuner/engine/，故上一级才是包根）
PACKAGE_MODELS_DIR: Path = (
    Path(__file__).resolve().parents[1] / "resources" / "models"
)

#: 工作目录/仓库根下的模型相对路径
WORKDIR_REL: Path = Path("data") / "models"


def model_search_paths(
    name: str, explicit: str | Path | None = None,
) -> list[Path]:
    """按优先级返回候选路径列表（调用方取第一个存在的）。"""
    paths: list[Path] = []
    if explicit is not None:
        paths.append(Path(explicit))
    paths.append(Path.cwd() / WORKDIR_REL / name)
    # 仓库根（源码运行：setup_tuner/engine/model_io.py → parents[2]）
    paths.append(Path(__file__).resolve().parents[2] / WORKDIR_REL / name)
    paths.append(PACKAGE_MODELS_DIR / name)
    # 去重且保持顺序
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def resolve_model_path(
    name: str, explicit: str | Path | None = None,
) -> Path:
    """解析模型文件路径：返回第一个存在的候选，都不存在时返回首选候选。

    返回"首选候选"（而非抛错）是刻意的：调用方据此报出**尝试过的路径**，
    便于用户把模型放到正确位置（错误信息里会列出全部候选）。
    """
    candidates = model_search_paths(name, explicit)
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]  # 包内资源：pip 安装后的规范位置
