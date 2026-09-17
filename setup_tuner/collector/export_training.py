"""兼容层 —— ``TrainingExporter`` 已上移到 :mod:`setup_tuner.telemetry.training_export`。

上移原因：**桌面接收器与 Web 系统必须共用同一份导出实现**，否则两处逻辑
必然分叉（task-81 教训：同一语义只允许一处实现）。

本模块只做再导出，保持既有导入路径
（``setup_tuner.collector.export_training.TrainingExporter``）继续可用。
新代码请直接从 :mod:`setup_tuner.telemetry.training_export` 导入。
"""

from __future__ import annotations

from setup_tuner.telemetry.training_export import _SETUP_KEYS, TrainingExporter

__all__ = ["TrainingExporter", "_SETUP_KEYS"]
