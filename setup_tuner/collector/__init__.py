"""桌面遥测接收器（task-81）。

UDP 全类型原始字节无损留存（复用 TelemetryListener + TelemetryRecorder），
供模型训练重放；tkinter 桌面界面由 ``python -m setup_tuner.collector``
启动。训练样本导出见 :class:`TrainingExporter`。
"""

from __future__ import annotations

from setup_tuner.collector.app import CollectorApp
from setup_tuner.collector.export_training import TrainingExporter

__all__ = ["CollectorApp", "TrainingExporter"]
