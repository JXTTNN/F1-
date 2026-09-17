"""桌面遥测接收器 —— tkinter 图形界面。

仅由 ``python -m setup_tuner.collector`` 入口加载（CI 测试不 import 本模块，
避免无显示环境/缺 tkinter 的依赖）。每 500ms 轮询 :meth:`CollectorApp.status`
刷新界面，不做跨线程 UI 调用。

界面布局：
- 顶部：监听/录制状态、监听地址、数据目录
- 按钮区：启动/停止监听、开始/停止收集、打开数据目录、导出训练数据
- 中部：实时摘要（赛道/圈/车速/油门/刹车）+ 最近一次收集摘要
- 底部：按包类型的计数（Treeview）
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from setup_tuner.collector.app import CollectorApp
from setup_tuner.collector.export_training import TrainingExporter

logger = logging.getLogger(__name__)

_REFRESH_MS = 500


class CollectorWindow:
    """桌面主窗口（tkinter），包装 :class:`CollectorApp`。"""

    def __init__(self, app: CollectorApp) -> None:
        self._app = app
        self._root = tk.Tk()
        self._root.title("F1OPT 桌面遥测接收器")
        self._root.geometry("680x560")
        self._build_widgets()
        self._refresh()

    # ------------------------------------------------------------------ #
    # 界面构建
    # ------------------------------------------------------------------ #
    def _build_widgets(self) -> None:
        root = self._root
        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        self._lbl_state = ttk.Label(top, text="监听：已停止   收集：未在录制")
        self._lbl_state.pack(anchor="w")
        self._lbl_target = ttk.Label(top, text="")
        self._lbl_target.pack(anchor="w")

        btns = ttk.Frame(root, padding=(8, 0))
        btns.pack(fill="x")
        self._btn_listen = ttk.Button(
            btns, text="启动监听", command=self._toggle_listen,
        )
        self._btn_listen.pack(side="left", padx=4)
        self._btn_collect = ttk.Button(
            btns, text="开始收集", command=self._toggle_collect, state="disabled",
        )
        self._btn_collect.pack(side="left", padx=4)
        ttk.Button(
            btns, text="打开数据目录", command=self._open_data_dir,
        ).pack(side="left", padx=4)
        ttk.Button(
            btns, text="导出训练数据…", command=self._export_training,
        ).pack(side="left", padx=4)

        live = ttk.LabelFrame(root, text="实时摘要", padding=8)
        live.pack(fill="x", padx=8, pady=8)
        self._lbl_live = ttk.Label(live, text="—", font=("Consolas", 10))
        self._lbl_live.pack(anchor="w")

        self._lbl_summary = ttk.Label(live, text="", wraplength=640)
        self._lbl_summary.pack(anchor="w")

        counts = ttk.LabelFrame(root, text="按包类型计数", padding=8)
        counts.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        cols = ("name", "count")
        self._tree = ttk.Treeview(counts, columns=cols, show="headings", height=12)
        self._tree.heading("name", text="包类型")
        self._tree.heading("count", text="计数")
        self._tree.column("name", width=220)
        self._tree.column("count", width=120, anchor="e")
        self._tree.pack(fill="both", expand=True)

    # ------------------------------------------------------------------ #
    # 动作
    # ------------------------------------------------------------------ #
    def _toggle_listen(self) -> None:
        try:
            if self._app.status()["listening"]:
                self._app.stop_listen()
            else:
                self._app.listen()
        except OSError as exc:
            messagebox.showerror("监听失败", str(exc), parent=self._root)
        self._refresh()

    def _toggle_collect(self) -> None:
        try:
            if self._app.status()["recording"]:
                summary = self._app.stop_collect()
                logger.info("collect stopped: %s", summary)
            else:
                self._app.start_collect()
        except Exception as exc:  # noqa: BLE001 —— GUI 层兜底，弹窗不崩溃
            messagebox.showerror("收集失败", str(exc), parent=self._root)
        self._refresh()

    def _open_data_dir(self) -> None:
        path = Path(self._app.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606 —— Windows 资源管理器
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])  # noqa: S603,S607
        else:
            subprocess.Popen(["xdg-open", str(path)])  # noqa: S603,S607

    def _export_training(self) -> None:
        f1rec = filedialog.askopenfilename(
            parent=self._root,
            title="选择录制文件（.f1rec）",
            initialdir=str(self._app.data_dir),
            filetypes=[("F1 录制", "*.f1rec"), ("全部文件", "*.*")],
        )
        if not f1rec:
            return
        out = filedialog.asksaveasfilename(
            parent=self._root,
            title="保存训练样本（JSONL）",
            defaultextension=".jsonl",
            initialfile=Path(f1rec).stem + "_laps.jsonl",
            filetypes=[("JSONL", "*.jsonl")],
        )
        if not out:
            return
        try:
            stats = TrainingExporter().export(f1rec, out)
        except Exception as exc:  # noqa: BLE001 —— GUI 层兜底
            messagebox.showerror("导出失败", str(exc), parent=self._root)
            return
        messagebox.showinfo(
            "导出完成",
            f"圈数 {stats['laps']} / 包数 {stats['packets']} / "
            f"跳过坏包 {stats['parse_errors']}\n{out}",
            parent=self._root,
        )

    # ------------------------------------------------------------------ #
    # 刷新
    # ------------------------------------------------------------------ #
    def _refresh(self) -> None:
        st: dict[str, Any] = self._app.status()
        self._lbl_state.config(
            text=(
                f"监听：{'运行中' if st['listening'] else '已停止'}   "
                f"收集：{'录制中' if st['recording'] else '未在录制'}   "
                f"总包数 {st['packets_total']}"
            ),
        )
        host_disp = (
            f"{st['host']}:{st['port']}（所有网卡）"
            if st["host"] == "0.0.0.0" else f"{st['host']}:{st['port']}"
        )
        self._lbl_target.config(
            text=f"UDP 监听 {host_disp}   数据目录 {st['data_dir']}",
        )
        self._btn_listen.config(
            text="停止监听" if st["listening"] else "启动监听",
        )
        self._btn_collect.config(
            text="停止收集" if st["recording"] else "开始收集",
            state="normal" if st["listening"] else "disabled",
        )
        latest = st["latest"]
        track = latest.get("m_trackId", "—")
        lap = latest.get("m_currentLapNum", "—")
        speed = latest.get("m_speed", "—")
        throttle = latest.get("m_throttle", "—")
        brake = latest.get("m_brake", "—")
        gear = latest.get("m_gear", "—")
        self._lbl_live.config(
            text=(
                f"赛道 ID {track}   圈 {lap}   车速 {speed} km/h   "
                f"挡位 {gear}   油门 {throttle}   刹车 {brake}"
            ),
        )
        summary = st.get("last_summary") or {}
        if summary:
            self._lbl_summary.config(
                text=(
                    "上次收集：包数 {packets}，丢弃 {dropped}，文件 {f1rec}"
                    "（会话 {session}）".format(
                        packets=summary.get("packet_count", 0),
                        dropped=summary.get("dropped_count", 0),
                        f1rec=summary.get("f1rec_path", "-"),
                        session=summary.get("session_id") or "-",
                    )
                ),
            )
        # 计数表
        self._tree.delete(*self._tree.get_children())
        for name, count in sorted(
            st["counts"].items(), key=lambda kv: -kv[1],
        ):
            self._tree.insert("", "end", values=(name, count))
        self._root.after(_REFRESH_MS, self._refresh)

    def run(self) -> None:
        """进入主循环（阻塞）。"""
        self._root.mainloop()


def main() -> int:
    """桌面入口：构造 app + 窗口并进入主循环。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    app = CollectorApp()
    CollectorWindow(app).run()
    app.shutdown()
    return 0
