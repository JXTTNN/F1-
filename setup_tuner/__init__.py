"""F1OPT 赛车调校优化助手（EA F1 2026 Season Pack）。

架构主线（2026-09-20）：

    游戏 UDP 遥测 + 车手反馈
        → 诊断向量 Dx（9 维需求）
        → 耦合矩阵 C：**参数矩阵给方向**（Dx × C → 圈级优化 → 整体收口）
        → **神经网络不断模拟优化**（遥测锚定仿真训练的调教性能 NN，
          纯标准库推理，坐标上升精修；见 engine/setup_sim.py +
          engine/sim_optimizer.py）
        → 整体性调教建议（报告含依据链与模拟轨迹）

依赖口径：运行时**零第三方 AI 依赖** —— 没有 PyTorch / numpy / sklearn，
神经网络是自带的纯标准库 MLP（engine/pure_nn.py）；也没有 LLM 参与调教计算
（给出的是确定性、可复现的建议）。
"""
__version__ = "1.6.10"
