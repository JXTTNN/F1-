"""切片级深度测试包。

对 setup_tuner 的 5 个核心切片分别用 5 种测试方式（unit / boundary / property /
static / smoke）进行深度测试，覆盖正常输入、边界异常、属性不变量、静态约束
与真实调用链路冒烟。
"""