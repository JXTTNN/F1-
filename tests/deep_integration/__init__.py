"""协作集成测试包 —— 验证关联模块之间的协作（3 种方式）。

子模块：
    - test_data_flow.py      ：方式 1 数据流串接（模块间数据正确传递）
    - test_e2e_business.py   ：方式 2 端到端业务闭环（TestClient 真实启动 API）
    - test_concurrent.py     ：方式 3 并发共享状态（多线程并发访问共享资源）
"""