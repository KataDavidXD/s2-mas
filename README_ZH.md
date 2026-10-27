# S²-MAS：面向社会规模模拟的多智能体框架与五亿家庭画像资源

S²-MAS 将家庭画像、学习型决策模型与金融账本连接起来。模型统一使用 **YN 系列**名称：YN-mini、YN-Lite-4B、YN-master-9B；后缀区分训练变体和运行格式。

- **5 亿合成家庭画像**构成人口资源。
- **1,200 万个不同家庭**完成配对购买决策实验：2,400 万次一次性选择、2.88 亿个月度账本状态，14 张 GPU 用时 3 小时 59 分 17 秒。
- **YN-mini-MLX 准确率 74.55%、题目宏平均 F1 38.37%**，共同测试包含 240,458 条记录。
- **1,060 户家庭消费实验**集中呈现 4B/9B 独立决策、完全理性基准、带建议变体、分组 MAE、未来信息输入测试及消费—现金配对变化。独立 9B 的全年 MAE 比确定性理性基准低 3.42%，预测总体均值同时比观测值低 19.7%。

[最新完整论文](paper/S2_MAS_Paper.pdf) · [LaTeX 与论文图](paper/) · [聚合结果](paper/source_data/) · [English](README.md)

论文 v67 · 2026 年 10 月 28 日 · 56 页 · 正文 13 图、补充 8 图 · [arXiv:2610.27746](https://arxiv.org/abs/2610.27746)。

![框架](paper/figures/F1.png)

![家庭消费实验](paper/figures/household_scenario.png)

资源入口：[模型目录](resources/MODEL_CATALOG.md)、[科研绘图](paper/figure_code/)、[模型训练与评测](reproduction/training/README.md)、[独立合成配套数据](resources/public_companion/README.md)、[Wikidata 组件](resources/wikidata/README.md)。既有模型仓库链接继续有效。

5 亿资源、1,200 万执行队列与公开合成配套数据用途不同。受控状态写回实验最高测试 1,000 个 agent；大规模实验测试输入准备、推理和独立账本流水线。决策吞吐量统计处理的输入 token。

公开科研代码采用 [Apache-2.0](LICENSE)；GENESE/MAS 引擎实现保持私有。配套数据与模型按各自组件许可使用。

科研联系：[yl9919@connect.hku.hk](mailto:yl9919@connect.hku.hk)。
