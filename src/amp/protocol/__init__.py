"""
AMP 交互协议核心。

一次交互回合（Agent A 收到 Agent B 的权重 W_B）：

1. Analysis  — 子空间分析：B 的知识主要在哪些参数子空间？
2. Trust     — 信任评估：B 值得信任吗？
3. Decision  — 自主决策：学不学？学什么？学多少？
4. Integrate — 选择性整合：W' = W_A + M⊙(W_B - W_A)
5. Feedback  — 反馈对话：告诉 B 哪些被接受了
"""
