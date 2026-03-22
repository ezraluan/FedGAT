## `log.txt` 逐行解析（PathMNIST / round0 截止）

> 对应文件：`results/PathMNIST_alpha0.05_10clients/ConvNetBN_1.0%_5000dc_1000epochs_2-2-1/log.txt`  
> 说明：该日志来自 `main.py` / `src/server.py` / `src/client.py` / `dataset/data/dataset.py` 等模块的 `logging.info(...)` 与 `print(...)`。  
> 注意：有些输出（尤其是 `bin edged:`）会**自动换行**，因此会出现“上一行的续行”，这种续行本身不代表新的事件。

---

## 总体结构（先帮助你建立阅读顺序）

- **第 1–2 行**：non-IID 划分统计（每客户端各类样本数、每客户端拥有的类别集合）。
- **第 3–23 行**：全局模型结构、参数量、以及本次运行的完整参数字典（等价于实验配置快照）。
- **第 24 行起**：进入第 0 轮联邦通信（round 0）
  - 对每个客户端：
    - 打印 GPU 显存信息（`pynvml`）
    - 初始化 synthetic 数据（此处为随机噪声）
    - 打印该客户端真实样本数、要浓缩的每类合成样本数（由 `compression_ratio` 推出）
    - 计算并上传原型（feature prototype / logit prototype）
    - 计算“重要性感知采样”的每类采样分布直方图（loss → sigmoid → softmax）
    - 执行知识浓缩迭代（0..5000），每隔 200 步打印一次当前浓缩损失
    - 打印该客户端浓缩耗时、以及每类真实样本数（用于服务端加权聚合原型）

---

## 逐行对照解释

| 行号 | 原始输出（节选） | 含义 / 来自哪里 |
|---:|---|---|
| 1 | `Data statistics: {...}` | **各客户端本地数据的类别计数统计**。键是 client id（0~9），值是 `{class_id: count}`。来自 `main.py` 中对 `split_file` 读取后，用 `np.unique(labels[dataidx])` 统计得到，便于检查 non-IID 程度。 |
| 2 | `client classes: [[4, 7], [0], ...]` | **每个客户端“拥有/参与”的类别集合**（通常要求该类样本数达到一定阈值才会被记入）。来自 `split_file` 的 `client_classes` 字段（由 `dataset_partition.py` 生成）。 |
| 3 | `ConvNet(` | 开始打印全局模型结构（`logging.info(global_model)`）。模型由 `src/utils.py::get_model()` 创建。 |
| 4 | `(net_act): ReLU(inplace=True)` | ConvNet 的激活函数设置为 ReLU。来自 `src/models.py::ConvNet`。 |
| 5 | `(net_pooling): AvgPool2d(...)` | ConvNet 的池化层为平均池化。 |
| 6 | `(features): Sequential(` | 特征提取 backbone（卷积 + BN + 激活 + 池化）是一个 `nn.Sequential`。 |
| 7 | `(0): Conv2d(3, 128, ...)` | 第 1 层卷积：输入 3 通道（PathMNIST 为 RGB），输出 128。 |
| 8 | `(1): BatchNorm2d(128, ...)` | 第 1 个 BatchNorm2d。这里说明你选的模型是 `ConvNetBN`（包含 BN）。 |
| 9 | `(2): ReLU(inplace=True)` | 第 1 个 ReLU。 |
| 10 | `(3): AvgPool2d(...)` | 第 1 次池化，下采样。 |
| 11 | `(4): Conv2d(128, 128, ...)` | 第 2 层卷积。 |
| 12 | `(5): BatchNorm2d(128, ...)` | 第 2 个 BN。 |
| 13 | `(6): ReLU(inplace=True)` | 第 2 个 ReLU。 |
| 14 | `(7): AvgPool2d(...)` | 第 2 次池化。 |
| 15 | `(8): Conv2d(128, 128, ...)` | 第 3 层卷积。 |
| 16 | `(9): BatchNorm2d(128, ...)` | 第 3 个 BN。 |
| 17 | `(10): ReLU(inplace=True)` | 第 3 个 ReLU。 |
| 18 | `(11): AvgPool2d(...)` | 第 3 次池化。 |
| 19 | `)` | 结束 `features` 的打印。 |
| 20 | `(classifier): Linear(in_features=2048, out_features=9, ...)` | 最后一层线性分类器：输出类别数为 9（PathMNIST）。`in_features=2048` 是经过卷积池化后展平的特征维度。 |
| 21 | `)` | 结束模型打印。 |
| 22 | `317961` | **模型参数量**（`get_n_params(global_model)` 结果）。 |
| 23 | `{ 'seed': ..., 'dataset': 'PathMNIST', ... }` | **本次运行的完整参数快照**（`args.__dict__`）。重点字段：`compression_ratio=0.01`、`dc_iterations=5000`、`weighted_mmd=True`、`init=random_noise`、`con_beta=0.05` 等。 |
| 24 | `====== round 0 ======` | 进入第 0 轮联邦通信。来自 `src/server.py::Server.fit()` 循环。 |
| 25 | `---------- client training ----------` | Round 0 的客户端侧开始执行（每个 client 将进行“浓缩”并上传）。 |
| 26 | `selected clients: [0,1,2,...,9]` | 本轮参与训练的客户端列表。这里 `join_ratio=1.0`，所以全员参加。 |
| 27 | `total 24564.0MB, used ...` | GPU 显存监控输出（NVML）。来自 `get_gpu_mem_info()`（client/server 中都有调用）。 |
| 28 | `initialized by random noise` | 客户端合成数据 `synthetic_images` 的初始化策略：随机噪声（对应 `--init random_noise`）。来自 `Client.initialization()`。 |
| 29 | `client 0 have real samples [5777, 9330]` | client0 拥有的真实样本数（按其 `classes` 中的类别顺序列出）。来自 `Client.train_weighted_MMD()`。 |
| 30 | `client 0 will condense {4: 58, 7: 94} ...` | client0 将对每个类别生成多少张合成图（`ipc_dict`）：由 `compression_ratio=0.01` 和每类真实样本数决定，且每类最少 5。 |
| 31 | `get_feature_prototype` | client0 正在计算 **feature prototype**：每类真实样本的 embedding 均值。来自 `Client.get_feature_prototype()`。 |
| 32 | `get_logit_prototype` | client0 正在计算 **logit prototype**：每类真实样本的 logit 均值。来自 `Client.get_logit_prototype()`。 |
| 33 | `loss weighted matching the samples` | 开始计算“重要性感知采样”所需的 per-sample loss，并据此得到每类采样分布。来自 `Client.train_weighted_MMD()` 调用 `train_set.cal_loss()`/`pre_sample()`。 |
| 34 | `class 4 have 5777 samples, histogram: ... bin edged: [...]` | **重要性采样分布直方图（类 4）**：`sample_prob[c]=softmax(sigmoid(loss))` 的分布被分 10 个 bin 统计。`bin edged:` 后跟 10 个箱体边界值。来自 `PerLabelDatasetNonIID.pre_sample()`。 |
| 35 | `0.00017308 ... 0.00017327]` | **第 34 行的续行**：因为 `bin edged` 数组太长自动换行。这一行没有新事件。 |
| 36 | `class 7 have 9330 samples, histogram: ... bin edged: [...]` | 同第 34 行，但对应类别 7。 |
| 37 | `0.00010722 ... 0.0001073 ]` | **第 36 行的续行**（bin edges 的剩余部分）。 |
| 38 | `client 0, data condensation 0, total loss = ...` | client0 的**浓缩迭代**进度打印：`dc_iteration=0` 时的浓缩损失。此处走 `weighted_mmd` 路径，loss 是对各类做 `M3DLoss(real_feature_c, syn_feature_c)` 的求和（核化 MMD）。 |
| 39 | `client 0, data condensation 200, ...` | 同上：每 200 步打印一次当前 loss（用于观察收敛/震荡）。 |
| 40 | `client 0, data condensation 400, ...` | 同上。 |
| 41 | `client 0, data condensation 600, ...` | 同上。 |
| 42 | `client 0, data condensation 800, ...` | 同上。 |
| 43 | `client 0, data condensation 1000, ...` | 同上。 |
| 44 | `client 0, data condensation 1200, ...` | 同上（这里 loss 突然变大，常见原因：采样 batch 变化、增强随机性、核估计波动、学习率/梯度裁剪影响等）。 |
| 45 | `client 0, data condensation 1400, ...` | 同上。 |
| 46 | `client 0, data condensation 1600, ...` | 同上。 |
| 47 | `client 0, data condensation 1800, ...` | 同上。 |
| 48 | `client 0, data condensation 2000, ...` | 同上。 |
| 49 | `client 0, data condensation 2200, ...` | 同上。 |
| 50 | `client 0, data condensation 2400, ...` | 同上。 |
| 51 | `client 0, data condensation 2600, ...` | 同上。 |
| 52 | `client 0, data condensation 2800, ...` | 同上。 |
| 53 | `client 0, data condensation 3000, ...` | 同上。 |
| 54 | `client 0, data condensation 3200, ...` | 同上。 |
| 55 | `client 0, data condensation 3400, ...` | 同上。 |
| 56 | `client 0, data condensation 3600, ...` | 同上。 |
| 57 | `client 0, data condensation 3800, ...` | 同上。 |
| 58 | `client 0, data condensation 4000, ...` | 同上。 |
| 59 | `client 0, data condensation 4200, ...` | 同上。 |
| 60 | `client 0, data condensation 4400, ...` | 同上。 |
| 61 | `client 0, data condensation 4600, ...` | 同上（再次出现大波动）。 |
| 62 | `client 0, data condensation 4800, ...` | 同上。 |
| 63 | `client 0, data condensation 5000, ...` | client0 最后一次打印：到达 `dc_iterations=5000`。 |
| 64 | `Round 0, client 0 condense time: 214.10...` | client0 本轮浓缩耗时（秒）。来自 `Server.fit()` 计时。 |
| 65 | `client 0, class 4 have 5777 samples` | client0 上传原型时附带的该类真实样本数，用于服务端按样本数加权聚合原型。 |
| 66 | `client 0, class 7 have 9330 samples` | 同上，类别 7。 |
| 67 | `total ... used ... free ...` | 显存监控（通常在一个 client 结束后打印一次）。 |
| 68 | `total ... used ... free ...` | 显存监控（重复打印一次，来自不同位置的调用）。 |
| 69 | `initialized by random noise` | client1 开始：合成数据初始化为随机噪声。 |
| 70 | `client 1 have real samples [9022]` | client1 只有类别 0，真实样本数 9022。 |
| 71 | `client 1 will condense {0: 91} ...` | 按 `compression_ratio=0.01` 得到合成样本数约 91。 |
| 72 | `get_feature_prototype` | client1 计算 feature prototype。 |
| 73 | `get_logit_prototype` | client1 计算 logit prototype。 |
| 74 | `loss weighted matching the samples` | client1 计算重要性采样分布。 |
| 75 | `class 0 have 9022 samples, histogram: ... bin edged: [...]` | 类 0 的采样分布直方图与 bin edges（第一行）。 |
| 76 | `0.00011093 ... 0.00011106]` | **第 75 行的续行**（bin edges 的剩余部分）。 |
| 77 | `client 1, data condensation 0, total loss = ...` | client1 的浓缩迭代打印（核化 MMD loss）。 |
| 78 | `client 1, data condensation 200, ...` | 同上。 |
| 79 | `client 1, data condensation 400, ...` | 同上。 |
| 80 | `client 1, data condensation 600, ...` | 同上。 |
| 81 | `client 1, data condensation 800, ...` | 同上。 |
| 82 | `client 1, data condensation 1000, ...` | 同上。 |
| 83 | `client 1, data condensation 1200, ...` | 同上。 |
| 84 | `client 1, data condensation 1400, ...` | 同上。 |
| 85 | `client 1, data condensation 1600, ...` | 同上。 |
| 86 | `client 1, data condensation 1800, ...` | 同上。 |
| 87 | `client 1, data condensation 2000, ...` | 同上。 |
| 88 | `client 1, data condensation 2200, ...` | 同上。 |
| 89 | `client 1, data condensation 2400, ...` | 同上。 |
| 90 | `client 1, data condensation 2600, ...` | 同上。 |
| 91 | `client 1, data condensation 2800, ...` | 同上。 |
| 92 | `client 1, data condensation 3000, ...` | 同上。 |
| 93 | `client 1, data condensation 3200, ...` | 同上。 |
| 94 | `client 1, data condensation 3400, ...` | 同上。 |
| 95 | `client 1, data condensation 3600, ...` | 同上（波动较大）。 |
| 96 | `client 1, data condensation 3800, ...` | 同上。 |
| 97 | `client 1, data condensation 4000, ...` | 同上。 |
| 98 | `client 1, data condensation 4200, ...` | 同上。 |
| 99 | `client 1, data condensation 4400, ...` | 同上。 |
| 100 | `client 1, data condensation 4600, ...` | 同上。 |
| 101 | `client 1, data condensation 4800, ...` | 同上。 |
| 102 | `client 1, data condensation 5000, ...` | 同上（最后一步）。 |
| 103 | `Round 0, client 1 condense time: 104.91...` | client1 本轮浓缩耗时（秒）。 |
| 104 | `client 1, class 0 have 9022 samples` | client1 的类别 0 样本数（用于服务端聚合原型加权）。 |
| 105 | `total ...` | 显存监控。 |
| 106 | `total ...` | 显存监控（重复）。 |
| 107 | `initialized by random noise` | client2 开始：随机噪声初始化合成数据。 |
| 108 | `client 2 have real samples [327, 12176]` | client2 拥有类别 0 与 5：分别 327、12176。 |
| 109 | `client 2 will condense {0: 5, 5: 122} ...` | 合成样本数：类 0 因真实样本很少，被 `max(5, ceil(n*ratio))` 卡到最少 5；类 5 为 122。 |
| 110 | `get_feature_prototype` | 计算 feature prototype。 |
| 111 | `get_logit_prototype` | 计算 logit prototype。 |
| 112 | `loss weighted matching the samples` | 计算重要性采样分布。 |
| 113 | `class 0 have 327 samples, histogram: ... bin edged: [...]` | 类 0 的采样分布直方图与 bin edges（第一行）。 |
| 114 | `0.00305823 ... 0.00305946]` | **第 113 行的续行**（bin edges 的剩余部分）。 |
| 115 | `class 5 have 12176 samples, histogram: ... bin edged: [...]` | 类 5 的采样分布直方图与 bin edges（第一行，科学计数法）。 |
| 116 | `8.21079219e-05 ...` | **第 115 行的续行**。 |
| 117 | `8.21696455e-05 ... 8.22005073e-05]` | **第 115 行的续行（继续）**。 |
| 118 | `client 2, data condensation 0, total loss = ...` | client2 浓缩 loss 打印。 |
| 119 | `client 2, data condensation 200, ...` | 同上。 |
| 120 | `client 2, data condensation 400, ...` | 同上。 |
| 121 | `client 2, data condensation 600, ...` | 同上。 |
| 122 | `client 2, data condensation 800, ...` | 同上。 |
| 123 | `client 2, data condensation 1000, ...` | 同上。 |
| 124 | `client 2, data condensation 1200, ...` | 同上（波动较大）。 |
| 125 | `client 2, data condensation 1400, ...` | 同上。 |
| 126 | `client 2, data condensation 1600, ...` | 同上。 |
| 127 | `client 2, data condensation 1800, ...` | 同上。 |
| 128 | `client 2, data condensation 2000, ...` | 同上。 |
| 129 | `client 2, data condensation 2200, ...` | 同上。 |
| 130 | `client 2, data condensation 2400, ...` | 同上。 |
| 131 | `client 2, data condensation 2600, ...` | 同上。 |
| 132 | `client 2, data condensation 2800, ...` | 同上（波动）。 |
| 133 | `client 2, data condensation 3000, ...` | 同上。 |
| 134 | `client 2, data condensation 3200, ...` | 同上。 |
| 135 | `client 2, data condensation 3400, ...` | 同上。 |
| 136 | `client 2, data condensation 3600, ...` | 同上。 |
| 137 | `client 2, data condensation 3800, ...` | 同上。 |
| 138 | `client 2, data condensation 4000, ...` | 同上。 |
| 139 | `client 2, data condensation 4200, ...` | 同上。 |
| 140 | `client 2, data condensation 4400, ...` | 同上。 |
| 141 | `client 2, data condensation 4600, ...` | 同上。 |
| 142 | `client 2, data condensation 4800, ...` | 同上。 |
| 143 | `client 2, data condensation 5000, ...` | 同上（最后一步）。 |
| 144 | `Round 0, client 2 condense time: 178.21...` | client2 本轮浓缩耗时。 |
| 145 | `client 2, class 0 have 327 samples` | 类 0 真实样本数（聚合权重）。 |
| 146 | `client 2, class 5 have 12176 samples` | 类 5 真实样本数。 |
| 147 | `total ...` | 显存监控。 |
| 148 | `total ...` | 显存监控（重复）。 |
| 149 | `initialized by random noise` | client3 开始：随机噪声初始化。 |
| 150 | `client 3 have real samples [313]` | client3 只有类别 6，真实样本数 313。 |
| 151 | `client 3 will condense {6: 5} ...` | 类 6 合成样本数被最小值 5 限制。 |
| 152 | `get_feature_prototype` | 计算 feature prototype。 |
| 153 | `get_logit_prototype` | 计算 logit prototype。 |
| 154 | `loss weighted matching the samples` | 计算重要性采样分布。 |
| 155 | `class 6 have 313 samples, histogram: ... bin edged: [...]` | 类 6 的采样分布直方图与 bin edges（第一行）。 |
| 156 | `0.00319542 ... 0.00319708]` | **第 155 行的续行**。 |
| 157 | `client 3, data condensation 0, ...` | client3 浓缩 loss 打印。 |
| 158 | `client 3, data condensation 200, ...` | 同上。 |
| 159 | `client 3, data condensation 400, ...` | 同上。 |
| 160 | `client 3, data condensation 600, ...` | 同上（波动）。 |
| 161 | `client 3, data condensation 800, ...` | 同上。 |
| 162 | `client 3, data condensation 1000, ...` | 同上。 |
| 163 | `client 3, data condensation 1200, ...` | 同上。 |
| 164 | `client 3, data condensation 1400, ...` | 同上。 |
| 165 | `client 3, data condensation 1600, ...` | 同上。 |
| 166 | `client 3, data condensation 1800, ...` | 同上。 |
| 167 | `client 3, data condensation 2000, ...` | 同上（波动）。 |
| 168 | `client 3, data condensation 2200, ...` | 同上。 |
| 169 | `client 3, data condensation 2400, ...` | 同上。 |
| 170 | `client 3, data condensation 2600, ...` | 同上。 |
| 171 | `client 3, data condensation 2800, ...` | 同上。 |
| 172 | `client 3, data condensation 3000, ...` | 同上。 |
| 173 | `client 3, data condensation 3200, ...` | 同上。 |
| 174 | `client 3, data condensation 3400, ...` | 同上。 |
| 175 | `client 3, data condensation 3600, ...` | 同上。 |
| 176 | `client 3, data condensation 3800, ...` | 同上。 |
| 177 | `client 3, data condensation 4000, ...` | 同上。 |
| 178 | `client 3, data condensation 4200, ...` | 同上。 |
| 179 | `client 3, data condensation 4400, ...` | 同上。 |
| 180 | `client 3, data condensation 4600, ...` | 同上（波动很大）。 |
| 181 | `client 3, data condensation 4800, ...` | 同上。 |
| 182 | `client 3, data condensation 5000, ...` | 同上（最后一步）。 |
| 183 | `Round 0, client 3 condense time: 64.25...` | client3 本轮浓缩耗时。 |
| 184 | `client 3, class 6 have 313 samples` | 类 6 真实样本数。 |
| 185 | `total ...` | 显存监控。 |
| 186 | `total ...` | 显存监控（重复）。 |
| 187 | `initialized by random noise` | client4 开始：随机噪声初始化。 |
| 188 | `client 4 have real samples [361, 1048, 7572]` | client4 拥有类别 1、4、6：真实样本数分别 361、1048、7572。 |
| 189 | `client 4 will condense {1: 5, 4: 11, 6: 76} ...` | 合成样本数：类 1 因样本少被最小值 5 限制；类 4 约 11；类 6 约 76。 |
| 190 | `get_feature_prototype` | 计算 feature prototype。 |
| 191 | `get_logit_prototype` | 计算 logit prototype。 |
| 192 | `loss weighted matching the samples` | 计算重要性采样分布。 |
| 193 | `class 1 have 361 samples, histogram: ... bin edged: [...]` | 类 1 的采样分布直方图与 bin edges（第一行）。 |
| 194 | `0.00277132 ... 0.00277405]` | **第 193 行的续行**。 |
| 195 | `class 4 have 1048 samples, histogram: ... bin edged: [...]` | 类 4 的采样分布直方图与 bin edges（第一行）。 |
| 196 | `0.00095427 ... 0.00095491]` | **第 195 行的续行**。 |
| 197 | `class 6 have 7572 samples, histogram: ... bin edged: [...]` | 类 6 的采样分布直方图与 bin edges（第一行）。 |
| 198 | `0.00013218 ... 0.00013235]` | **第 197 行的续行**。 |
| 199 | `client 4, data condensation 0, ...` | client4 浓缩 loss 打印。 |
| 200 | `client 4, data condensation 200, ...` | 同上。 |
| 201 | `client 4, data condensation 400, ...` | 同上。 |
| 202 | `client 4, data condensation 600, ...` | 同上。 |
| 203 | `client 4, data condensation 800, ...` | 同上。 |
| 204 | `client 4, data condensation 1000, ...` | 同上。 |
| 205 | `client 4, data condensation 1200, ...` | 同上。 |
| 206 | `client 4, data condensation 1400, ...` | 同上。 |
| 207 | `client 4, data condensation 1600, ...` | 同上。 |
| 208 | `client 4, data condensation 1800, ...` | 同上。 |
| 209 | `client 4, data condensation 2000, ...` | 同上（波动较大）。 |
| 210 | `client 4, data condensation 2200, ...` | 同上。 |
| 211 | `client 4, data condensation 2400, ...` | 同上。 |
| 212 | `client 4, data condensation 2600, ...` | 同上。 |
| 213 | `client 4, data condensation 2800, ...` | 同上。 |
| 214 | *(文件在此处截断)* | 该 `log.txt` 到第 214 行就结束了，说明程序在 client4 还没跑完或后续 client5~9 未开始之前被中断（例如你后续遇到的联网下载失败、手动停止等）。 |

---

## 你这份 log 的两个关键信号（帮你快速读懂“训练在干什么”）

- **“histogram/bin edged” 行（如 34–37、75–76…）**  
  这是 FedVCK 3.3“重要性感知采样”的可视化：每个类别的样本被赋予权重后，softmax 得到采样概率分布；直方图越偏向后面的 bin，表示少数样本权重更大（更“难/缺失知识”）。

- **“data condensation XXX, total loss=...” 行**  
  这是客户端侧在优化 `synthetic_images`：让合成数据在 embedding 空间匹配真实数据（这里启用了 `--weighted_mmd`，即核化 MMD）。loss 大起大落不一定是 bug，更多是 batch/采样变化带来的估计波动。

