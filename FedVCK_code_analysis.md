# FedVCK_2024 代码梳理与 FedVCK 方法对齐报告

> 面向你给的 FedVCK 论文方法（3.1–3.4）做“代码做了什么”的逐段对齐说明。  
> 结论先说：**项目实现了“联邦训练 + 客户端数据集浓缩（可选：均值匹配 / 核化 MMD）+ 基于样本难度（损失）加权采样 + 服务端基于原型关系的监督式对比学习”** 的完整闭环。

---

## 项目入口与整体结构

- **入口脚本**：`main.py`
  - 负责解析参数、构造数据划分、初始化全局模型与 `Server/Client`，然后执行 `server.fit()`。
- **核心算法代码（你提到的 `src/`）**：
  - `src/client.py`：客户端侧“浓缩知识数据集” \(S_k\) 的优化（含重要性感知采样）。
  - `src/server.py`：服务端通信轮流程、聚合原型、构造对比学习关系集合、用累计知识数据更新全局模型。
  - `src/utils.py`：核函数/MMD、DiffAugment、模型构建、对比学习损失等通用组件。
  - `src/models.py`：ConvNet/ResNet18 等模型与 `embed()`（把输入映射到潜在表征）。
- **数据集与划分**：
  - `dataset/data/dataset.py`：加载数据集 + `PerLabelDatasetNonIID`（将本地数据按类别组织、计算样本权重、按权重采样）。
  - `dataset/data/dataset_partition.py`：生成 non-IID 划分 json（Dirichlet/label/pathological）。

---

## 运行流程（从 `main.py` 看端到端）

`main.py` 的关键流程可以概括为：

1. **读取划分文件**：按 `args.partition` 找到 `dataset/split_file/*.json`，得到每个客户端的样本索引与拥有的类别集合。
2. **构造客户端训练集**：
   - `train_sets[i] = Subset(train_set, indices)`
   - 再包装成 `PerLabelDatasetNonIID(train_sets[i], client_classes[i], ...)`，以便：
     - 直接拿到该 client 的 `images_all`（tensor）；
     - 快速按类别取样本；
     - 计算“重要性采样分布”。
3. **构造全局模型**：`global_model = get_model(args.model, dataset_info)`（定义在 `src/utils.py`）。
4. **构造 `Client` 列表**：每个 client 拥有：
   - 自己的 `synthetic_images`（可学习知识集 \(S_k\)）与 `synthetic_labels`；
   - 当前/上一轮全局模型快照（用于重要性估计）。
5. **构造 `Server`** 并启动训练：`server.fit()`

---

## FedVCK 3.1：分布匹配（Distribution Matching / MMD）

论文 3.1 的核心是让浓缩数据集 \(S\) 在潜在空间里匹配真实数据 \(T\) 的分布，常见是“按类对齐 + MMD”。

### 代码对应位置

#### 1) “按类对齐的均值匹配”（更像式(2)的特征均值对齐）
- **实现**：`Client.train_weighted_sample()`（`src/client.py`）
- **做法**：
  - 每轮每个类别 \(c\) 取一个真实 batch `real_image`（但不是均匀采样，而是后面 3.3 的加权采样）。
  - 用全局模型提取嵌入：
    - `real_feature = global_model.embed(real_images).detach()`
    - `synthetic_feature = global_model.embed(synthetic_images)`
  - 逐类计算嵌入均值差的 \(L_2^2\)：
    - `loss += sum((mean_real - mean_syn)**2)`

这对应论文中的条件分布匹配（按类对齐潜在特征的“均值”，属于 MMD 的一阶矩版本）。

#### 2) “核化 MMD”（更像式(3)–(4)的 RKHS 对齐）
- **实现**：`Client.train_weighted_MMD()`（`src/client.py`）
- **损失**：`mmd_criterion = M3DLoss(kernel_type=args.kernel, ...)`（`src/utils.py`）
  - `M3DLoss` 内部先把 `X` 与 `Y` 堆叠，然后算核矩阵 \(K\)，最后按均值估计返回：
    - \(\mathrm{MMD}^2(X,Y)=\mathbb{E}[K(X,X)]-2\mathbb{E}[K(X,Y)]+\mathbb{E}[K(Y,Y)]\)
  - 对应到你文档式(3)/(4)时，可以把代码里的三项分别看作：
    - `XX = K[:|X|,:|X|].mean()` \(\leftrightarrow \hat K^{B_c,B_c}\)
    - `YY = K[|X|:,|X|:].mean()` \(\leftrightarrow \hat K^{S_c,S_c}\)
    - `XY = K[:|X|,|X|:].mean()` \(\leftrightarrow \hat K^{B_c,S_c}\)
    - `return XX - 2*XY + YY` \(\leftrightarrow \hat K^{B_c,B_c} + \hat K^{S_c,S_c} - 2\hat K^{B_c,S_c}\)
  - 支持核类型：`gaussian` / `linear` / `polinominal` / `laplace`（注意：字符串拼写是 `polinominal`）。

### 重要实现细节

- **优化变量就是知识数据集 \(S_k\)**：`self.synthetic_images` 是 `requires_grad=True` 的可学习 tensor，优化器直接对它做 SGD：
  - `optimizer_image = SGD([self.synthetic_images], lr=...)`
- **全局模型参数被冻结**：`param.requires_grad = False`，保证梯度只更新 `synthetic_images`。
- **可选 DiffAugment**：当 `args.dsa_strategy != 'None'` 时，会对 real/syn 同 seed 增强（更稳定）。

---

## FedVCK 3.2：知识初始化 + 中间层统计量迁移（BN/Norm 统计约束）

论文里强调两点：

1) **初始化用高斯噪声** \(N(0,1)\) 减少隐私泄漏；  
2) **迁移/固定编码器各层的均值方差统计量**（像“冻结 BN 统计量”那样的约束），让浓缩不只匹配最后一层表征，避免走捷径。

### 代码现状（与论文的对齐/差异）

#### 1) 初始化方式：代码支持三种
- **实现**：`Client.initialization()`（`src/client.py`）
  - `init == 'random_noise'`：保持 `torch.randn(...)` 的随机噪声（这与你文档描述一致）
  - `init == 'real'`：用真实图像直接初始化（默认参数 `config.py` 里 `--init real`）
  - `init == 'real_avg'`：用真实图像的均值图初始化

> **差异点**：论文描述“为避免隐私泄露使用随机噪声初始化”，但这份代码的默认设置是 `real`，实际运行如果不改参数，**会直接把真实图像拷贝进 `synthetic_images`**（隐私假设会不同）。

#### 2) “中间层统计量迁移/固定”在代码中未看到对应实现
在 `src/models.py` 的网络实现里虽然有 BatchNorm/GroupNorm，但 **没有看到**：

- 在嵌入真实 batch 时记录每层 \(\{\mu_l,\sigma_l\}\)；
- 在合成数据前向时替换/固定这些统计量；
- 或类似“把 BN running mean/var 替换为真实统计量并冻结”的操作。

因此，当前代码层面更像是 **“像素空间可学习数据集 + 在 embedding 空间做匹配损失”**，并未显式实现你文档里 3.2 的“统计量迁移约束”那部分。

---

## FedVCK 3.3：面向缺失知识的重要性感知浓缩（Importance-aware condensation）

论文 3.3 的关键是：不要每轮均匀采样真实数据，而要优先“当前模型缺失的知识”（高误差/高损失样本），并用 \(M_t\) 与 \(M_{t-1}\) 自集成平滑误差。

### 代码对应位置

#### 1) 误差/重要性估计（按样本损失）
- **实现类**：`PerLabelDatasetNonIID`（`dataset/data/dataset.py`）
- **核心函数**：`cal_loss(model, prev_model, lamda, gamma, b, ...)`
  - 计算当前模型与上一轮模型对每个样本的预测 `all_preds`、`all_preds_prev`
  - 做 logit 级别的自集成（代码里权重参数叫 `lamda`）：
    - `all_preds = (1-lamda) * all_preds + lamda * all_preds_prev`
  - 逐样本交叉熵（`reduction='none'`）
  - 将损失通过一个 sigmoid 映射到 \( (0,1) \)：
    - `loss_all = 1 / (1 + exp(-gamma*(loss-b)))`

> **对齐说明**：这对应你文档的式(7)“自集成后计算误差”，但符号不同：论文是 \(\alpha\)，代码用 `lamda`；另外论文写的是 `1 + e^{-err+b}` 一类形式，代码是 sigmoid(\(\gamma(\mathrm{loss}-b)\)) 的等价变体（都是把损失单调映射为权重）。

#### 2) 重要性采样分布与采样实现
- **实现**：`pre_sample(it, bs)`
  - 对每个类别 \(c\) 单独计算采样概率：
    - `sample_prob[c] = softmax(loss_all[indices_class[c]])`
  - 然后按该分布预采样 `it*bs` 个索引：
    - `sample_indices[c] = np.random.choice(indices_class[c], p=sample_prob[c], size=it*bs, replace=True)`

#### 3) 在浓缩训练中使用加权采样的真实 batch
- **实现**：`Client.train_weighted_sample()` / `Client.train_weighted_MMD()`
  - 在每次 condensation iteration，真实 batch 由：
    - `real_image = images_all[sample_indices[c][...]]`
  - 从而实现“每轮更偏向高损失/模型不擅长样本”的浓缩。

### 代码里的“每轮动态更新”如何发生

- 在 `Client.recieve_model()` 中：
  - `prev_global_model` 会保存上一轮的全局模型；
  - `global_model` 更新为当前服务端下发的模型；
  - 然后每轮 condensation 前调用 `train_set.cal_loss(copy(global_model), copy(prev_global_model), ...)`。

这使得重要性分布是 **随通信轮更新的**，符合论文 3.3 的“逐轮从已掌握知识转向缺失知识”叙述。

---

## FedVCK 3.4：服务端基于原型的关系监督对比学习

论文 3.4 的关键：

- 每类聚合一个“全局对数几率原型” \(p_c\)，用它找 Top-K 难负类集合 \(HN(c)\)；
- 另用累计知识集 \(S_{\le t}\) 计算每类“特征原型” \(f_c\)；
- 训练时对样本特征做“关系监督的对比学习”（让 \(c\) 远离它的难负类）。

### 代码对应位置与执行顺序（`src/server.py::fit()`）

1) **客户端上传**：
   - 浓缩知识图像 `imgs` 与其标签 `labels`
   - 两种原型：
     - `Client.get_feature_prototype()`：每类真实数据在 embedding 空间的均值（特征原型）
     - `Client.get_logit_prototype()`：每类真实数据的 logit 均值（logit 原型）

2) **服务端聚合原型**：
   - `server_prototypes[c]`：按各 client 的该类样本数加权平均得到全局特征原型
   - `server_logit_prototypes[c]`：同理聚合全局 logit 原型

3) **从 logit 原型找 Top-K“关系类别”**
   - `relation_class = get_mask(server_logit_proto_tensor, k=topk)` 返回每行 top-k 的类别索引
   - 代码里把这个 `relation_class[c]` 当作“与类别 c 相关/需要拉开”的类别集合来用

> **差异点（需注意）**：论文写的是“对数几率（log-odds）原型”并用 \(p_c[j]\) 找难负类。  
> 代码里 `Client.get_logit_prototype()` 计算过 `real_score_c = log(p/(1-p))`，但 **最终并没有把它用于 prototype**；返回的 prototype 仍是 **logit 均值**。  
> 因此，“log-odds 原型”在当前实现里更像是“logit 原型”。

4) **用累计的浓缩知识训练全局模型（交叉熵 + 对比项）**
   - 服务端把每轮收到的浓缩数据追加到 `all_synthetic_data`，形成累计知识集 \(S_{\le t}\)（可截断保留最近若干轮，`preserve_thres`）。
   - 主损失：`CrossEntropy(pred, target)` 在 `synthetic_dataloader` 上训练全局模型。
   - 若 `con_beta > 0` 且 `rounds > 0`：
     - 从模型得到 `features = embed(x)`（`global_model(x, train=True)` 返回 `inter_out, logits`）
     - 计算 `SupervisedContrastiveLoss`（`src/utils.py`），其中 `relation_class` 控制分母里纳入哪些“关系类/难负类”。
     - 当 `contrastive_way == 'supcon_asym_syn'` 时：
       - 正原型来自上一轮的 `prev_syn_proto[target]`（服务端在每轮末用累计知识重新估计并 `normalize`）

---

## 关键配置参数（`config.py` 与论文超参的对应）

下面列出与 FedVCK 3.1–3.4 最相关的参数：

- **浓缩规模**
  - `--ipc`：每类合成样本数（若 `compression_ratio==0`）
  - `--compression_ratio`：按真实类样本数比例决定每类合成数量（client 侧 `ipc_dict`，并设置最小值 `max(5, ceil(n*ratio))`）
- **浓缩优化**
  - `--dc_iterations`：合成数据优化步数
  - `--dc_batch_size`：每步每类真实 batch 大小
  - `--image_lr / --image_momentum / --image_weight_decay`：合成数据优化器超参
  - `--clip_norm`：对 `synthetic_images` 梯度裁剪
  - `--kernel`：MMD 核类型（用于 `weighted_mmd` 路径）
- **重要性感知（3.3）**
  - `--lamda`：当前模型 vs 上一轮模型的自集成权重（代码中混合 logit）
  - `--gamma`：sigmoid 的斜率
  - `--b`：sigmoid 平移项（控制权重数值范围）
- **服务端对比学习（3.4）**
  - `--con_beta`：对比损失权重（为 0 则关闭该模块）
  - `--con_temp`：温度系数
  - `--topk`：每类选择 Top-K 关系/难负类
  - `--lr_head / --weight_decay_head`：投影头（`Projector`）优化超参

---

## 论文符号 vs 代码变量：快速对照表

> 方便你把“代码实现”写进方法/实验设置里（符号以你给的文档为准）。

- **\(S\)**（可学习浓缩知识数据集）
  - 代码：`Client.synthetic_images`（`src/client.py`，优化变量），`Client.synthetic_labels`
- **\(T\)**（真实本地数据）
  - 代码：`PerLabelDatasetNonIID.images_all / labels_all`（`dataset/data/dataset.py`）
- **\(\psi_\theta(\cdot)\)**（共享嵌入/编码器）
  - 代码：`global_model.embed(x)`（`src/models.py` + `src/utils.py::get_model`）
- **\(L_{\text{cond}}\)**（分布匹配损失）
  - 代码：
    - 均值对齐：`sum((mean_real - mean_syn)**2)`（`Client.train_weighted_sample()`）
    - 核化 MMD：`M3DLoss(real_feature_c, syn_feature_c)`（`Client.train_weighted_MMD()`）
- **\(M_t, M_{t-1}\)**（当前/上一轮模型）
  - 代码：`Client.global_model` / `Client.prev_global_model`（`Client.recieve_model()`）
- **\(\alpha\)**（自集成权重，文档式(7)）
  - 代码：`lamda`（注意拼写为 `lamda`，在 `PerLabelDatasetNonIID.cal_loss()` 混合 logit）
- **\(w(x)\)**（样本重要性/采样权重）
  - 代码：
    - 中间量：`loss_all = sigmoid(gamma*(CE-b))`
    - 采样分布：`sample_prob[c] = softmax(loss_all[class_c])`
    - 采样：`np.random.choice(..., p=sample_prob[c])`
- **\(HN(c)\)**（Top-K 难负类集合，文档式(13)）
  - 代码：`relation_class[c] = topk_indices(server_logit_proto_tensor[c])`（`Server.get_mask()`）
- **\(h(\cdot)\)**（投影头）
  - 代码：`SupervisedContrastiveLoss.head`（`src/utils.py`，内部是 `Projector`）


---

## 代码层面的“做了什么工作”总结（按通信轮）

在每一轮 \(t\)（`Server.fit()` 的一次循环）中，代码完成：

- **客户端侧（对每个参与 client）**
  - 拿到 `M_{t-1}`（上一轮）与 `M_t`（当前下发）的模型快照
  - 基于自集成预测的逐样本损失，得到每类的**重要性采样分布**
  - 从随机噪声/真实图像初始化的 `synthetic_images` 出发，优化得到该轮要上传的浓缩知识 \(S_k^t\)
  - 同时计算并上传本地每类的特征原型与 logit 原型

- **服务端侧**
  - 聚合各 client 的原型，推导每个类的 Top-K 关系集合（近似“难负类”）
  - 把本轮收到的浓缩知识追加为累计知识集 \(S_{\le t}\)
  - 用交叉熵 +（可选）关系监督对比学习更新全局模型
  - 重新用累计知识估计 `prev_syn_proto`（供下一轮对比学习使用）
  - 定期在测试集上评估并记录

---

## 与你提供的 FedVCK 文档对齐时，最需要注意的差异点

1) **3.2 的“层级统计量迁移/固定”目前未在代码中体现**  
   代码没有记录/替换每层 \(\mu,\sigma\) 的逻辑，当前浓缩监督主要来自 embedding 的均值差或核化 MMD。

2) **初始化默认不是随机噪声**  
   论文强调用 \(N(0,1)\) 初始化降低隐私泄露；但默认 `--init real` 会用真实图像初始化合成数据。

3) **3.4 的“log-odds 原型”在实现中更接近“logit 原型”**  
   `get_logit_prototype()` 虽计算了 log-odds（`real_score_c`），但未用于返回的 prototype；服务端 Top-K 关系由 logit 原型张量直接计算。

---

## 快速定位：想看“算法关键实现”该从哪几处开始读

- **客户端浓缩主循环**：`src/client.py`
  - `Client.train_weighted_sample()`（均值匹配）
  - `Client.train_weighted_MMD()`（核化 MMD）
  - `Client.initialization()`（知识初始化）
- **重要性采样（3.3 的核心）**：`dataset/data/dataset.py`
  - `PerLabelDatasetNonIID.cal_loss()` / `pre_sample()`
- **服务端聚合 + 对比学习（3.4）**：`src/server.py`
  - `Server.fit()` 中聚合原型、`get_mask()` 得 Top-K 关系、训练 global model
- **核函数/MMD + 对比损失**：`src/utils.py`
  - `M3DLoss`、`SupervisedContrastiveLoss`

