import copy
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Subset

from src.client import Client
from src.server import Server
from config import parser
from dataset.data.dataset import get_dataset, PerLabelDatasetNonIID
from src.utils import setup_seed, get_model, ParamDiffAug
import logging

def get_n_params(model):
    """
    统计模型参数量（parameter count）。

    这里按 tensor 元素总数累计，不区分可训练/不可训练参数。
    """
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def main():
    # -----------------------------
    # 0) 解析命令行参数 / 实验配置
    # -----------------------------
    # 如需手动限制可见 GPU，可在这里设置（当前代码默认不启用）
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    args = parser.parse_args()

    # DiffAugment（可微分数据增强）相关配置：
    # - args.dsa_param：增强用到的参数容器
    # - args.dsa：根据 dsa_strategy 是否为 None 来开关增强
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy == 'None' else True

    
    # --------------------------------------------------------
    # 1) 根据 non-IID 划分方式，确定数据划分文件与实验保存路径
    # --------------------------------------------------------
    # split_file：记录每个客户端包含的样本索引（client_idx）与类别集合（client_classes）
    # model_identification：本次实验的唯一标识，用于组织 results/ 目录与日志
    if args.partition == 'dirichlet':
        split_file = f'/{args.dataset}_client_num={args.client_num}_alpha={args.alpha}.json'
        args.split_file = os.path.join(os.path.dirname(__file__), "dataset/split_file"+split_file)
        if args.compression_ratio > 0.:
            # 按压缩率决定每类合成样本数（每个客户端内部会按各类真实样本数计算 ipc_dict）
            model_identification = f'{args.dataset}_alpha{args.alpha}_{args.client_num}clients/{args.model}_{100*args.compression_ratio}%_{args.dc_iterations}dc_{args.model_epochs}epochs_{args.tag}'
        else:
            # 直接指定 ipc（images per class）
            model_identification = f'{args.dataset}_alpha{args.alpha}_{args.client_num}clients/{args.model}_{args.ipc}ipc_{args.dc_iterations}dc_{args.model_epochs}epochs_{args.tag}'
            # raise Exception('Compression ratio should > 0')
    elif args.partition == 'label':
        split_file = f'/{args.dataset}_client_num={args.client_num}_label={args.num_classes_per_client}.json'
        args.split_file = os.path.join(os.path.dirname(__file__), "dataset/split_file"+split_file)
        if args.compression_ratio > 0.:
            model_identification = f'{args.dataset}_label{args.num_classes_per_client}_{args.client_num}clients/{args.model}_{100*args.compression_ratio}%_{args.dc_iterations}dc_{args.model_epochs}epochs_{args.tag}'
        else:
            raise Exception('Compression ratio should > 0')
    elif args.partition == 'pathological':
        split_file = f'/{args.dataset}_client_num={args.client_num}_pathological={args.num_classes_per_client}.json'
        args.split_file = os.path.join(os.path.dirname(__file__), "dataset/split_file"+split_file)
        if args.compression_ratio > 0.:
            model_identification = f'{args.dataset}_pathological{args.num_classes_per_client}_{args.client_num}clients/{args.model}_{100*args.compression_ratio}%_{args.dc_iterations}dc_{args.model_epochs}epochs_{args.tag}'
        else:
            raise Exception('Compression ratio should > 0')

    # -----------------------------
    # 2) 日志与结果保存目录初始化
    # -----------------------------
    args.save_root_path = os.path.join(os.path.dirname(__file__), 'results/')
    args.save_root_path = os.path.join(args.save_root_path, model_identification)
    os.makedirs(args.save_root_path, exist_ok=True)
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    log_file = 'log.txt'
    log_path = os.path.join(args.save_root_path, log_file)
    print(log_path)
    if os.path.exists(log_path):
        raise Exception('log file already exists!')
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # -----------------------------
    # 3) 复现性与设备设置
    # -----------------------------
    setup_seed(args.seed)
    device = torch.device(args.device)
    # 显式设置当前进程使用哪块 GPU（args.device 形如 "cuda:0"）
    torch.cuda.set_device(device)

    # --------------------------------------------------------
    # 4) 加载数据集 + 加载 non-IID 客户端划分（split_file）
    # --------------------------------------------------------
    # get_dataset 会返回：
    # - dataset_info：数据集元信息（通道数、图像尺寸、类别数、均值方差等）
    # - train_set / test_set：原始训练/测试集对象
    # - test_loader：测试集 DataLoader（服务端评估用）
    dataset_info, train_set, test_set, test_loader = get_dataset(args.dataset, args.dataset_root, args.batch_size)
    print("load data: done")

    # split_file 由 dataset/data/dataset_partition.py 生成，包含：
    # - client_idx：每个客户端的样本索引列表
    # - client_classes：每个客户端“有效类别集合”（通常要求该类样本数 >= 某阈值）
    with open(args.split_file, 'r') as file:
        file_data = json.load(file)
    client_indices, client_classes = file_data['client_idx'], file_data['client_classes']

    # --------------------------------------------------------
    # 5) 统计每个客户端的类别分布（仅用于日志观察）
    # --------------------------------------------------------
    # 不同数据集存 label 的字段不一样，这里统一取出 labels 便于统计
    if args.dataset in ['CIFAR10', 'FMNIST',]:
        labels = np.array(train_set.targets, dtype='int64')
    elif args.dataset in ['PathMNIST', 'OCTMNIST', 'OrganSMNIST', 'OrganCMNIST', 'ImageNette', 'OrganCMNIST224', 'PneumoniaMNIST224', 'RetinaMNIST224', 'STL', 'STL32']:
        labels = train_set.labels
    net_cls_counts = {}
    dict_users = {i: idcs for i, idcs in enumerate(client_indices)}
    for net_i, dataidx in dict_users.items():
        unq, unq_cnt = np.unique(labels[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logging.info(f'Data statistics: {net_cls_counts}')
    logging.info(f'client classes: {client_classes}')
    
    # 将全局训练集按客户端索引切分成多个 Subset
    train_sets = [Subset(train_set, indices) for indices in client_indices]

    # --------------------------------------------------------
    # 6) 初始化全局模型（服务端持有并下发给客户端）
    # --------------------------------------------------------
    # 注意：在本项目里，客户端进行“知识浓缩”时会：
    # - 冻结全局模型参数
    # - 只优化 synthetic_images（可学习小数据集）
    # - 使用 global_model.embed(·) 得到潜在表征并做分布匹配（均值匹配或 MMD）
    global_model = get_model(args.model, dataset_info)
    logging.info(global_model)
    logging.info(get_n_params(global_model))
    logging.info(args.__dict__)

    # --------------------------------------------------------
    # 7) 初始化客户端列表（Client）与服务端（Server）
    # --------------------------------------------------------
    # 每个 Client 拥有自己的本地数据（按类组织）与可学习的 synthetic_images：
    # - PerLabelDatasetNonIID 会把 Subset 转为 tensor，并建立 indices_class 等索引结构
    # - 后续“重要性感知采样”（FedVCK 3.3）会在该对象内部计算 loss_all，并生成 sample_indices
    client_list = [Client(
        cid=i,
        train_set=PerLabelDatasetNonIID(
            train_sets[i],
            client_classes[i],
            dataset_info['channel'],
            device,
        ),
        classes=client_classes[i],
        dataset_info=dataset_info,
        ipc=args.ipc,
        compression_ratio=args.compression_ratio,
        dc_iterations=args.dc_iterations,
        real_batch_size=args.dc_batch_size,
        image_lr=args.image_lr,
        image_momentum=args.image_momentum,
        image_weight_decay=args.image_weight_decay,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        local_ep=args.local_ep,
        dsa=args.dsa,
        dsa_strategy=args.dsa_strategy,
        # synthetic_images 初始化策略：
        # - 'random_noise'：保持 N(0,1) 噪声（更贴近你给的文档 3.2 隐私友好设定）
        # - 'real' / 'real_avg'：用真实图像或其均值初始化（更易收敛但隐私假设更弱）
        init=args.init,
        # 梯度裁剪：对 synthetic_images 的梯度做 clip，避免优化不稳定
        clip_norm=args.clip_norm,
        # 重要性感知采样（FedVCK 3.3）：
        # - gamma, b：把 per-sample 交叉熵 loss 经 sigmoid 映射为权重时的形状控制参数
        # - lamda：当前模型与上一轮模型的自集成系数（代码里拼写为 lamda）
        gamma=args.gamma,
        lamda=args.lamda,
        b=args.b,
        # 服务端对比学习温度（客户端侧通常仅透传/记录）
        con_temp=args.con_temp,
        # MMD 核类型（当走 weighted_mmd 路径时使用）
        kernel=args.kernel,
        # 你的修改：OGCA（严格使用）
        ogca_eps=args.ogca_eps,
        ogca_iters=args.ogca_iters,
        ogca_sigmas=args.ogca_sigmas,
        save_root_path=args.save_root_path,
        device=device,
    ) for i in range(args.client_num)]

    # 服务端也包装一个 PerLabelDatasetNonIID（覆盖全类别），用于一些全局统计/可扩展用途
    # （当前主要评估使用 test_loader；训练使用客户端上传的 synthetic 数据）
    server = Server(
        train_set = PerLabelDatasetNonIID(
            train_set,
            range(0,dataset_info['num_classes']),
            dataset_info['channel'],
            device,
        ),
        ipc = args.ipc,
        dataset_info=dataset_info,
        global_model_name=args.model,
        global_model=global_model,
        clients=client_list,
        # 联邦训练轮数与每轮参与比例（join_ratio<1 则随机抽 client）
        communication_rounds=args.communication_rounds,
        join_ratio=args.join_ratio,
        batch_size=args.batch_size,
        # 服务端用累计 synthetic 数据更新全局模型的训练 epoch 数
        model_epochs=args.model_epochs,
        lr_server=args.lr_server,
        momentum_server=args.momentum_server,
        weight_decay_server=args.weight_decay_server,
        # 对比学习投影头（Projector/head）的优化超参
        lr_head=args.lr_head,
        momentum_head=args.momentum_head, 
        weight_decay_head=args.weight_decay_head,
        # 下面三个开关决定客户端“浓缩损失/采样策略”走哪条分支：
        # - weighted_sample：按类均值匹配（embedding 均值差）+ 重要性感知采样
        # - weighted_mmd：核化 MMD + 重要性感知采样
        # - weighted_matching：预留开关（具体逻辑在 server/client 内部使用方式决定）
        weighted_matching = args.weighted_matching,
        weighted_sample = args.weighted_sample,
        weighted_mmd = args.weighted_mmd,
        # 服务端关系监督对比学习（FedVCK 3.4）配置
        contrastive_way = args.contrastive_way,
        con_beta = args.con_beta,
        con_temp = args.con_temp,
        topk = args.topk,
        # DiffAugment（服务端训练全局模型时也可对 synthetic data 做增强）
        dsa = args.dsa,
        dsa_strategy = args.dsa_strategy,
        # 是否保留所有历史轮的 synthetic 数据；否则只保留最近若干轮（节省内存）
        preserve_all = args.preserve_all,
        # 测试评估间隔（round 为单位）
        eval_gap=args.eval_gap,
        test_set=test_set,
        test_loader=test_loader,
        device=device,
        model_identification=model_identification,
        save_root_path=args.save_root_path
    )
    print('Server and Clients have been created.')

    # -----------------------------
    # 8) 启动联邦训练（主循环在 Server.fit()）
    # -----------------------------
    # 在每一轮中，通常会发生：
    # - 选客户端 -> 下发全局模型 -> 客户端进行“浓缩”（优化 synthetic_images 并上传）
    # - 服务端聚合原型/关系集合 -> 用累计 synthetic 数据更新全局模型
    # - 在 test_loader 上评估
    server.fit()

if __name__ == "__main__":
    main()