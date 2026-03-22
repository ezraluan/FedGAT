import argparse

parser = argparse.ArgumentParser()

# parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--seed", type=int, default=19260817)
parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument("--dataset_root", type=str, default="./dataset/torchvision")
parser.add_argument("--split_file", type=str, default="")
parser.add_argument("--dataset", type=str, default='CIFAR10')
parser.add_argument("--client_num", type=int, default=10)
parser.add_argument("--partition", type=str, default='dirichlet')
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument('--num_classes_per_client', type=int, default=2, help="label_split") 

parser.add_argument("--model", type=str, default="ConvNet")
parser.add_argument("--communication_rounds", type=int, default=10)
parser.add_argument("--join_ratio", type=float, default=1.0)
parser.add_argument("--lr_server", type=float, default=0.01)
parser.add_argument("--momentum_server", type=float, default=0.9)
parser.add_argument("--weight_decay_server", type=float, default=0)

parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--model_epochs", type=int, default=1000)
parser.add_argument("--local_ep", type=int, default=40)
parser.add_argument("--ipc", type=int, default=10)
parser.add_argument("--compression_ratio", type=float, default=0.)
parser.add_argument("--dc_iterations", type=int, default=1000)
parser.add_argument("--dc_batch_size", type=int, default=256)
parser.add_argument("--image_lr", type=float, default=1.0)
parser.add_argument("--image_momentum", type=float, default=0.5)
parser.add_argument("--image_weight_decay", type=float, default=0)
parser.add_argument("--init", type=str, default='real')
parser.add_argument("--clip_norm", type=float, default=30)
parser.add_argument("--weighted_matching", action='store_true', default=False)
parser.add_argument("--weighted_sample", action='store_true', default=False)
parser.add_argument("--weighted_mmd", action='store_true', default=False)
parser.add_argument("--contrastive_way", type=str, default='supcon_asym_syn', choices=['supcon_asym', 'supcon_asym_syn', 'supcon_relation'])
parser.add_argument("--con_beta", type=float, default=0.)
parser.add_argument("--con_temp", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=3)
parser.add_argument("--lr_head", type=float, default=0.01)
parser.add_argument("--momentum_head", type=float, default=0.9)
parser.add_argument("--weight_decay_head", type=float, default=0)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--lamda", type=float, default=0.5)
parser.add_argument("--b", type=float, default=0.7)
parser.add_argument("--kernel", type=str, default='linear')

# ---------------------------
# DC: OGCA (your modification)
# ---------------------------
# 说明：浓缩阶段的 cross-term 使用 Sinkhorn OT 学到的 Γ* 做加权（式(7)–(16)）
parser.add_argument("--ogca_eps", type=float, default=0.05, help="entropy regularization epsilon for Sinkhorn OT")
parser.add_argument("--ogca_iters", type=int, default=30, help="Sinkhorn iterations")
parser.add_argument(
    "--ogca_sigmas",
    type=str,
    default="0.5,1,2,4",
    help="comma-separated gaussian kernel scales for OGCA similarity/kernel",
)

parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.5)
parser.add_argument("--weight_decay", type=float, default=0)

parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
parser.add_argument("--preserve_all", action='store_true', default=False)

parser.add_argument("--eval_gap", type=int, default=1)

parser.add_argument("--tag", type=str, default='0')
parser.add_argument("--save_root_path", type=str, default='../results/')