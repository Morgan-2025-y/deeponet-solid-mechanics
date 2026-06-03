"""
DeepONet 压缩工况预测脚本（支持新生成数据）
============================================
"""

import numpy as np
import scipy.io as sio
from scipy.interpolate import griddata
import deepxde as dde
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

dde.config.set_default_float("float32")
os.makedirs("results_compression/pred_output", exist_ok=True)

# ============================================================
# ★ 配置区 ★
# ============================================================
# 数据来源:
#   "original" : 用原始训练数据 plate_no_hole_planestress.mat
#   "new"      : 用 MATLAB 新生成的 new_compression_data.mat
DATA_SOURCE = "new"

# 预测哪些样本 (仅 DATA_SOURCE="original" 时有效):
#   "all"    : 全部 1000 个压缩样本
#   "test"   : 测试集 200 个
#   "custom" : 手动指定 CUSTOM_IDS
PRED_MODE  = "all"
CUSTOM_IDS = [0, 5, 10, 50, 100]

# 模型权重文件（改成你实际的步数）
CKPT_PATH = "results_compression/model_ux_final-53820.pt"

# ============================================================
# 网络结构（必须与训练完全一致）
# ============================================================
def build_model():
    P = 128
    net = dde.nn.DeepONetCartesianProd(
        layer_sizes_branch=[101, 256, 256, 256, P],
        layer_sizes_trunk =[  2, 256, 256, 256, P],
        activation="tanh",
        kernel_initializer="Glorot normal",
    )
    return net

# ============================================================
# Step 1: 加载归一化参数
# ============================================================
print("Step 1: Loading normalization params...")
norm      = np.load("results_compression/norm_params.npy", allow_pickle=True).item()
b_mean    = float(norm['b_mean']);   b_std  = float(norm['b_std'])
ux_mean   = float(norm['ux_mean']); ux_std = float(norm['ux_std'])
trunk_pts = norm['trunk_pts'].astype(np.float32)
N_GRID    = 32;  HALF = 0.25

# ============================================================
# Step 2: 加载模型权重
# ============================================================
print("Step 2: Loading model weights...")
dummy_b = np.zeros((1, 101), dtype=np.float32)
dummy_y = np.zeros((1, trunk_pts.shape[0]), dtype=np.float32)
data    = dde.data.TripleCartesianProd(
    X_train=(dummy_b, trunk_pts), y_train=dummy_y,
    X_test =(dummy_b, trunk_pts), y_test =dummy_y,
)
net   = build_model()
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

checkpoint = torch.load(CKPT_PATH, map_location="cpu")
net.load_state_dict(checkpoint["model_state_dict"]
                    if "model_state_dict" in checkpoint else checkpoint)
net.eval()
print("  Model loaded OK.")

# ============================================================
# Step 3: 加载数据
# ============================================================
print(f"\nStep 3: Loading data (source={DATA_SOURCE})...")

if DATA_SOURCE == "new":
    mat        = sio.loadmat('new_compression_data.mat', squeeze_me=True)
    f_bc_left  = mat['f_bc_left'].astype(np.float32)
    coors_list = mat['coors_dict']
    ux_list    = mat['final_u']
    pred_ids   = list(range(len(ux_list)))   # 全部新样本都预测
    print(f"  新数据: {len(pred_ids)} 个样本")

else:   # original
    mat        = sio.loadmat('plate_no_hole_planestress.mat', squeeze_me=True)
    f_type     = mat['f_type'].astype(int)
    f_bc_left  = mat['f_bc_left'].astype(np.float32)
    coors_list = mat['coors_dict']
    ux_list    = mat['final_u']
    idx_comp   = np.where(f_type == 1)[0]

    if PRED_MODE == "all":
        pred_ids = list(range(len(idx_comp)))
    elif PRED_MODE == "test":
        np.random.seed(42)
        perm     = np.random.permutation(len(idx_comp))
        pred_ids = list(perm[800:])
    else:
        pred_ids = CUSTOM_IDS

    # 映射回全局索引
    pred_ids   = [idx_comp[i] for i in pred_ids]
    print(f"  原始数据: 预测 {len(pred_ids)} 个压缩样本")

N_PRED = len(pred_ids)

# ============================================================
# Step 4: 插值 + 批量预测
# ============================================================
print(f"\nStep 4: Interpolating & predicting {N_PRED} samples...")
N_Q          = trunk_pts.shape[0]
branch_batch = np.zeros((N_PRED, 101), dtype=np.float32)
ux_true_all  = np.zeros((N_PRED, N_Q), dtype=np.float32)

for k, sid in enumerate(pred_ids):
    if k % 50 == 0: print(f"  {k}/{N_PRED}")
    branch_batch[k] = f_bc_left[sid]
    ux_true_all[k]  = griddata(
        coors_list[sid].astype(np.float64),
        ux_list[sid],
        trunk_pts.astype(np.float64),
        method='linear', fill_value=0.0
    ).astype(np.float32)

branch_norm  = (branch_batch - b_mean) / b_std
ux_pred_norm = model.predict((branch_norm, trunk_pts))
ux_pred_all  = ux_pred_norm * ux_std + ux_mean

# ============================================================
# Step 5: 误差统计
# ============================================================
errors = (np.linalg.norm(ux_pred_all - ux_true_all, axis=1) /
          (np.linalg.norm(ux_true_all, axis=1) + 1e-12))

print("\n" + "=" * 45)
print(f"  误差统计 ({N_PRED} 个样本, 数据源: {DATA_SOURCE})")
print("=" * 45)
print(f"  Mean   : {errors.mean()*100:.3f}%")
print(f"  Std    : {errors.std()*100:.3f}%")
print(f"  Median : {np.median(errors)*100:.3f}%")
print(f"  Min    : {errors.min()*100:.3f}%")
print(f"  Max    : {errors.max()*100:.3f}%")
print(f"  误差<1%  : {(errors<0.01).sum()}/{N_PRED} ({(errors<0.01).mean()*100:.1f}%)")

# ============================================================
# Step 6: 可视化 — 只生成 3 张图
#   图1: 误差分布直方图
#   图2: 最佳样本对比 (误差最小)
#   图3: 最差样本对比 (误差最大)
# ============================================================
print("\nStep 6: Saving 3 plots...")
kw = dict(origin='lower', cmap='RdBu_r', extent=[-HALF, HALF, -HALF, HALF])

def plot_comparison(k, sid, tag):
    """画单个样本的 FEM vs DeepONet 对比图"""
    pred_2d = ux_pred_all[k].reshape(N_GRID, N_GRID)
    true_2d = ux_true_all[k].reshape(N_GRID, N_GRID)
    err_2d  = np.abs(pred_2d - true_2d)
    err_val = errors[k]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f'[{tag}]  Sample #{sid}  |  Rel L2 = {err_val*100:.3f}%', fontsize=11)

    vmin = min(true_2d.min(), pred_2d.min())
    vmax = max(true_2d.max(), pred_2d.max())
    for ax, data, title in zip(axes,
            [true_2d, pred_2d, err_2d],
            ['FEM $u_x$ (m)', 'DeepONet $u_x$ (m)', '|Error| (m)']):
        v0, v1 = (vmin, vmax) if 'Error' not in title else (0, err_2d.max())
        im = ax.imshow(data, **kw, vmin=v0, vmax=v1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fname = f'results_compression/pred_output/{tag}_sample{sid}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

# 图1: 误差直方图
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(errors * 100, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
ax.axvline(errors.mean()*100,    color='red',    linestyle='--',
           label=f'Mean = {errors.mean()*100:.2f}%')
ax.axvline(np.median(errors)*100, color='orange', linestyle='--',
           label=f'Median = {np.median(errors)*100:.2f}%')
ax.set_xlabel('Relative L2 Error (%)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Error Distribution — {N_PRED} samples ({DATA_SOURCE})', fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig('results_compression/pred_output/error_histogram.png', dpi=150)
plt.close()
print("  Saved: error_histogram.png")

# 图2: 最佳样本
best_k = np.argmin(errors)
plot_comparison(best_k, pred_ids[best_k], "BEST")

# 图3: 最差样本
worst_k = np.argmax(errors)
plot_comparison(worst_k, pred_ids[worst_k], "WORST")

print("\n=== Done. 3 plots saved to results_compression/pred_output/ ===")
