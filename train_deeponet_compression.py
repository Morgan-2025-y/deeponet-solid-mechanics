"""
DeepONet 训练脚本 — 左右压缩工况，只预测主位移 ux
===================================================
算子映射: G : f_left(y) -> u_x(x, y)
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

# ============================================================
# 0. 基础设置
# ============================================================
dde.config.set_default_float("float32")
np.random.seed(42)
torch.manual_seed(42)
os.makedirs("results_compression", exist_ok=True)

# ============================================================
# 1. 加载数据 & 筛选压缩工况
# ============================================================
print("Loading data...")
mat = sio.loadmat('plate_no_hole_planestress.mat', squeeze_me=True)

f_type    = mat['f_type'].astype(int)
f_bc_left = mat['f_bc_left'].astype(np.float32)   # (3000, 101)
coors_list = mat['coors_dict']
ux_list    = mat['final_u']

idx_comp = np.where(f_type == 1)[0]               # 1000 个压缩样本
print(f"压缩样本数: {len(idx_comp)}")

branch_all = f_bc_left[idx_comp, :]               # (1000, 101)
coors_all  = [coors_list[i] for i in idx_comp]
ux_all_raw = [ux_list[i]    for i in idx_comp]

# ============================================================
# 2. 插值到固定均匀网格 (Trunk 查询点)
# ============================================================
print("Interpolating to uniform grid...")
N_GRID = 32
HALF   = 0.25
xq = np.linspace(-HALF, HALF, N_GRID)
yq = np.linspace(-HALF, HALF, N_GRID)
XX, YY    = np.meshgrid(xq, yq)
trunk_pts = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)  # (1024, 2)
N_Q       = trunk_pts.shape[0]

N = len(idx_comp)
ux_grid = np.zeros((N, N_Q), dtype=np.float32)
uy_all_raw = [mat['final_v'][i] for i in idx_comp]
uy_grid    = np.zeros((N, N_Q), dtype=np.float32)
for i in range(N):
    if i % 200 == 0: print(f"  {i}/{N}")
    ux_grid[i] = griddata(
        coors_all[i].astype(np.float64),
        ux_all_raw[i],
        trunk_pts.astype(np.float64),
        method='linear', fill_value=0.0
    ).astype(np.float32)
    uy_grid[i] = griddata(
        coors_all[i].astype(np.float64),
        uy_all_raw[i],
        trunk_pts.astype(np.float64),
        method='linear', fill_value=0.0
    ).astype(np.float32)

print(f"ux 范围: [{ux_grid.min():.4e}, {ux_grid.max():.4e}] m")

# ============================================================
# 3. 标准化
# ============================================================
b_mean, b_std   = branch_all.mean(), branch_all.std() + 1e-12
ux_mean, ux_std = ux_grid.mean(),    ux_grid.std()    + 1e-12

branch_norm = (branch_all - b_mean) / b_std
ux_norm     = (ux_grid    - ux_mean) / ux_std
uy_mean = uy_grid.mean();  uy_std = uy_grid.std() + 1e-12
uy_norm = (uy_grid - uy_mean) / uy_std

np.save("results_compression/norm_params.npy",
        {'b_mean': b_mean, 'b_std': b_std,
         'ux_mean': ux_mean, 'ux_std': ux_std,
         'uy_mean': uy_mean, 'uy_std': uy_std,   # 新增
         'trunk_pts': trunk_pts})

# ============================================================
# 4. 训练 / 测试划分
# ============================================================
perm      = np.random.permutation(N)
b_train   = branch_norm[perm[:800]];   b_test  = branch_norm[perm[800:]]
y_train = np.stack([ux_norm[perm[:800]], uy_norm[perm[:800]]], axis=-1)  # (800, 1024, 2)
y_test  = np.stack([ux_norm[perm[800:]], uy_norm[perm[800:]]], axis=-1)  # (200, 1024, 2)
print(f"Train: {b_train.shape}, Test: {b_test.shape}")

# ============================================================
# 5. 构建 DeepONet (Cartesian Product 格式)
# ============================================================
P = 128   # Branch / Trunk 最后一层宽度 (内积维度)

data = dde.data.TripleCartesianProd(
    X_train=(b_train, trunk_pts),
    y_train=y_train,   # (800, 1024, 2)
    X_test=(b_test,  trunk_pts),
    y_test=y_test      # (200, 1024, 2)
)

net = dde.nn.DeepONetCartesianProd(
    layer_sizes_branch=[101, 256, 256, 256, P * 2],   # 256 → split_both 后每输出 128 维
    layer_sizes_trunk =[  2, 256, 256, 256, P * 2],
    activation="tanh",
    kernel_initializer="Glorot normal",
    num_outputs=2,
    multi_output_strategy="split_both",
)

model = dde.Model(data, net)

# ============================================================
# 6. 训练: Adam → L-BFGS 两阶段
# ============================================================
print("\n--- Stage 1: Adam (40000 iters) ---")
model.compile("adam", lr=1e-3,
              metrics=["mean l2 relative error"],
              decay=("inverse time", 10000, 0.5))
losshistory, trainstate = model.train(
    iterations=40000,
    batch_size=None,          # CartesianProd 全批次
    display_every=2000,
    model_save_path="results_compression/model_ux"
)

print("\n--- Stage 2: L-BFGS fine-tuning ---")
model.compile("L-BFGS", metrics=["mean l2 relative error"])
losshistory, trainstate = model.train(
    display_every=500,
    model_save_path="results_compression/model_ux_final"
)

# ============================================================
# 7. 评估 & 可视化
# ============================================================
pred_norm = model.predict((b_test, trunk_pts))     # (200, 1024, 2)
ux_pred   = pred_norm[..., 0] * ux_std + ux_mean
uy_pred   = pred_norm[..., 1] * uy_std + uy_mean

ux_true = y_test[..., 0] * ux_std + ux_mean       # ← ux_test 改为 y_test[...,0]
uy_true = y_test[..., 1] * uy_std + uy_mean       # ← 新增这行

ux_errors = (np.linalg.norm(ux_pred - ux_true, axis=1) /
             (np.linalg.norm(ux_true, axis=1) + 1e-12))
uy_errors = (np.linalg.norm(uy_pred - uy_true, axis=1) /
             (np.linalg.norm(uy_true, axis=1) + 1e-12))
print(f"ux — Mean: {ux_errors.mean():.4f} | Std: {ux_errors.std():.4f} | Max: {ux_errors.max():.4f}")
print(f"uy — Mean: {uy_errors.mean():.4f} | Std: {uy_errors.std():.4f} | Max: {uy_errors.max():.4f}")

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle('Compression DeepONet — u_x and u_y Prediction vs FEM')
kw = dict(cmap='RdBu_r', origin='lower', extent=[-HALF, HALF, -HALF, HALF])

# 第一行：ux
pred_2d = ux_pred[0].reshape(N_GRID, N_GRID)
true_2d = ux_true[0].reshape(N_GRID, N_GRID)
for ax, data, title in zip(axes[0],
    [true_2d, pred_2d, np.abs(pred_2d - true_2d)],
    ['FEM $u_x$', 'DeepONet $u_x$', '|Error| $u_x$']):
    im = ax.imshow(data, **kw)
    ax.set_title(title); plt.colorbar(im, ax=ax, fraction=0.046)

# 第二行：uy
pred_2d = uy_pred[0].reshape(N_GRID, N_GRID)
true_2d = uy_true[0].reshape(N_GRID, N_GRID)
for ax, data, title in zip(axes[1],
    [true_2d, pred_2d, np.abs(pred_2d - true_2d)],
    ['FEM $u_y$', 'DeepONet $u_y$', '|Error| $u_y$']):
    im = ax.imshow(data, **kw)
    ax.set_title(title); plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('results_compression/validation_ux_uy.png', dpi=150, bbox_inches='tight')