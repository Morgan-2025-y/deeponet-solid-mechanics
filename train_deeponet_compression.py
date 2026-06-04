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
    layer_sizes_branch=[101, 256, 256, 256, P],   # 输入: 101 个传感器
    layer_sizes_trunk =[  2, 256, 256, 256, P],   # 输入: (x, y) 坐标
    activation="tanh",
    kernel_initializer="Glorot normal",
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
ux_pred_norm = model.predict((b_test, trunk_pts))   # (200, 1024)
ux_pred = ux_pred_norm * ux_std + ux_mean
ux_true = ux_test      * ux_std + ux_mean

rel_err = (np.linalg.norm(ux_pred - ux_true, axis=1) /
           (np.linalg.norm(ux_true, axis=1) + 1e-12))
print(f"\n测试集相对 L2 误差 — Mean: {rel_err.mean():.4f} | Std: {rel_err.std():.4f} | Max: {rel_err.max():.4f}")

# 画第一个测试样本的对比图
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle('Compression DeepONet — u_x Prediction vs FEM')
kw = dict(cmap='RdBu_r', origin='lower', extent=[-HALF, HALF, -HALF, HALF])

pred_2d = ux_pred[0].reshape(N_GRID, N_GRID)
true_2d = ux_true[0].reshape(N_GRID, N_GRID)
err_2d  = np.abs(pred_2d - true_2d)

vmin, vmax = true_2d.min(), true_2d.max()
for ax, data, title in zip(axes,
                            [true_2d, pred_2d, err_2d],
                            ['FEM $u_x$', 'DeepONet $u_x$', '|Error|']):
    v0, v1 = (vmin, vmax) if title != '|Error|' else (0, err_2d.max())
    im = ax.imshow(data, **kw, vmin=v0, vmax=v1)
    ax.set_title(title); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('results_compression/validation_ux.png', dpi=150, bbox_inches='tight')
print("Saved: results_compression/validation_ux.png")