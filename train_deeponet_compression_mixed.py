"""
DeepONet 训练脚本 — 单分量显式混合输出 (Ablation Study)
===================================================
算子映射: G : (f_left(y), ν, E) -> u_i(x, y, i)
其中 i=0 为 u_x, i=1 为 u_y。
网络退化为单输出，通过指示器 i 来区分输出物理量。
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
os.makedirs("results_compression_mixed", exist_ok=True) # 更改了保存文件夹，防止覆盖

# ============================================================
# 1. 加载数据 & 筛选压缩工况 (Type 1)
# ============================================================
print("Loading data...")
mat = sio.loadmat('plate_no_hole_planestress.mat', squeeze_me=True)

f_type    = mat['f_type'].astype(int)
f_bc_left = mat['f_bc_left'].astype(np.float32)
f_nu      = mat['f_nu'].astype(np.float32)
f_young   = mat['f_young'].astype(np.float32)

coors_list = mat['coors_dict']
ux_list    = mat['final_u']

idx_comp = np.where(f_type == 1)[0]
print(f"压缩样本数: {len(idx_comp)}")

branch_all = f_bc_left[idx_comp, :]             # (1000, 101)
nu_all     = f_nu[idx_comp]                     # (1000,)
young_all  = f_young[idx_comp]                  # (1000,)

coors_all  = [coors_list[i] for i in idx_comp]
ux_all_raw = [ux_list[i]    for i in idx_comp]
uy_all_raw = [mat['final_v'][i] for i in idx_comp]

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
uy_grid = np.zeros((N, N_Q), dtype=np.float32)

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

# ============================================================
# 3. 数据标准化 & 特征拼接
# ============================================================
b_mean, b_std = branch_all.mean(), branch_all.std() + 1e-12
ux_mean, ux_std = ux_grid.mean(), ux_grid.std() + 1e-12
uy_mean, uy_std = uy_grid.mean(), uy_grid.std() + 1e-12

nu_mean, nu_std       = nu_all.mean(), nu_all.std() + 1e-12
young_mean, young_std = young_all.mean(), young_all.std() + 1e-12

branch_norm = (branch_all - b_mean) / b_std
ux_norm     = (ux_grid    - ux_mean) / ux_std
uy_norm     = (uy_grid    - uy_mean) / uy_std

nu_norm    = (nu_all    - nu_mean)    / nu_std
young_norm = (young_all - young_mean) / young_std

branch_aug = np.concatenate([
    branch_norm,
    nu_norm[:, None],
    young_norm[:, None]
], axis=1).astype(np.float32) # (1000, 103)

# ------------------------------------------------------------
# 【核心修改区 1】：构建含有指示器的 Trunk 查询点和目标标签
# ------------------------------------------------------------
# 给 x, y 坐标后加一维指示器：0 代表 ux，1 代表 uy
trunk_pts_ux = np.column_stack([trunk_pts, np.zeros((N_Q, 1))]).astype(np.float32) # (1024, 3)
trunk_pts_uy = np.column_stack([trunk_pts, np.ones((N_Q, 1))]).astype(np.float32)  # (1024, 3)

# 垂直拼接，此时 Trunk 输入总长度翻倍为 2048，维度为 3
trunk_pts_mixed = np.vstack([trunk_pts_ux, trunk_pts_uy]) # (2048, 3)

# 对应的目标标签展平拼接，ux在前，uy在后
y_mixed = np.concatenate([ux_norm, uy_norm], axis=1) # (1000, 2048)
y_mixed = np.concatenate([ux_norm, uy_norm], axis=1) # 保持 (1000, 2048)
# y_mixed = y_mixed[..., np.newaxis]                   # (1000, 2048, 1)

# 保存所有归一化参数
np.save("results_compression_mixed/norm_params.npy",
        {'b_mean': b_mean, 'b_std': b_std,
         'ux_mean': ux_mean, 'ux_std': ux_std,
         'uy_mean': uy_mean, 'uy_std': uy_std,
         'nu_mean': nu_mean, 'nu_std': nu_std,
         'young_mean': young_mean, 'young_std': young_std,
         'trunk_pts': trunk_pts})

# ============================================================
# 4. 训练 / 测试划分 (使用新构建的 mixed 数据)
# ============================================================
perm    = np.random.permutation(N)
b_train = branch_aug[perm[:800]]
b_test  = branch_aug[perm[800:]]

y_train_mixed = y_mixed[perm[:800]]
y_test_mixed  = y_mixed[perm[800:]]

print(f"Train Branch: {b_train.shape}, Test Branch: {b_test.shape}")
print(f"Trunk Points Mixed: {trunk_pts_mixed.shape}")
print(f"Train Targets Mixed: {y_train_mixed.shape}")

# ============================================================
# 5. 构建 DeepONet (退化为单分量输出)
# ============================================================
P = 128

data = dde.data.TripleCartesianProd(
    X_train=(b_train, trunk_pts_mixed),
    y_train=y_train_mixed,
    X_test=(b_test,  trunk_pts_mixed),
    y_test=y_test_mixed
)

# ------------------------------------------------------------
# 【核心修改区 2】：网络架构退化为 1 输出
# ------------------------------------------------------------
net = dde.nn.DeepONetCartesianProd(
    layer_sizes_branch=[103, 256, 256, 256, P],   # 输出维度改回 P，不再是 P*2
    layer_sizes_trunk =[  3, 256, 256, 256, P],   # 输入维度改为 3 (x, y, index)
    activation="tanh",
    kernel_initializer="Glorot normal",
    num_outputs=1,                                # 显式单输出
    multi_output_strategy=None,                   # 不使用分割策略
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
    batch_size=None,
    display_every=2000,
    model_save_path="results_compression_mixed/model_mixed_103"
)

print("\n--- Stage 2: L-BFGS fine-tuning ---")
model.compile("L-BFGS", metrics=["mean l2 relative error"])
losshistory, trainstate = model.train(
    display_every=500,
    model_save_path="results_compression_mixed/model_mixed_final_103"
)

# ============================================================
# 7. 评估 & 可视化
# ============================================================
# 预测出的 pred_mixed 包含了 ux 和 uy 的归一化值，需解包
pred_mixed = model.predict((b_test, trunk_pts_mixed))
# 将维度 reshape 为 (N_test, 2048)
pred_mixed = pred_mixed.reshape(b_test.shape[0], 2 * N_Q)

# ------------------------------------------------------------
# 【核心修改区 3】：结果解包与反归一化
# ------------------------------------------------------------
ux_pred_norm = pred_mixed[:, :N_Q]   # 前 1024 个点是 ux
uy_pred_norm = pred_mixed[:, N_Q:]   # 后 1024 个点是 uy

ux_pred = ux_pred_norm * ux_std + ux_mean
uy_pred = uy_pred_norm * uy_std + uy_mean

ux_true = y_test_mixed[:, :N_Q,] * ux_std + ux_mean
uy_true = y_test_mixed[:, N_Q:,] * uy_std + uy_mean

ux_errors = (np.linalg.norm(ux_pred - ux_true, axis=1) /
             (np.linalg.norm(ux_true, axis=1) + 1e-12))
uy_errors = (np.linalg.norm(uy_pred - uy_true, axis=1) /
             (np.linalg.norm(uy_true, axis=1) + 1e-12))

print(f"ux — Mean: {ux_errors.mean():.4f} | Std: {ux_errors.std():.4f} | Max: {ux_errors.max():.4f}")
print(f"uy — Mean: {uy_errors.mean():.4f} | Std: {uy_errors.std():.4f} | Max: {uy_errors.max():.4f}")

# 以下绘图部分保持不变...
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle('Compression DeepONet — Parametric $E, \\nu$ vs FEM (Mixed Single Output)')
kw = dict(cmap='RdBu_r', origin='lower', extent=[-HALF, HALF, -HALF, HALF])

# ux
pred_2d = ux_pred[0].reshape(N_GRID, N_GRID)
true_2d = ux_true[0].reshape(N_GRID, N_GRID)
for ax, data, title in zip(axes[0],
    [true_2d, pred_2d, np.abs(pred_2d - true_2d)],
    ['FEM $u_x$', 'DeepONet $u_x$', '|Error| $u_x$']):
    im = ax.imshow(data, **kw)
    ax.set_title(title); plt.colorbar(im, ax=ax, fraction=0.046)

# uy
pred_2d = uy_pred[0].reshape(N_GRID, N_GRID)
true_2d = uy_true[0].reshape(N_GRID, N_GRID)
for ax, data, title in zip(axes[1],
    [true_2d, pred_2d, np.abs(pred_2d - true_2d)],
    ['FEM $u_y$', 'DeepONet $u_y$', '|Error| $u_y$']):
    im = ax.imshow(data, **kw)
    ax.set_title(title); plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('results_compression_mixed/validation_ux_uy_parametric.png', dpi=150, bbox_inches='tight')