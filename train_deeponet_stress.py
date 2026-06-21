"""
DeepONet 应力场训练脚本 — 压缩工况，预测四个应力分量
======================================================
算子映射: G : f_left(y) -> (σ_xx, σ_yy, σ_xy, Von Mises)(x, y)

依赖文件:
  - plate_no_hole_planestress.mat   (MATLAB FEM 输出，包含应力场数据)

输出文件:
  - results_stress/norm_params_stress.npy   归一化参数
  - results_stress/model_stress-*.pt        训练中间检查点
  - results_stress/model_stress_final-*.pt  最终模型权重
  - results_stress/validation_stress.png    验证可视化图
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
os.makedirs("results_stress_shear", exist_ok=True)

# ============================================================
# 1. 加载数据 & 筛选压缩工况 (Type 1)
# ============================================================
print("=" * 55)
print("Step 1: Loading data...")
print("=" * 55)

mat = sio.loadmat('plate_no_hole_planestress_no_ev.mat', squeeze_me=True)

f_type    = mat['f_type'].astype(int)
f_bc_top   = mat['f_bc_top'].astype(np.float32)  # (3000, 101)

# 应力场
sxx_list = mat['final_sxx']
syy_list = mat['final_syy']
sxy_list = mat['final_sxy']
vm_list  = mat['final_vonmises']

# 节点坐标
coors_list = mat['coors_dict']

# 筛选压缩工况
idx_comp   = np.where(f_type == 2)[0]
N          = len(idx_comp)
print(f"压缩样本数: {N}")

branch_all = f_bc_top[idx_comp, :]             # (N, 101)
coors_all  = [coors_list[i] for i in idx_comp]
sxx_all    = [sxx_list[i]   for i in idx_comp]
syy_all    = [syy_list[i]   for i in idx_comp]
sxy_all    = [sxy_list[i]   for i in idx_comp]
vm_all     = [vm_list[i]    for i in idx_comp]

# ============================================================
# 2. 插值到固定均匀网格 (Trunk 查询点)
# ============================================================
print("\n" + "=" * 55)
print("Step 2: Interpolating stress fields to uniform grid...")
print("=" * 55)

N_GRID    = 32
HALF      = 0.25
xq        = np.linspace(-HALF, HALF, N_GRID)
yq        = np.linspace(-HALF, HALF, N_GRID)
XX, YY    = np.meshgrid(xq, yq)
trunk_pts = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)  # (1024, 2)
N_Q       = trunk_pts.shape[0]

sxx_grid = np.zeros((N, N_Q), dtype=np.float32)
syy_grid = np.zeros((N, N_Q), dtype=np.float32)
sxy_grid = np.zeros((N, N_Q), dtype=np.float32)
vm_grid  = np.zeros((N, N_Q), dtype=np.float32)

for i in range(N):
    if i % 200 == 0:
        print(f"  {i}/{N}")
    coords = coors_all[i].astype(np.float64)
    sxx_grid[i] = griddata(coords, sxx_all[i], trunk_pts.astype(np.float64),
                            method='linear', fill_value=0.0).astype(np.float32)
    syy_grid[i] = griddata(coords, syy_all[i], trunk_pts.astype(np.float64),
                            method='linear', fill_value=0.0).astype(np.float32)
    sxy_grid[i] = griddata(coords, sxy_all[i], trunk_pts.astype(np.float64),
                            method='linear', fill_value=0.0).astype(np.float32)
    vm_grid[i]  = griddata(coords, vm_all[i],  trunk_pts.astype(np.float64),
                            method='linear', fill_value=0.0).astype(np.float32)

print(f"\n应力范围检查:")
print(f"  sxx : [{sxx_grid.min():.4e}, {sxx_grid.max():.4e}] GPa")
print(f"  syy : [{syy_grid.min():.4e}, {syy_grid.max():.4e}] GPa")
print(f"  sxy : [{sxy_grid.min():.4e}, {sxy_grid.max():.4e}] GPa")
print(f"  vm  : [{vm_grid.min():.4e},  {vm_grid.max():.4e}] GPa")

# ============================================================
# 3. 标准化
# ============================================================
print("\n" + "=" * 55)
print("Step 3: Normalizing...")
print("=" * 55)

# Branch 载荷曲线
b_mean, b_std     = branch_all.mean(), branch_all.std() + 1e-12
branch_norm        = (branch_all - b_mean) / b_std

# 四个应力分量各自独立标准化
sxx_mean, sxx_std = sxx_grid.mean(), sxx_grid.std() + 1e-12
syy_mean, syy_std = syy_grid.mean(), syy_grid.std() + 1e-12
sxy_mean, sxy_std = sxy_grid.mean(), sxy_grid.std() + 1e-12
vm_mean,  vm_std  = vm_grid.mean(),  vm_grid.std()  + 1e-12

sxx_norm = (sxx_grid - sxx_mean) / sxx_std
syy_norm = (syy_grid - syy_mean) / syy_std
sxy_norm = (sxy_grid - sxy_mean) / sxy_std
vm_norm  = (vm_grid  - vm_mean)  / vm_std

print(f"  Branch: mean={b_mean:.4e}, std={b_std:.4e}")
print(f"  sxx   : mean={sxx_mean:.4e}, std={sxx_std:.4e}")
print(f"  syy   : mean={syy_mean:.4e}, std={syy_std:.4e}")
print(f"  sxy   : mean={sxy_mean:.4e}, std={sxy_std:.4e}")
print(f"  vm    : mean={vm_mean:.4e},  std={vm_std:.4e}")

# 保存归一化参数
np.save("results_stress_shear/norm_params_stress.npy", {
    'b_mean':   b_mean,   'b_std':   b_std,
    'sxx_mean': sxx_mean, 'sxx_std': sxx_std,
    'syy_mean': syy_mean, 'syy_std': syy_std,
    'sxy_mean': sxy_mean, 'sxy_std': sxy_std,
    'vm_mean':  vm_mean,  'vm_std':  vm_std,
    'trunk_pts': trunk_pts
})
print("  归一化参数已保存: results_stress_shear/norm_params_stress.npy")

# ============================================================
# 4. 训练 / 测试划分
# ============================================================
print("\n" + "=" * 55)
print("Step 4: Train/Test split (8:2)...")
print("=" * 55)

perm    = np.random.permutation(N)
n_train = 800

b_train = branch_norm[perm[:n_train]]    # (800, 101)
b_test  = branch_norm[perm[n_train:]]    # (200, 101)

# 标签: (N, N_Q, 4) — 四个应力分量叠加
y_train = np.stack([
    sxx_norm[perm[:n_train]],
    syy_norm[perm[:n_train]],
    sxy_norm[perm[:n_train]],
    vm_norm [perm[:n_train]],
], axis=-1).astype(np.float32)   # (800, 1024, 4)

y_test = np.stack([
    sxx_norm[perm[n_train:]],
    syy_norm[perm[n_train:]],
    sxy_norm[perm[n_train:]],
    vm_norm [perm[n_train:]],
], axis=-1).astype(np.float32)   # (200, 1024, 4)

print(f"  Train branch: {b_train.shape}, label: {y_train.shape}")
print(f"  Test  branch: {b_test.shape},  label: {y_test.shape}")

# ============================================================
# 5. 构建 DeepONet (Cartesian Product 格式, 4 输出)
# ============================================================
print("\n" + "=" * 55)
print("Step 5: Building DeepONet (4 outputs)...")
print("=" * 55)

P = 128   # 每个输出头的内积维度
# split_both: Branch 和 Trunk 末层各 P*num_outputs 维，分拆给每个输出头

data = dde.data.TripleCartesianProd(
    X_train=(b_train, trunk_pts),
    y_train=y_train,    # (800, 1024, 4)
    X_test =(b_test,  trunk_pts),
    y_test =y_test      # (200, 1024, 4)
)

net = dde.nn.DeepONetCartesianProd(
    layer_sizes_branch=[101, 256, 256, 256, P * 4],   # 末层 512 维，分给 4 个输出头
    layer_sizes_trunk =[  2, 256, 256, 256, P * 4],
    activation="tanh",
    kernel_initializer="Glorot normal",
    num_outputs=4,
    multi_output_strategy="split_both",
)

model = dde.Model(data, net)

print("  网络结构:")
print(f"    Branch: [101, 256, 256, 256, {P*4}]")
print(f"    Trunk : [2,   256, 256, 256, {P*4}]")
print(f"    输出头: 4 (sxx, syy, sxy, vm), 每头维度 P={P}")

# ============================================================
# 6. 训练: Adam → L-BFGS 两阶段
# ============================================================
print("\n" + "=" * 55)
print("Step 6: Training — Stage 1: Adam (40000 iters)")
print("=" * 55)

model.compile(
    "adam",
    lr=1e-3,
    metrics=["mean l2 relative error"],
    decay=("inverse time", 10000, 0.5)
)
losshistory, trainstate = model.train(
    iterations=40000,
    batch_size=None,           # CartesianProd 全批次
    display_every=2000,
    model_save_path="results_stress_shear/model_stress_shear"
)

print("\n" + "=" * 55)
print("Step 6: Training — Stage 2: L-BFGS fine-tuning")
print("=" * 55)

model.compile("L-BFGS", metrics=["mean l2 relative error"])
losshistory, trainstate = model.train(
    display_every=500,
    model_save_path="results_stress_shear/model_stress_shear_final"
)

# ============================================================
# 7. 评估
# ============================================================
print("\n" + "=" * 55)
print("Step 7: Evaluation...")
print("=" * 55)

pred_norm = model.predict((b_test, trunk_pts))   # (200, 1024, 4)

# 反归一化
sxx_pred = pred_norm[..., 0] * sxx_std + sxx_mean
syy_pred = pred_norm[..., 1] * syy_std + syy_mean
sxy_pred = pred_norm[..., 2] * sxy_std + sxy_mean
vm_pred  = pred_norm[..., 3] * vm_std  + vm_mean

sxx_true = y_test[..., 0] * sxx_std + sxx_mean
syy_true = y_test[..., 1] * syy_std + syy_mean
sxy_true = y_test[..., 2] * sxy_std + sxy_mean
vm_true  = y_test[..., 3] * vm_std  + vm_mean

print(f"\n  {'分量':>8s}  {'Mean':>8s}  {'Std':>8s}  {'Median':>8s}  "
      f"{'Max':>8s}  {'<1%':>8s}")
print("  " + "-" * 55)

for name, pred, true in [
    ('sxx', sxx_pred, sxx_true),
    ('syy', syy_pred, syy_true),
    ('sxy', sxy_pred, sxy_true),
    ('vm',  vm_pred,  vm_true),
]:
    err = (np.linalg.norm(pred - true, axis=1) /
           (np.linalg.norm(true, axis=1) + 1e-12))
    print(f"  {name:>8s}  "
          f"{err.mean()*100:7.3f}%  "
          f"{err.std()*100:7.3f}%  "
          f"{np.median(err)*100:7.3f}%  "
          f"{err.max()*100:7.3f}%  "
          f"{(err<0.01).mean()*100:6.1f}%")

# ============================================================
# 8. 可视化 — 2行 x 4列 (FEM 真值 vs DeepONet 预测)
#    行1: FEM 真值   sxx / syy / sxy / vm
#    行2: DeepONet  sxx / syy / sxy / vm
#    另附一张 4列误差图
# ============================================================
print("\n" + "=" * 55)
print("Step 8: Saving validation plots...")
print("=" * 55)

comp_names = [r'$\sigma_{xx}$ (GPa)', r'$\sigma_{yy}$ (GPa)',
              r'$\sigma_{xy}$ (GPa)', 'Von Mises (GPa)']
comp_cmaps  = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'jet']
preds_list  = [sxx_pred, syy_pred, sxy_pred, vm_pred]
trues_list  = [sxx_true, syy_true, sxy_true, vm_true]

kw = dict(origin='lower', extent=[-HALF, HALF, -HALF, HALF])

# --- 图1: FEM 真值 vs DeepONet 预测 (取测试集第0个样本) ---
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle('Stress Field — FEM vs DeepONet (Test Sample #0, Type 2 Shear)',
             fontsize=12)

for col, (name, pred, true, cmap) in enumerate(
        zip(comp_names, preds_list, trues_list, comp_cmaps)):

    true_2d = true[0].reshape(N_GRID, N_GRID)
    pred_2d = pred[0].reshape(N_GRID, N_GRID)

    # 对称色标（法向/剪切应力），von Mises 从 0 开始
    if 'Von' in name:
        vmin, vmax = 0, max(true_2d.max(), pred_2d.max())
    else:
        vabs       = max(abs(true_2d.min()), abs(true_2d.max()))
        vmin, vmax = -vabs, vabs

    for row, (data, row_label) in enumerate([
        (true_2d, 'FEM'),
        (pred_2d, 'DeepONet'),
    ]):
        ax = axes[row, col]
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, **kw)
        ax.set_title(f'{row_label} {name}', fontsize=10)
        ax.set_xlabel('x (m)', fontsize=9)
        ax.set_ylabel('y (m)', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
save_path = 'results_stress_shear/validation_stress_shear_pred.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}")

# --- 图2: 绝对误差场 (1行 x 4列) ---
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
fig.suptitle('|Error| Field — DeepONet vs FEM (Test Sample #0)', fontsize=12)

for ax, name, pred, true, cmap in zip(
        axes, comp_names, preds_list, trues_list, comp_cmaps):
    true_2d = true[0].reshape(N_GRID, N_GRID)
    pred_2d = pred[0].reshape(N_GRID, N_GRID)
    err_2d  = np.abs(pred_2d - true_2d)

    im = ax.imshow(err_2d, cmap='hot_r', origin='lower',
                   extent=[-HALF, HALF, -HALF, HALF],
                   vmin=0, vmax=err_2d.max())
    ax.set_title(f'|Error| {name}', fontsize=10)
    ax.set_xlabel('x (m)', fontsize=9)
    ax.set_ylabel('y (m)', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
save_path = 'results_stress_shear/validation_stress_shear_error.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}")

# --- 图3: 误差分布直方图 (4个分量叠在一张图) ---
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle('Relative L2 Error Distribution — 200 Test Samples', fontsize=12)

short_names = ['sxx', 'syy', 'sxy', 'vm']
for ax, name, pred, true in zip(axes, short_names, preds_list, trues_list):
    err = (np.linalg.norm(pred - true, axis=1) /
           (np.linalg.norm(true, axis=1) + 1e-12))
    ax.hist(err * 100, bins=25, color='steelblue',
            edgecolor='white', linewidth=0.5)
    ax.axvline(err.mean() * 100, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean={err.mean()*100:.2f}%')
    ax.axvline(np.median(err) * 100, color='orange', linestyle='--', linewidth=1.5,
               label=f'Median={np.median(err)*100:.2f}%')
    ax.set_xlabel('Relative L2 Error (%)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'σ_{name} Error', fontsize=11)
    ax.legend(fontsize=8)

plt.tight_layout()
save_path = 'results_stress_shear/validation_stress_shear_histogram.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}")

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 55)
print("  全部完成")
print("=" * 55)
print("输出文件:")
print("  results_stress_shear/norm_params_stress_shear.npy       — 归一化参数")
print("  results_stress_shear/model_stress_shear_final-*.pt      — 最终模型权重")
print("  results_stress_shear/validation_stress_shear_pred.png   — FEM vs DeepONet 对比图")
print("  results_stress_shear/validation_stress_shear_error.png  — 误差场云图")
print("  results_stress_shear/validation_stress_shear_histogram.png — 误差分布直方图")
