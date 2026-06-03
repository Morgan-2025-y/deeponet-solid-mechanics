"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
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
for i in range(N):
    if i % 200 == 0: print(f"  {i}/{N}")
    ux_grid[i] = griddata(
        coors_all[i].astype(np.float64),
        ux_all_raw[i],
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

# Load dataset
d = np.load("antiderivative_aligned_train.npz", allow_pickle=True)
X_train = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
y_train = d["y"].astype(np.float32)
d = np.load("antiderivative_aligned_test.npz", allow_pickle=True)
X_test = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
y_test = d["y"].astype(np.float32)
# 4. 训练 / 测试划分
# ============================================================
perm      = np.random.permutation(N)
b_train   = branch_norm[perm[:800]];   b_test  = branch_norm[perm[800:]]
ux_train  = ux_norm[perm[:800]];       ux_test = ux_norm[perm[800:]]
print(f"Train: {b_train.shape}, Test: {b_test.shape}")

data = dde.data.TripleCartesianProd(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

# Choose a network
m = 100
dim_x = 1
net = dde.nn.DeepONetCartesianProd(
    [m, 40, 40],
    [dim_x, 40, 40],
    "relu",
    "Glorot normal",
)

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

# Plot the loss trajectory
dde.utils.plot_loss_history(losshistory)
plt.show()