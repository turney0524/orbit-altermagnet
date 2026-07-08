import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 模型参数定义
# ==========================================
eps_A = -1.0
eps_BC = 0.0
t1 = -0.6
t2 = -1.0
phi = np.pi / 3  # 相位 phi = pi/3

# ==========================================
# 2. 构建 k 空间哈密顿量和它的偏导数
# ==========================================
def get_H_and_derivs(kx, ky):
    H = np.zeros((6, 6), dtype=complex)
    dH_dkx = np.zeros((6, 6), dtype=complex)
    dH_dky = np.zeros((6, 6), dtype=complex)
    
    t2_phase = t2 * np.exp(1j * phi)
    
    # 原子编号: A1(0), A2(1), B1(2), B2(3), C1(4), C2(5)
    # 跃迁键列表：(跃迁终点 index, 跃迁起点 index, 跃迁振幅(复数), dx, dy)
    bonds = [
        # === 修正后的 t2 键 ===
        # 根据公式，所有正相位 +phi 的 hopping 都是从 B 指向 C
        (4, 3, t2_phase,  0.25,  0.0),   # B2(3) -> C1(4)
        (5, 3, t2_phase,  0.0,   0.25),  # B2(3) -> C2(5)
        (4, 2, t2_phase,  0.0,  -0.25),  # B1(2) -> C1(4)
        (5, 2, t2_phase, -0.25,  0.0),   # B1(2) -> C2(5)
        
        # === t1 键 (不变，保持正确) ===
        (3, 0, t1, -0.125,  0.375),      # A1 -> B2
        (4, 0, t1,  0.125,  0.375),      # A1 -> C1
        (2, 0, t1,  0.125, -0.375),      # A1 -> B1 (下原胞)
        (5, 0, t1, -0.125, -0.375),      # A1 -> C2 (下原胞)
        
        (3, 1, t1,  0.375, -0.125),      # A2 -> B2
        (5, 1, t1,  0.375,  0.125),      # A2 -> C2
        (2, 1, t1, -0.375,  0.125),      # A2 -> B1 (左原胞)
        (4, 1, t1, -0.375, -0.125),      # A2 -> C1 (左原胞)
    ]
    
    for out_idx, in_idx, amp, dx, dy in bonds:
        phase = 2 * np.pi * (kx * dx + ky * dy)
        term = amp * np.exp(1j * phase)
        
        # 正向跃迁
        H[out_idx, in_idx] += term
        dH_dkx[out_idx, in_idx] += 1j * 2 * np.pi * dx * term
        dH_dky[out_idx, in_idx] += 1j * 2 * np.pi * dy * term
        
        # 厄米共轭（反向跃迁）
        term_hc = np.conj(amp) * np.exp(-1j * phase)
        H[in_idx, out_idx] += term_hc
        dH_dkx[in_idx, out_idx] += 1j * 2 * np.pi * (-dx) * term_hc
        dH_dky[in_idx, out_idx] += 1j * 2 * np.pi * (-dy) * term_hc
        
    H[0, 0] += eps_A
    H[1, 1] += eps_A
    return H, dH_dkx, dH_dky

# ==========================================
# 3. 计算特定 k 点的能带本征值和 Lz
# ==========================================
def calc_bands_and_Lz(kx, ky):
    H, dH_dkx, dH_dky = get_H_and_derivs(kx, ky)
    evals, evecs = np.linalg.eigh(H)
    
    Lz = np.zeros(6)
    for n in range(6):
        un = evecs[:, n]
        lz_val = 0
        for m in range(6):
            if m == n: continue
            um = evecs[:, m]
            
            vx_nm = np.vdot(un, dH_dkx @ um)
            vy_mn = np.vdot(um, dH_dky @ un)
            vy_nm = np.vdot(un, dH_dky @ um)
            vx_mn = np.vdot(um, dH_dkx @ un)
            
            term1 = vx_nm * vy_mn
            term2 = vy_nm * vx_mn
            
            # 使用真实分母 (线性关系)
            lz_val += np.imag(term1 - term2) / (evals[n] - evals[m])
            
        Lz[n] = lz_val
    return evals, Lz

# ==========================================
# 4. 路径设定与网格计算
# ==========================================
k_nodes = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5), (0.0, 0.0)]
labels = [r'$\Gamma$', 'X', 'M', 'Y', r'$\Gamma$']

k_path = []
x_ticks = [0]
dist = 0
pts_per_seg = 80

for i in range(len(k_nodes)-1):
    k1, k2 = np.array(k_nodes[i]), np.array(k_nodes[i+1])
    segment = np.linspace(k1, k2, pts_per_seg, endpoint=False)
    k_path.extend(segment)
    dist += np.linalg.norm(k2 - k1)
    x_ticks.append(dist)
k_path.append(np.array(k_nodes[-1]))
k_path = np.array(k_path)

x_axis_dist = np.zeros(len(k_path))
for i in range(1, len(k_path)):
    x_axis_dist[i] = x_axis_dist[i-1] + np.linalg.norm(k_path[i] - k_path[i-1])

E_bands = np.zeros((len(k_path), 6))
Lz_bands = np.zeros((len(k_path), 6))

for i, (kx, ky) in enumerate(k_path):
    E_bands[i], Lz_bands[i] = calc_bands_and_Lz(kx, ky)

# ==========================================
# 5. 绘图 (应用百分位数截断来增强色彩)
# ==========================================
plt.figure(figsize=(10, 4.5))

# --- 图 1(b) ---
plt.subplot(1, 2, 1)
# 排除交叉点导致的奇异极大值，取 95% 分位数作为色彩阈值，使得普通区域颜色变浓！
vmax_1b = np.percentile(np.abs(Lz_bands), 95) 

for n in range(6):
    plt.scatter(x_axis_dist, E_bands[:, n], c=Lz_bands[:, n], 
                cmap='bwr', vmin=-vmax_1b, vmax=vmax_1b, s=8, alpha=1.0) # s=8 让点变大更明显

for tk in x_ticks:
    plt.axvline(tk, color='gray', linestyle='--', linewidth=0.5)

plt.xticks(x_ticks, labels)
plt.ylabel("Energy (eV)")
plt.xlim(0, x_axis_dist[-1])
plt.ylim(-3.2, 2.5)
plt.title("(b) Band structure with $L_z$")

# --- 图 1(c) ---
N_grid = 120
kx_mesh = np.linspace(-0.5, 0.5, N_grid)
ky_mesh = np.linspace(-0.5, 0.5, N_grid)
Lz_map = np.zeros((N_grid, N_grid))

target_band = 3 # 第 4 条能带 (index=3)

for ix, kx in enumerate(kx_mesh):
    for iy, ky in enumerate(ky_mesh):
        _, Lz_all = calc_bands_and_Lz(kx, ky)
        Lz_map[iy, ix] = Lz_all[target_band]

plt.subplot(1, 2, 2)
# 同样使用 98% 分位数截断发散点
vmax_1c = np.percentile(np.abs(Lz_map), 98)

mesh = plt.pcolormesh(kx_mesh, ky_mesh, Lz_map, 
                      cmap='bwr', shading='auto', 
                      vmin=-vmax_1c, vmax=vmax_1c)
plt.colorbar(mesh, label=r'$L_z(\mathbf{k})$ (arb. units)')

plt.plot([-0.5, 0.5], [-0.5, 0.5], color='gray', linestyle='--', alpha=0.5)
plt.plot([-0.5, 0.5], [0.5, -0.5], color='gray', linestyle='--', alpha=0.5)

plt.xlabel(r"$k_x$")
plt.ylabel(r"$k_y$")
plt.title(r"(c) $L_z$ distribution (4th band)")
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)
plt.xticks([-0.5, 0.0, 0.5])
plt.yticks([-0.5, 0.0, 0.5])

plt.tight_layout()
plt.show()
