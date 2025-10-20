import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm
import scipy.io as sio

def extract_data_pairs(dataset_folder='Dataset1', num_pairs=5):
    """
    提取Dataset1中的NIR和RFI_FREE数据对
    
    参数:
        dataset_folder: 数据集文件夹名称，默认为'Dataset1'
        num_pairs: 要提取的数据对数量，默认为5
    
    返回:
        包含所有数据对的字典列表
    """
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    dataset_path = script_dir / dataset_folder
    
    # 获取所有文件
    all_files = sorted(os.listdir(dataset_path))
    
    # 筛选出txt文件
    txt_files = [f for f in all_files if f.endswith('.txt')]
    
    # 按文件编号分组
    file_groups = {}
    for filename in txt_files:
        # 解析文件名: 编号_模式_类型.txt
        parts = filename.replace('.txt', '').split('_')
        file_number = parts[0]  # 文件编号
        mode = parts[1]  # 计划模式 (HHH, HHV, VVV等)
        data_type = '_'.join(parts[2:])  # NIR 或 RFI_FREE
        
        key = f"{file_number}_{mode}"
        if key not in file_groups:
            file_groups[key] = {}
        file_groups[key][data_type] = filename
    
    # 提取前num_pairs对数据
    result_list = []
    count = 0
    
    for key in sorted(file_groups.keys()):
        if count >= num_pairs:
            break
        
        group = file_groups[key]
        # 确保同时存在NIR和RFI_FREE文件
        if 'NIR' in group and 'RFI_FREE' in group:
            file_number, mode = key.split('_')
            
            # 读取NIR数据
            nir_path = dataset_path / group['NIR']
            nir_data = np.loadtxt(nir_path)
            
            # 读取RFI_FREE数据并reshape为(2556, 2)
            rfi_free_path = dataset_path / group['RFI_FREE']
            rfi_free_data = np.loadtxt(rfi_free_path)
            
            # 确保reshape为(2556, 2)
            if rfi_free_data.shape[0] == 2556 and rfi_free_data.shape[1] == 2:
                # 已经是正确的形状
                pass
            elif rfi_free_data.size == 2556 * 2:
                # 如果数据总量正确，但形状不对，进行reshape
                rfi_free_data = rfi_free_data.reshape(2556, 2)
            else:
                print(f"警告: {group['RFI_FREE']} 的数据大小不是2556*2，实际大小为{rfi_free_data.shape}")
            
            # 创建数据字典
            data_dict = {
                '文件编号': file_number,
                '计划模式': mode,
                'NIR': nir_data,
                'RFI_FREE': rfi_free_data,
                'NIR文件名': group['NIR'],
                'RFI_FREE文件名': group['RFI_FREE']
            }
            
            result_list.append(data_dict)
            count += 1
    
    return result_list

def convert_visib_full(file_data, pol_flag):
    """
    从原始可见度数据中提取特定极化模式的可见度数据并转换为2346×1的向量
    
    参数:
        file_data: numpy数组，形状为(2556, 2)，第一列为实部，第二列为虚部
        pol_flag: int，极化标志，1表示VVV极化，2表示HHH极化
    
    返回:
        visib_dual: numpy数组，形状为(2346,)，复数向量
    """
    # 系统参数设置
    LICEF_NUMBER = 72
    LICEFS_PER_ARM = 24
    LICEF_PER_POL = 69
    VISIB_COL_DUAL = LICEF_PER_POL * (LICEF_PER_POL - 1) // 2  # 2346
    
    # 将实部和虚部组合成复数形式
    calib_visib_complex = file_data[:, 0] + 1j * file_data[:, 1]
    
    # 极化选择
    rem_pol = pol_flag
    
    # 寻找有效的LICEF对(去除特定极化通道)
    h1 = []
    h2 = []
    
    for k in range(LICEF_NUMBER - 1):
        # 如果当前LICEF的极化模式不是要移除的
        if k % LICEFS_PER_ARM != rem_pol:
            # 寻找与当前LICEF配对的其他LICEF
            for i_p in range(k + 1, LICEF_NUMBER):
                # 如果配对LICEF的极化模式也不是要移除的
                if i_p % LICEFS_PER_ARM != rem_pol:
                    h1.append(k)
                    h2.append(i_p)
    
    # 转换为numpy数组
    h1 = np.array(h1)
    h2 = np.array(h2)
    
    # 将2556×1的原始可见度数据转换为72×72矩阵
    visib_dual_matrix = convert_calib_visib(calib_visib_complex)
    
    # 从72×72矩阵中提取筛选后的LICEF对的可见度数据
    visib_dual = np.zeros(VISIB_COL_DUAL, dtype=complex)
    
    for j in range(VISIB_COL_DUAL):
        visib_dual[j] = visib_dual_matrix[h1[j], h2[j]]
    
    return visib_dual

def convert_calib_visib(calib_visib):
    """
    将2556×1的可见度矩阵转换为72×72的对称矩阵
    
    参数:
        calib_visib: numpy数组，形状为(2556, 1)或(2556,)，包含可见度数据
    
    返回:
        visib_dual: numpy数组，形状为(72, 72)，对称矩阵
    """
    # 确保输入是一维数组
    if calib_visib.ndim > 1:
        calib_visib = calib_visib.flatten()
    
    # 初始化72×72的输出矩阵
    visib_dual = np.zeros((72, 72), dtype=calib_visib.dtype)
    
    # 填充上三角矩阵
    ii = 0
    for i1 in range(72):
        for i2 in range(i1 + 1, 72):
            visib_dual[i1, i2] = calib_visib[ii]
            ii += 1
    
    return visib_dual


def calculate_antenna_positions(ant_min_space):
    # 初始化A臂天线位置
    arm_A = np.zeros(18, dtype=complex)
    
    # 计算A臂天线位置(MIRAS)
    for j in range(6, 24):  # MATLAB的6:23对应Python的6:24
        arm_A[j-6] = -1 * (j-2) * ant_min_space
    
    # 计算子阵列A的位置
    sub_A = np.zeros(5, dtype=complex)
    sub_A[0] = 3 * ant_min_space * np.exp(1j * 2 * np.pi / 3)
    sub_A[1] = ant_min_space * np.exp(1j * 2 * np.pi / 3)
    for j in range(3, 6):  # MATLAB的3:5对应Python的3:6
        sub_A[j-1] = -1 * (j-2) * ant_min_space
    
    # 组合A臂和子阵列A
    A = np.concatenate([sub_A, arm_A])
    
    # 计算Y形阵列的B臂和C臂(旋转120°和240°)
    B = A * np.exp(-1j * 2 * np.pi / 3)
    C = B * np.exp(-1j * 2 * np.pi / 3)
    
    # 合并所有天线位置
    ant_pos = np.concatenate([A, B, C])
    
    # 提取实部和虚部坐标
    x = np.real(ant_pos)
    y = np.imag(ant_pos)
    
    return ant_pos


def calculate_baseline_distribution(ant_pos, visib_dual, T_NIR):
    # 计算共轭可见度(正反基线)
    visib_conj = np.concatenate([visib_dual, np.conj(visib_dual)])
    
    # 初始化协方差矩阵
    ant_num = len(ant_pos)
    visib_matrix = T_NIR * np.eye(ant_num, dtype=complex)  # 对角线为场景平均亮温
    
    # 计算基线向量和填充协方差矩阵
    UV_redun = np.zeros(len(visib_dual), dtype=complex)
    index = 0
    
    for i in range(ant_num):
        for j in range(i+1, ant_num):
            UV_redun[index] = ant_pos[i] - ant_pos[j]  # 基线向量
            visib_matrix[i, j] = visib_conj[index]  # 填充上半角
            index += 1
    
    # 使协方差矩阵共轭对称
    visib_matrix = visib_matrix + visib_matrix.conj().T
    # 包含正反方向的基线向量
    UV_redun_conj = np.concatenate([UV_redun, -UV_redun])
    
    return visib_matrix, UV_redun_conj


def generate_RFI_scenario(flag,index):
    # 初始化128×128=16384像素的亮度温度矩阵
    T_RFI = np.zeros((16384, 1))
    #flag决定RFI场景，RFI数目为4个，index在（1000，15000）上均匀分布，不可重复
    # index = np.random.choice(np.arange(1000, 15000), 4, replace=False)
    # index = [14699, 14734, 1429, 1150]
    if flag == 1:
        T_RFI[index, 0] = [350, 600, 350, 600]
    elif flag == 2:
        T_RFI[index, 0] = [900, 1500, 1300, 1000]
    elif flag == 3:
        T_RFI[index, 0] = 6000

    # 根据flag选择不同的RFI场景
    # if flag == 1:
    #     # 场景1：中等强度点源
    #     T_RFI[14699, 0] = 350  # MATLAB索引14700对应Python索引14699
    #     T_RFI[14744, 0] = 600  # MATLAB索引14745对应Python索引14744
    #     T_RFI[1399, 0] = 350   # MATLAB索引1400对应Python索引1399
    #     T_RFI[1744, 0] = 600   # MATLAB索引1745对应Python索引1744
    #     index = [14699, 14734, 1429, 1150]  # 对应MATLAB的[14700, 14735, 1430, 1151]
        
    # elif flag == 2:
    #     # 场景2：中强度点源
    #     T_RFI[14699, 0] = 900   # MATLAB索引14700对应Python索引14699
    #     T_RFI[14744, 0] = 1500  # MATLAB索引14745对应Python索引14744
    #     T_RFI[1399, 0] = 1300   # MATLAB索引1400对应Python索引1399
    #     T_RFI[1400, 0] = 1000   # MATLAB索引1401对应Python索引1400
    #     index = [14699, 14734, 1429, 1150]  # 对应MATLAB的[14700, 14735, 1430, 1151]
        
    # elif flag == 3:
    #     # 场景3：高强度点源
    #     T_RFI[14699, 0] = 2000  # MATLAB索引14700对应Python索引14699
    #     T_RFI[14734, 0] = 3000  # MATLAB索引14735对应Python索引14734
    #     T_RFI[1429, 0] = 2800   # MATLAB索引1430对应Python索引1429
    #     T_RFI[1150, 0] = 2500   # MATLAB索引1151对应Python索引1150
    #     index = [14699, 14734, 1429, 1150]  # 对应MATLAB的[14700, 14735, 1430, 1151]
        
    else:
        raise ValueError('无效的flag值，请输入1-3之间的整数')
    
    return T_RFI, index

def V2R(V, UV, R_UV):
    R = V[-1] * np.eye(R_UV.shape[0], dtype=np.complex128)
    rows, cols = R_UV.shape
    uv_len = len(UV)
    
    # Triple nested loop to fill the matrix R
    for i in range(rows):
        for j in range(cols):
            for k in range(uv_len):
                # Check for a match within the given tolerance
                # This matches: if abs(R_UV(i,j)-UV(k))<1e-4
                if np.abs(R_UV[i, j] - UV[k]) < 1e-4:
                    R[i, j] = V[k]
                    break  # Match found, break the innermost loop
    return R

def calculate_RFI_visible_function(F, T_RFI, UV_noredun, R_UV):
    # Calculate parameters
    u_max = 2 * 27.56
    del_xi = 2 / np.sqrt(3) / u_max
    del_s = del_xi**2 * np.sqrt(3) / 2

    # Calculate RFI visibility
    # F' * T_RFI in MATLAB is F.T @ T_RFI in NumPy
    V_RFI = F.conj().T @ T_RFI * del_s
    V_RFI_zero = np.sum(T_RFI * del_s)

    # Construct the RFI covariance matrix by calling the V2R function
    D_RFI_WZ = V2R(V_RFI, UV_noredun, R_UV)


    return  D_RFI_WZ


def calculate_dirty_visibility_function(D_RFI_WZ, D_original):
    D_RFI = D_RFI_WZ.copy()
    D_original_mod = D_original.copy()

    # --- 混合场景分析 (RFI + 自然背景) ---
    D = D_RFI + D_original_mod  # 组合协方差矩阵

    # 去除自然场景的对角线 (自相关)
    # MATLAB: D_original = D_original - diag(diag(D_original));
    D_original_mod = D_original_mod - np.diag(np.diag(D_original_mod))
    
    # 去除混合场景的对角线
    # MATLAB: D = D - diag(diag(D));
    D = D - np.diag(np.diag(D))

    return D, D_original_mod

def classify_imaging_regions(xi_kk, eta_kk):
    # 参数初始化
    theta_centers = np.arange(30, 361, 60)  # 参考圆中心角度分布（6个方向）
    radius = 2 * np.max(eta_kk)  # 参考圆分布半径

    # 生成参考圆中心坐标（复数平面）
    circ_center = radius * (np.cos(np.deg2rad(theta_centers)) + 1j * np.sin(np.deg2rad(theta_centers)))

    # 生成单位圆用于距离计算
    theta_circ = np.arange(0, 360)  # 单位圆角度采样
    circ = np.cos(np.deg2rad(theta_circ)) + 1j * np.sin(np.deg2rad(theta_circ))  # 单位圆坐标

    # 初始化输出
    AFFOV_Index = []
    AFOV_Index = []

    # 遍历所有像素点
    for i in range(len(xi_kk)):
        # 当前像素的复数坐标
        pixel_pos = xi_kk[i] + 1j * eta_kk[i]
        
        # 计算到所有参考圆中心的距离
        dist_to_centers = np.abs(pixel_pos - circ_center)
        
        # 判断混叠特性
        if np.all(dist_to_centers >= 1):
            AFFOV_Index.append(i)  # 无混叠区域
        else:
            AFOV_Index.append(i)  # 混叠区域

    # 转换为列向量（保持输出格式统一）
    AFFOV_Index = np.array(AFFOV_Index).reshape(-1, 1)
    AFOV_Index = np.array(AFOV_Index).reshape(-1, 1)

    return AFFOV_Index, AFOV_Index

def construct_middle_matrices(G_matrix_H_reg, u, v, xi_kk, eta_kk, del_s):    
    # 确保输入为 numpy 数组，并检查数据类型
    G_matrix_H_reg = np.array(G_matrix_H_reg, dtype=np.complex128)  # 形状 (4692, 16384)
    u = np.array(u, dtype=np.float64)  # 形状 (2790, 1)
    v = np.array(v, dtype=np.float64)  # 形状 (2790, 1)
    xi_kk = np.array(xi_kk, dtype=np.float64)  # 形状 (35533, 1)
    eta_kk = np.array(eta_kk, dtype=np.float64)  # 形状 (35533, 1)
    
    # 步骤 1: 构造复数形式的 G_matrix_H_complex
    # 前 2346 行是实部，后 2346 行是虚部
    G_matrix_H_complex = G_matrix_H_reg[:2346, :] + 1j * G_matrix_H_reg[2346:, :]  # 形状 (2346, 16384)
    
    # 步骤 2: 构造共轭扩展的 G 矩阵
    # 将 G_matrix_H_complex 和其共轭垂直拼接
    G_matrix_H_conj = np.vstack([G_matrix_H_complex, np.conj(G_matrix_H_complex)])  # 形状 (4692, 16384)
    
    # 步骤 3: 构造正向傅里叶矩阵 F_C
    # F_C = exp(2i * pi * (xi_kk * u^T + eta_kk * v^T))
    F_C = np.exp(2j * np.pi * (xi_kk @ u.T + eta_kk @ v.T))  # 形状 (35533, 2790)
    
    # 步骤 4: 构造反向傅里叶矩阵 F_Forward
    # F_Forward = exp(-2i * pi * (u * xi_kk^T + v * eta_kk^T))
    F_Forward = np.exp(-2j * np.pi * (u @ xi_kk.T + v @ eta_kk.T))  # 形状 (2790, 35533)
    
    # 步骤 5: 构造 F_real 矩阵
    # 在 F_C 后追加一列零，并乘以缩放因子 del_s
    zero_column = np.zeros((35533, 1), dtype=np.complex128)  # 零列，形状 (35533, 1)
    F_real = del_s * np.hstack([F_C, zero_column])  # 形状 (35533, 2791)
    
    return G_matrix_H_conj, F_C, F_Forward, F_real

def inverse_DFT_scale_complex(k, visib_matrix_fil, UV_redun_conj, G_matrix_H_conj, F_real, xi_kk, eta_kk, T_NIR, AFFOV_idx):
    # 1. 从协方差矩阵提取可见度数据
    ant_num = 69  # MIRAS 系统天线数量
    visib_fil = []
    index = 0

    # 遍历上三角部分提取非冗余可见度
    for i in range(ant_num):
        for j in range(i + 1, ant_num):
            index += 1
            visib_fil.append(visib_matrix_fil[i, j])  # 提取基线 (i,j) 的可见度

    visib_fil = np.array(visib_fil, dtype=np.complex128)  # 转换为 numpy 数组

    # 2. 构建共轭对称可见度向量
    visib_conj = np.concatenate([visib_fil.conj().T, visib_fil.T])   # 共轭扩展（正反基线）

    # 3. 基线坐标处理
    UV = np.column_stack([np.real(UV_redun_conj), np.imag(UV_redun_conj)])  # 转换为 (u,v) 坐标对
    UV_t = UV * 10e10  # 放大坐标（避免浮点精度问题）
    UV_T = np.round(UV_t)  # 取整便于唯一性判断

    # 去除冗余基线（相同 UV 坐标只保留一个）
    _, N_index = np.unique(UV_T, axis=0, return_index=True)
    UV_noredun = UV_redun_conj[N_index]  # 非冗余基线坐标
    visib_noredun = visib_conj[N_index]  # 对应的可见度

    G_matrix_H_noredun = G_matrix_H_conj[N_index, :]  # 对应的系统响应

    # 4. 冗余数据平均
    visib_noredun_ave = np.zeros_like(visib_noredun, dtype=np.complex128)

    G_matrix_H_conj_ave = np.zeros_like(G_matrix_H_noredun, dtype=np.complex128)
    redun_num = np.zeros_like(visib_noredun, dtype=np.float64)  # 记录每个基线的冗余次数

    # 遍历所有基线，对冗余测量取平均
    for i in range(len(UV_noredun)):
        for j in range(len(UV_redun_conj)):
            # 判断基线是否匹配（考虑浮点误差）
            if np.abs(UV[j, 0] + 1j * UV[j, 1] - UV_noredun[i]) < 1e-4:
                redun_num[i] += 1
                visib_noredun_ave[i] += visib_conj[j]
                G_matrix_H_conj_ave[i, :] += G_matrix_H_conj[j, :]
        # 计算平均值
        visib_noredun_ave[i] /= redun_num[i]
        G_matrix_H_conj_ave[i, :] /= redun_num[i]

    # 5. 合并 NIR 数据并反演
    visib_noredun_ave_real = np.concatenate([visib_noredun_ave, [T_NIR]])  # 添加噪声基底
    T_dft = F_real @ visib_noredun_ave_real  # DFT 反演，形状 (35533, 1)

    # 6. 可视化设置
    zoom_para = 2  # 显示范围缩放因子
    max_arm = 21  # 最大臂长（波长数）
    min_space = 0.875  # 最小基线间距
    Fov_min_space = zoom_para / np.sqrt(3) / (3 * max_arm) / min_space
    r = Fov_min_space / np.sqrt(3)  # 六边形单元半径

    # 生成六边形顶点坐标
    angle = np.arange(0, 2 * np.pi, np.pi / 3)
    x = r * np.cos(angle) / 2
    y = r * np.sin(angle) / 2

    AFFOV_idx = AFFOV_idx.flatten()  # 展平索引数组以便迭代

    # 8. 绘制亮温分布（六边形网格）
    X = xi_kk.flatten()
    Y = eta_kk.flatten()
    verts = []  # 存储六边形顶点
    colors = []  # 存储亮温值
    for p in range(len(AFFOV_idx)):
        if (X[AFFOV_idx[p]]**2 + Y[AFFOV_idx[p]]**2) <= 1:  # 仅在单位圆内绘制
            xaxis = X[AFFOV_idx[p]] + x
            yaxis = Y[AFFOV_idx[p]] + y
            verts.append(list(zip(xaxis, yaxis)))  # 六边形顶点坐标
            colors.append(np.real(T_dft[AFFOV_idx[p]]) + T_NIR)  # 亮温值

    # 创建 PolyCollection，关联颜色
    fig, ax = plt.subplots()
    poly_collection = PolyCollection(verts, array=np.array(colors), cmap=cm.jet, edgecolors='none')
    ax.add_collection(poly_collection)


    # 9. 图形修饰
    cbar = plt.colorbar(poly_collection, ax=ax)  # 关联 PolyCollection 的颜色条
    cbar.set_label('[K]', fontsize=12)  # 添加单位标签
    ax.set_title(k)
    ax.set_xlim(np.min(xi_kk), np.max(xi_kk))
    ax.set_ylim(np.min(eta_kk), np.max(eta_kk))
    ax.text(0.83, 0.71, '[K]', fontsize=12)  # 添加单位标注
    ax.set_xlabel(r'$\xi$', fontsize=14)
    ax.set_ylabel(r'$\eta$', fontsize=14)
    ax.set_aspect('equal')  # 确保坐标轴比例相等
    
    #存储本次绘图结果到./1_pic/，取名为输入k
    plt.savefig(f"./1_pic/{k}_{index1}.png", bbox_inches='tight')
    return T_dft, visib_noredun_ave_real

def inverse_DFT_complex(k, visib_matrix_fil, UV_redun_conj, G_matrix_H_conj, F_real, xi_kk, eta_kk, T_NIR, AFFOV_idx, index1):
    # 功能：基于离散傅里叶变换(DFT)的亮温反演与可视化
    # 输入：
    #   k - 图像标题字符串（如'originalBTmap'）
    #   visib_matrix_fil - 滤波后的协方差矩阵（69×69复数矩阵）
    #   UV_redun_conj - 冗余基线向量集合（含正反方向，N×1复数）
    #   G_matrix_H_conj - 系统响应矩阵（共轭扩展后的复数矩阵）
    #   F_real - 傅里叶变换矩阵（已补零和缩放）
    #   xi_kk, eta_kk - 非规则网格坐标（35533×1）
    #   T_NIR - 噪声基底温度（标量）
    # 输出：
    #   T_dft - 反演亮温分布（35533×1）
    #   visib_noredun_ave_real - 平均后的可见度数据（含NIR）

    ## 第一阶段：数据预处理
    # 1.1 从协方差矩阵提取非冗余可见度（上三角部分）
    ant_num = 69  # MIRAS系统天线数量
    num_nonredundant = ant_num * (ant_num - 1) // 2
    visib_fil = np.zeros(num_nonredundant, dtype=complex)
    index = 0
    for i in range(ant_num):
        for j in range(i + 1, ant_num):
            visib_fil[index] = visib_matrix_fil[i, j]  # 提取基线(i,j)的可见度
            index += 1

    # 1.2 构建共轭对称可见度向量
    visib_conj = np.concatenate([visib_fil.conj().T, visib_fil.T])

    # 1.3 基线坐标去冗余
    UV_real = np.real(UV_redun_conj)
    UV_imag = np.imag(UV_redun_conj)
    UV = np.column_stack((UV_real, UV_imag))  # 转换为(u,v)坐标
    UV_t = UV * 10e10  # 数值放大（避免浮点误差）
    UV_T = np.round(UV_t)  # 取整
    _, N_index = np.unique(UV_T, axis=0, return_index=True)  # 获取唯一基线索引
    UV_noredun = UV_redun_conj[N_index]  # 非冗余基线坐标
    visib_noredun = visib_conj[N_index]  # 对应可见度
    G_matrix_H_noredun = G_matrix_H_conj[N_index, :]  # 对应系统响应

    ## 第二阶段：冗余数据平均
    visib_noredun_ave = np.zeros_like(visib_noredun, dtype=complex)
    G_matrix_H_conj_ave = np.zeros_like(G_matrix_H_noredun, dtype=complex)
    redun_num = np.zeros(len(UV_noredun), dtype=int)

    # 2.1 遍历所有基线，对冗余测量取平均
    UV_redun_complex = UV_redun_conj
    for i in range(len(UV_noredun)):
        for j in range(len(UV_redun_conj)):
            if abs(UV_redun_complex[j] - UV_noredun[i]) < 1e-4:  # 基线匹配判断
                redun_num[i] += 1
                visib_noredun_ave[i] += visib_conj[j]
                G_matrix_H_conj_ave[i, :] += G_matrix_H_conj[j, :]
        if redun_num[i] > 0:
            visib_noredun_ave[i] /= redun_num[i]
            G_matrix_H_conj_ave[i, :] /= redun_num[i]

    # 2.2 合并噪声基底数据
    visib_noredun_ave_real = np.append(visib_noredun_ave, T_NIR)

    ## 第三阶段：DFT反演
    # 3.1 权重计算（Tukey窗函数）
    UV_noredun = np.append(UV_noredun, 0)  # 添加零基线
    u = np.real(UV_noredun)
    v = np.imag(UV_noredun)
    # Tukey窗权重计算（MATLAB转Python）
    pi = np.pi
    uv_norm = np.sqrt(u**2 + v**2) / np.sqrt(3) / 0.875 / 23
    w = 0.42 + 0.5 * np.cos(pi * uv_norm) + 0.08 * np.cos(2 * pi * uv_norm)

    # 3.2 执行反演
    T_dft = F_real @ (w * visib_noredun_ave_real)  # 加权DFT变换

    # ## 第四阶段：可视化
    # # 4.1 计算六边形网格参数
    # zoom_para = 2
    # max_arm = 21
    # min_space = 0.875
    # Fov_min_space = zoom_para / np.sqrt(3) / (3 * max_arm) / min_space
    # r = Fov_min_space / np.sqrt(3)  # 六边形单元半径

    # # 4.2 生成六边形顶点
    # angle = np.arange(0, 2 * np.pi, np.pi / 3)
    # x = r * np.cos(angle) / 2
    # y = r * np.sin(angle) / 2

    # AFFOV_idx = AFFOV_idx.flatten()  # 展平索引数组以便迭代
    # # 4.4 绘制亮温分布（六边形网格）
    # X = xi_kk.flatten()
    # Y = eta_kk.flatten()
    # verts = []  # 存储六边形点的位置
    # colors = []  # 存储亮温值
    # for p in range(len(xi_kk[AFFOV_idx])):
    #     if (X[AFFOV_idx[p]]**2 + Y[AFFOV_idx[p]]**2) <= 1:  # 仅在单位圆内绘制
    #         xaxis = X[AFFOV_idx[p]] + x
    #         yaxis = Y[AFFOV_idx[p]] + y
    #         verts.append(list(zip(xaxis, yaxis)))  # 六边形顶点坐标
    #         colors.append(np.real(T_dft[AFFOV_idx[p]]) + T_NIR)  # 亮温值

    # # 创建 PolyCollection，关联颜色
    # fig, ax = plt.subplots()
    # poly_collection = PolyCollection(verts, array=np.array(colors), cmap=cm.jet, edgecolors='none')
    # ax.add_collection(poly_collection)


    # # 4.6 图形修饰
    # cbar = plt.colorbar(poly_collection, ax=ax)  # 关联 PolyCollection 的颜色条
    # cbar.set_label('[K]', fontsize=12)  # 添加单位标签
    # ax.set_title(k)
    # ax.set_xlim(np.min(xi_kk), np.max(xi_kk))
    # ax.set_ylim(np.min(eta_kk), np.max(eta_kk))
    # ax.text(0.83, 0.71, '[K]', fontsize=12)  # 添加单位标注
    # ax.set_xlabel(r'$\xi$', fontsize=14)
    # ax.set_ylabel(r'$\eta$', fontsize=14)
    # ax.set_aspect('equal')  # 确保坐标轴比例相等
    # #存储本次绘图结果到./1_pic/，取名为输入k，如已经存在则覆盖
    # plt.savefig(f"./1_pic/{k}_{index1}.png", bbox_inches='tight')
    

    return T_dft, visib_noredun_ave_real

def load_NIR(T_NIR_data, pol_flag):
    # 确保输入是一维数组
    if T_NIR_data.ndim > 1:
        T_NIR_data = T_NIR_data.flatten()
    
    # 将12个元素重塑为4×3矩阵，使用Fortran风格（列优先）以匹配MATLAB
    calib_NIR_matrix = T_NIR_data.reshape(4, 3, order='F')
    
    if pol_flag == 1:
        # VV极化 - 使用第2行(索引1)的平均值
        calib_NIR_HH = np.mean(calib_NIR_matrix[1, :])
    else:
        # HH极化 - 使用第1行(索引0)的平均值
        calib_NIR_HH = np.mean(calib_NIR_matrix[0, :])
    
    return calib_NIR_HH

    
def read_file_to_array(file_path):
    # 获取文件扩展名
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == '.txt':
        # 读取txt文件并转换为numpy数组
        # 使用numpy.loadtxt来处理文本文件，支持逗号分隔
        data = np.loadtxt(file_path, delimiter=',', dtype=float)
        
        return data.astype(np.float32)
    
    elif ext == '.mat':
        # 读取mat文件
        mat_data = sio.loadmat(file_path)
        
        # 获取第一个非系统变量的数据
        # mat文件通常包含__header__, __version__, __globals__等系统变量
        data_key = None
        for key in mat_data.keys():
            if not key.startswith('__'):
                data_key = key
                break
        
        if data_key is None:
            raise ValueError("mat文件中没有找到有效的数据变量")
        
        data = mat_data[data_key]
        
        # 检查数据类型，如果是复数则保持复数格式
        if np.iscomplexobj(data):
            return data.astype(np.complex64)  # 使用complex64保持复数精度
        else:
            return data.astype(np.float32)
    
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
