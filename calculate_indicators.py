import os
import numpy as np
from log import log_print
from pathlib import Path
from method import convert_visib_full, convert_calib_visib,extract_data_pairs,calculate_antenna_positions,calculate_baseline_distribution,calculate_RFI_visible_function,calculate_dirty_visibility_function,classify_imaging_regions,construct_middle_matrices,load_NIR,inverse_DFT_complex,inverse_DFT_scale_complex,read_file_to_array,generate_RFI_scenario
import matplotlib.pyplot as plt
import time
from pathlib import Path
from method import convert_visib_full, convert_calib_visib,extract_data_pairs,calculate_antenna_positions,calculate_baseline_distribution,calculate_RFI_visible_function,calculate_dirty_visibility_function,classify_imaging_regions,construct_middle_matrices,load_NIR,inverse_DFT_complex,inverse_DFT_scale_complex,read_file_to_array,generate_RFI_scenario
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":

    pol_flag = 2 # VV极化1 HH极化2
    ant_min_space = 0.875; # 天线最小间距(波长)
    del_s = ant_min_space**2 * np.sqrt(3) / 2
    UV_noredun = read_file_to_array('data/UV_noredun.mat')
    u = np.real(UV_noredun)      # 取 UV_noredun 的实部作为 u
    v = np.imag(UV_noredun)      # 取 UV_noredun 的虚部作为 v
    R_UV = read_file_to_array('data/R_UV.mat')
    F = read_file_to_array('data/F.mat')

    xi_kk = read_file_to_array('data/XI.mat')
    eta_kk = read_file_to_array('data/ETA.mat')

    G_matrix_H_reg = read_file_to_array('data/G_matrix_H_reg.mat')
    G_matrix_H_conj, F_C, F_Forward, F_real = construct_middle_matrices(G_matrix_H_reg, u, v, xi_kk, eta_kk, del_s)

    
    ant_pos = calculate_antenna_positions(ant_min_space)
    [AFFOV_idx, AFOV_idx] = classify_imaging_regions(xi_kk, eta_kk)


    
    visib_dual = convert_visib_full(data['RFI_FREE'], pol_flag)
    visib_matrix, UV_redun_conj = calculate_baseline_distribution(ant_pos, visib_dual, 0)

            

    T_dirty1, visib_noredun_dirty_real = inverse_DFT_scale_complex(
        'test_diffusion', D, UV_redun_conj, G_matrix_H_conj, F_real, xi_kk, eta_kk,
        0, AFFOV_idx)

    T_original, visib_noredun_original_real = inverse_DFT_complex(
        'ground_truth', D_original_mod, UV_redun_conj, G_matrix_H_conj, F_real, xi_kk, eta_kk,
        0, AFFOV_idx)



                



