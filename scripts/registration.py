import numpy as np
import pandas as pd
import open3d as o3d
import small_gicp
import sys

def random_sampling(array, sample_rate=0.5): # random sampling

    num_sample = int(array.shape[0] * sample_rate)

    indices_1 = np.random.choice(array.shape[0], num_sample, replace=False)
    sampled_array1 = array[indices_1]

    indices_2 = np.random.choice(array.shape[0], num_sample, replace=False)
    sampled_array2 = array[indices_2]

    return sampled_array1, sampled_array2

def points_noise(array, scale_translation=1):
    rng = np.random.default_rng()

    # x, y, z 軸方向の各点に正規分布からサンプリングされたノイズを定義
    noise_x = rng.normal(0, scale_translation, array.shape[0])
    noise_y = rng.normal(0, scale_translation, array.shape[0])
    noise_z = rng.normal(0, scale_translation, array.shape[0])

    # ノイズを与える
    array_noised = array.copy()
    array_noised[:, 0] += noise_x
    array_noised[:, 1] += noise_y
    array_noised[:, 2] += noise_z

    return array_noised

def calc_factor(source_points, target_points):
    # update : ヘッセ行列を直接使用(2024 5/8)

    source, source_tree = small_gicp.preprocess_points(source_points, downsampling_resolution=0.5)
    target, target_tree = small_gicp.preprocess_points(target_points, downsampling_resolution=0.5)

    result = small_gicp.align(target, source, target_tree)
    result = small_gicp.align(target, source, target_tree, result.T_target_source)

    factors = [small_gicp.GICPFactor()]
    rejector = small_gicp.DistanceRejector()

    # initialize
    sum_H = np.zeros((6, 6))
    sum_b = np.zeros(6)
    sum_e = 0.0
    
    eigen_value, eigen_vector = np.linalg.eig(result.H)
    global_max_value = np.argmax(eigen_value)
    global_max_vector = eigen_vector[:, global_max_value] # 最も拘束が弱い方向

    list_xyz = []
    list_cov_eigen_value = []

    for i in range(source.size()):
        succ, H, b, e = factors[0].linearize(target, source, target_tree, result.T_target_source, i, rejector)
        if succ:
            point_eigen_value, point_eigen_vector = np.linalg.eig(H) #各点の固有値、固有ベクトルを求める
            local_min_value = np.argmin(eigen_value) 
            local_min_vector = point_eigen_vector[:, local_min_value] # 最も拘束が強い固有ベクトル
            naiseki = np.dot(global_max_vector, local_min_vector)
            list_cov_eigen_value.append((naiseki.real + 1) / 2) # 値を0から1に範囲に制限
            list_xyz.append(source_points[i])

            sum_H += H
            sum_b += b
            sum_e += e
    
    return np.array(list_xyz), np.array(list_cov_eigen_value)

def execute_gicp(array, num_iteration=2):
    counter = 1
    pc1, pc2 = random_sampling(array)

    while counter <= int(num_iteration):

        if counter == 1:
            #print("iteration:{}/{}".format(counter, num_iteration))
            source, target = pc1, points_noise(pc2)
            coordinate_origin, dot_eigen_value_origin = calc_factor(source, target)
            counter += 1

        else:
            #print("iteration:{}/{}".format(counter, num_iteration))
            source, target = pc1, points_noise(pc2)
            coordinate, dot_eigen_value = calc_factor(source, target)
            coordinate_origin = np.concatenate([coordinate_origin, coordinate])
            dot_eigen_value_origin = np.concatenate([dot_eigen_value_origin, dot_eigen_value])
            counter += 1
    
    return coordinate_origin, dot_eigen_value_origin

