import os, re
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt

def sort_files(dir_name):
    list_dir = os.listdir(dir_name)
    return sorted(list_dir, key=lambda s: int(re.search(r'\d+', s).group()))

def csv_2_array(file_name):
    df = pd.read_csv(file_name)
    pointcloud = df[["x", "y", "z"]]
    eigenvalue = df[["eigen_value"]]

    array_points, array_values = np.array(pointcloud), np.array(eigenvalue)

    return array_points, array_values

def calc_angle(xyz):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    angle = np.degrees(np.arctan2(y, x)) + 180
    return angle

def count_eigen_score(angle_array, eigen_array, step):
    list_angle, list_score = [], []
    for i in range(int(360 / step)):
        mask = ((angle_array >= step * i) & (angle_array < step * (i+1)))
        score = np.sum(eigen_array[mask])
        list_score.append(score)
        list_angle.append(step * (i + 0.5))
    
    return list_angle, list_score

def downsample_visualize(x, y, z, eigenvalue, rate=0.25):

    num_sample = int(x.shape[0] * rate)

    indices_1 = np.random.choice(x.shape[0], num_sample, replace=False)
    x_downsampled = x[indices_1]
    y_downsampled = y[indices_1]
    z_downsampled = z[indices_1]
    eigenvalue_downsampled = eigenvalue[indices_1]

    return x_downsampled, y_downsampled, z_downsampled, eigenvalue_downsampled

dir_name = "/home/rokuto/result_15153947"
rosbag_speed_rate = 0.5
FILES = sort_files(dir_name)

# set gigure
fig = plt.figure(figsize=(9, 6))
plt.subplots_adjust(wspace=0.5)

ax1 = fig.add_subplot(1, 2, 1, projection='3d')

ax2 = fig.add_subplot(1, 2, 2)

for file in FILES:
    ax1.cla() # initialize
    ax2.cla()
    path = dir_name + "/" + str(file)
    t = int(file[:-4]) / (1e6 * (1 / rosbag_speed_rate))
    array_points, array_values = csv_2_array(path)

    # pre process to visualize
    x, y, z, eigen_value = downsample_visualize(array_points[:, 0], array_points[:, 1], array_points[:, 2], array_values)
    list_angle, list_score = count_eigen_score(calc_angle(array_points), array_values, 10) # 角度情報, 重要度情報, 水平方位角範囲の広さ(deg)

    # visualize
    ax1.scatter(x, y, z, c = eigen_value, s = 5, cmap = "jet")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("t={}".format(round(t, 3)))

    ax2.plot(list_angle, list_score, marker="o")
    ax2.set_xlabel("angle (deg)")
    ax2.set_ylabel("score")
    ax2.set_title("t={}".format(round(t, 3)))

    plt.pause(0.2)

plt.show()