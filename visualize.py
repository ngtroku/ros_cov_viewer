#!/usr/bin/env python3

import os, re
import rospy
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

def downsample_visualize(x, y, z, eigenvalue, rate=1.0):

    num_sample = int(x.shape[0] * rate)

    indices_1 = np.random.choice(x.shape[0], num_sample, replace=False)
    x_downsampled = x[indices_1]
    y_downsampled = y[indices_1]
    z_downsampled = z[indices_1]
    eigenvalue_downsampled = eigenvalue[indices_1]

    return x_downsampled, y_downsampled, z_downsampled, eigenvalue_downsampled

#dir_name = "/home/rokuto/result_219267"
dir_name = rospy.get_param('directory_name', '/home')
rosbag_speed_rate = float(rospy.get_param('rosbag_run_speed', 1.0))
FILES = sort_files(dir_name)

# set gigure
fig = plt.figure(figsize=(12, 6))
plt.subplots_adjust(wspace=0.5)

ax1 = fig.add_subplot(1, 3, 1)

ax2 = fig.add_subplot(1, 3, 2)

ax3 = fig.add_subplot(1, 3, 3)

for file in FILES:
    ax1.cla() # initialize
    ax2.cla()
    ax3.cla()
    path = dir_name + "/" + str(file)
    array_points, array_values = csv_2_array(path)

    # pre process to visualize
    x, y, z, eigen_value = downsample_visualize(array_points[:, 0], array_points[:, 1], array_points[:, 2], array_values)
    list_angle, list_score = count_eigen_score(calc_angle(array_points), array_values, 10) # 角度情報, 重要度情報, 水平方位角範囲の広さ(deg)

    # visualize
    ax1.scatter(y, x, c = eigen_value, cmap = "jet")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("bird eye plot")

    ax2.hist(calc_angle(array_points), bins=15)
    ax2.set_title("points histgram")
    ax2.set_xticks([60 * i for i in range(7)])
    ax2.set_xlabel("angle (deg)")
    ax2.set_ylabel("# of points")

    ax3.plot(list_angle, list_score, marker="o")
    ax3.set_xticks([60 * i for i in range(7)])
    ax3.set_xlabel("angle (deg)")
    ax3.set_ylabel("score")
    ax3.set_title("score distribution")

    plt.pause(0.1)

plt.show()