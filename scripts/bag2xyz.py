#!/usr/bin/env python3

import struct, os, datetime
import registration
from multiprocessing import Pool

import rospy
import numpy as np
import pandas as pd
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

def ros_now():
    return rospy.get_time()

def date():
    dt_now = datetime.datetime.now()
    day = dt_now.day
    hour = dt_now.hour
    minute = dt_now.minute
    second = dt_now.second
    DATE = str(day) + str(hour) + str(minute) + str(second)
    return DATE

def binary2float(data):
    float_value = struct.unpack('<f', data)[0]
    return float_value

def array2float(bin_array):
    float_array = np.apply_along_axis(binary2float, axis=1, arr = bin_array)
    return float_array

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

def write_csv(coordinate, eigen_score, save_dir, file_name):
    xyz = pd.DataFrame(coordinate, columns = ["x","y","z"])
    value = pd.DataFrame(eigen_score, columns=["eigen_value"])
    output_df = pd.concat([xyz, value], axis=1)

    save_path = save_dir + "/" + file_name
    output_df.to_csv(save_path)
    print("{} saved.".format(save_path))
    
class Node():
    def __init__(self):
        self.sub1 = rospy.Subscriber('/velodyne_points', PointCloud2, self.subscriber_pointcloud) #subscriber
        
    def subscriber_pointcloud(self, msg):
        iteration = int(len(msg.data)/22) 

        hoge = np.frombuffer(msg.data, dtype=np.uint8)
        fuga = hoge.reshape(iteration, 22)

        # block2 
        # x, y, z 座標だけあればよい
        # pointcloud processing
        x_bin = fuga[:, 0:4]
        y_bin = fuga[:, 4:8]
        z_bin = fuga[:, 8:12]

        # block3 
        p = Pool(processes=3)

        bin_list = [x_bin, y_bin, z_bin]
        float_array = (p.map(array2float, bin_list))

        x = float_array[0]
        y = float_array[1]
        z = float_array[2]

        p.close()
        p.join()

        coordinate_array = np.vstack((x, y, z)).T
        coordinate, score = registration.execute_gicp(coordinate_array)
        now_timestamp = ros_now()
        # ここから結果をcsvに書き出し
        erapsed = str(round(now_timestamp - start, 6))
        erapsed = erapsed.replace('.', '')
        file_name = erapsed + ".csv"
        write_csv(coordinate, score, save_dir, file_name)

if __name__ == '__main__':
    global start, save_dir
    rospy.init_node('test_node')
    now_date = date()
    save_dir = "/home/rokuto/result_" + now_date # 絶対パス

    # initial setting
    os.mkdir(save_dir)

    start = ros_now()
    node = Node()

    while not rospy.is_shutdown():
        rospy.sleep(0.001)
