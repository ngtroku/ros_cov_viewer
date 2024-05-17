#!/usr/bin/env python3

import struct, os, datetime, time
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

def write_csv(coordinate, eigen_score, save_dir, file_name):
    #start_evaluate = time.time()
    xyz = pd.DataFrame(coordinate, columns = ["x","y","z"])
    value = pd.DataFrame(eigen_score, columns=["eigen_value"])
    output_df = pd.concat([xyz, value], axis=1)

    save_path = save_dir + "/" + file_name
    output_df.to_csv(save_path)
    print("{} saved.".format(save_path))
    #print("write csv file:{} sec".format(time.time() - start_evaluate))
    
class Node():
    def __init__(self):
        self.sub1 = rospy.Subscriber('/points_raw', PointCloud2, self.subscriber_pointcloud) #subscriber
        
    def subscriber_pointcloud(self, msg):
        iteration = int(len(msg.data)/18) 

        # block1
        #start_evaluate = time.time()
        hoge = np.frombuffer(msg.data, dtype=np.uint8)
        fuga = hoge.reshape(iteration, 18)
        #print("block1 read_binary:{} sec".format(time.time() - start_evaluate))

        # block2 
        # x, y, z 座標だけあればよい
        # pointcloud processing
        #start_evaluate = time.time()
        x_bin = fuga[:, 0:4]
        y_bin = fuga[:, 4:8]
        z_bin = fuga[:, 8:12]
        #print("block2 set_coordinate:{} sec".format(time.time() - start_evaluate))

        # block3
        #start_evaluate = time.time() 
        p = Pool(processes=3)

        bin_list = [x_bin, y_bin, z_bin]
        float_array = (p.map(array2float, bin_list))

        x = float_array[0]
        y = float_array[1]
        z = float_array[2]

        p.close()
        p.join()

        #print("block3 convert bin to array:{} sec".format(time.time() - start_evaluate))

        coordinate_array = np.vstack((x, y, z)).T
        coordinate, score = registration.execute_gicp(coordinate_array)
        print(len(coordinate))
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
