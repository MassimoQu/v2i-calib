import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('./reader')
sys.path.append('./process/corresponding')
sys.path.append('./visualize')
from InfraReader import InfraReader
from VehicleReader import VehicleReader
from CooperativeReader import CooperativeReader
from CorrespondingDetector import CorrespondingDetector
from BBoxVisualizer_open3d_standardized import BBoxVisualizer_open3d_standardized


def plot_hust_distribution(data, str = 'HUST Distribution'):
    # 设置绘图风格
    sns.set(style="whitegrid")
    # 创建一个直方图
    plt.figure(figsize=(10, 6))  # 设置图像大小
    sns.histplot(data, color='blue')  # bins决定了直方图分区数目，kde为核密度估计
    # 设置标题和标签
    plt.title(str, fontsize=15)
    plt.xlabel('Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    # 显示图像
    plt.show()

def plot_boxplot_distribution(data):
    # 创建一个箱型图
    plt.figure(figsize=(8, 6))  # 设置图形的大小
    plt.boxplot(data, meanline=True, showmeans=True)
    # 设置图形的标题和轴标签
    plt.title('Distance Distribution')
    plt.ylabel('Distance')
    # 显示网格
    plt.grid(True)
    # 显示图形
    plt.show()


def plot_distance_distribution_between_two_adjacent_frame(boxes_object_list, last_frame_boxes_object_list):
    '''
    plot the distance distribution between two adjacent frame.
    '''
    distance_list = []
    matches_with_distance = CorrespondingDetector(boxes_object_list, last_frame_boxes_object_list).get_matches_with_score()
    
    # BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene([boxes_object_list, last_frame_boxes_object_list], [], matches_with_score = matches_with_distance)

    for match, distance in matches_with_distance.items():
        distance_list.append(distance)

    # print(distance_list)

    # plot_hust_distribution(distance_list, 'Distance Distribution Between Two Adjacent Frame')
    # plot_boxplot_distribution(distance_list)    


if __name__ == '__main__':
    for i in range(10):
        boxes_object_list = InfraReader('00000' + str(i)).get_infra_boxes_object_list()
        last_frame_boxes_object_list = InfraReader('00000' + str(i + 1)).get_infra_boxes_object_list()
        plot_distance_distribution_between_two_adjacent_frame(boxes_object_list, last_frame_boxes_object_list)
    boxes_object_list = InfraReader('000010').get_infra_boxes_object_list()
    last_frame_boxes_object_list = InfraReader('000011').get_infra_boxes_object_list()
    plot_distance_distribution_between_two_adjacent_frame(boxes_object_list, last_frame_boxes_object_list)

