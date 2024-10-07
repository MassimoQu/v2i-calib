import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./reader')
sys.path.append('./process/corresponding')
sys.path.append('./process/utils')

from CooperativeReader import CooperativeReader
from CooperativeBatchingReader import CooperativeBatchingReader
from CorrespondingDetector import CorrespondingDetector
from extrinsic_utils import implement_T_3dbox_object_list, convert_6DOF_to_T

# x y z roll pitch yaw
def plot_curve_oIoU_under_trueT_with_bias(infra_id, vehicle_id, cnt, step):
    # 读取数据
    cooperative_reader = CooperativeReader(infra_file_name = str(infra_id).zfill(6), vehicle_file_name = str(vehicle_id).zfill(6))
    infra_bbox_list, vehicle_bbox_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_lists_vehicle_coordinate()
    oIoU_list = {}
    key_list = ['x', 'y', 'z', 'roll', 'pitch', 'yaw'] 
    extrinsic_unit = {}
    identity6 = np.identity(6)
    for i, key in enumerate(key_list):
        oIoU_list[key] = []
        extrinsic_unit[key] = identity6[i] * 0.1

    # 计算oIoU   
    for i in range(-cnt - 1, cnt + 1):
        for key in key_list:
            converted_infra_bbox_list = implement_T_3dbox_object_list(convert_6DOF_to_T(extrinsic_unit[key] * i), infra_bbox_list)
            oIoU = CorrespondingDetector(converted_infra_bbox_list, vehicle_bbox_list, core_similarity_component='iou').get_Yscore()
            oIoU_list[key].append(oIoU)

    # Create two figures for separate plots
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    # Define colors for better visibility
    colors_xyz = ['blue', 'green', 'red']
    colors_rpy = ['cyan', 'magenta', 'yellow']

    # Plot x, y, z with 'm' unit
    for key, color in zip(['x', 'y', 'z'], colors_xyz):
        bias_levels = [step * i for i in range(-cnt - 1, cnt + 1)]
        ax1.plot(bias_levels, oIoU_list[key], label=key.capitalize(), color=color)
        ax1.axvline(0, color='black', linestyle='--')  # Add a vertical dashed line at x=0 for zero bias

    # Plot roll, pitch, yaw with '°' unit
    for key, color in zip(['roll', 'pitch', 'yaw'], colors_rpy):
        bias_levels = [step * i for i in range(-cnt - 1, cnt + 1)]
        ax2.plot(bias_levels, oIoU_list[key], label=key.capitalize(), color=color)
        ax2.axvline(0, color='black', linestyle='--')  # Add a vertical dashed line at x=0 for zero bias

    # Add legends
    ax1.legend()
    ax2.legend()

    # Add titles and labels
    ax1.set_title('oIoU for Translational Motion Under True T with Bias')
    ax1.set_xlabel('Bias (m)')
    ax1.set_ylabel('oIoU')
    
    ax2.set_title('oIoU for Rotational Motion Under True T with Bias')
    ax2.set_xlabel('Bias (°)')
    ax2.set_ylabel('oIoU')

    # Show the plots
    plt.show()


def plot_curve_overall_distance_under_trueT_with_bias(infra_id, vehicle_id, cnt, step):
    # 读取数据
    cooperative_reader = CooperativeReader(infra_file_name = str(infra_id).zfill(6), vehicle_file_name = str(vehicle_id).zfill(6))
    infra_bbox_list, vehicle_bbox_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_lists_vehicle_coordinate()
    odistance_list = {}
    key_list = ['x', 'y', 'z', 'roll', 'pitch', 'yaw'] 
    extrinsic_unit = {}
    identity6 = np.identity(6)
    for i, key in enumerate(key_list):
        odistance_list[key] = []
        extrinsic_unit[key] = identity6[i] * 0.1

    # 计算oIoU   
    for i in range(-cnt - 1, cnt + 1):
        for key in key_list:
            converted_infra_bbox_list = implement_T_3dbox_object_list(convert_6DOF_to_T(extrinsic_unit[key] * i), infra_bbox_list)
            odistance = (CorrespondingDetector(converted_infra_bbox_list, vehicle_bbox_list, core_similarity_component='centerpoint_distance').get_Yscore() + CorrespondingDetector(converted_infra_bbox_list, vehicle_bbox_list, core_similarity_component='vertex_distance').get_Yscore()) / 2
            odistance_list[key].append(odistance)

    # Create two figures for separate plots
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    # Define colors for better visibility
    colors_xyz = ['blue', 'green', 'red']
    colors_rpy = ['cyan', 'magenta', 'yellow']

    # Plot x, y, z with 'm' unit
    for key, color in zip(['x', 'y', 'z'], colors_xyz):
        bias_levels = [step * i for i in range(-cnt - 1, cnt + 1)]
        ax1.plot(bias_levels, odistance_list[key], label=key.capitalize(), color=color)
        ax1.axvline(0, color='black', linestyle='--')  # Add a vertical dashed line at x=0 for zero bias

    # Plot roll, pitch, yaw with '°' unit
    for key, color in zip(['roll', 'pitch', 'yaw'], colors_rpy):
        bias_levels = [step * i for i in range(-cnt - 1, cnt + 1)]
        ax2.plot(bias_levels, odistance_list[key], label=key.capitalize(), color=color)
        ax2.axvline(0, color='black', linestyle='--')  # Add a vertical dashed line at x=0 for zero bias

    # Add legends
    ax1.legend()
    ax2.legend()

    # Add titles and labels
    ax1.set_title('odistance for Translational Motion Under True T with Bias')
    ax1.set_xlabel('Bias (m)')
    ax1.set_ylabel('odistance')
    
    ax2.set_title('odistance for Rotational Motion Under True T with Bias')
    ax2.set_xlabel('Bias (°)')
    ax2.set_ylabel('odistance')

    # Show the plots
    plt.show()


def plot_combined_motion_and_metrics(infra_id, vehicle_id, cnt, step):
    cooperative_reader = CooperativeReader(infra_file_name = str(infra_id).zfill(6), vehicle_file_name = str(vehicle_id).zfill(6))
    infra_bbox_list, vehicle_bbox_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_lists_vehicle_coordinate()
    oIoU_data = {}
    oDistance_data = {}
    extrinsic_unit = {}
    key_list = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    identity6 = np.identity(6)
    for i, key in enumerate(key_list):
        oIoU_data[key] = []
        oDistance_data[key] = []
        extrinsic_unit[key] = identity6[i] * 0.1
    
    for i in range(-cnt, cnt + 1):
        for key in key_list:
            converted_infra_bbox_list = implement_T_3dbox_object_list(convert_6DOF_to_T(extrinsic_unit[key] * i), infra_bbox_list)
            oIoU = CorrespondingDetector(converted_infra_bbox_list, vehicle_bbox_list, core_similarity_component='iou').get_Yscore()
            oDistance = (CorrespondingDetector(converted_infra_bbox_list, vehicle_bbox_list, core_similarity_component='centerpoint_distance').get_Yscore() + CorrespondingDetector(converted_infra_bbox_list, vehicle_bbox_list, core_similarity_component='vertex_distance').get_Yscore()) / 2
            oIoU_data[key].append(oIoU)
            oDistance_data[key].append(oDistance)

    # Setup figure and subplot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define harmonious colors
    colors = {'x': 'darkorange', 'y': 'royalblue', 'yaw': 'forestgreen'}
    
    # Plot for translational motion x and y, and rotational motion yaw
    keys = ['x', 'y', 'yaw']
    for key in keys:
        bias_levels = np.arange(-cnt, cnt + 1) * step
        ax.plot(bias_levels, oIoU_data[key], label=f'oIoU {key.capitalize()}', color=colors[key], linestyle='--')
        ax.plot(bias_levels, oDistance_data[key], label=f'oDistance {key.capitalize()}', color=colors[key], linestyle='-')
        ax.axvline(0, color='black', linestyle='--')

    # Adding labels, legends, and title
    ax.set_xlabel('Bias')
    ax.set_ylabel('Metrics')
    ax.set_title('oIoU and oDistance for Selected Motions')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


def plot_separate_motion_metrics(infra_id, vehicle_id, cnt, step):
    cooperative_reader = CooperativeReader(infra_file_name=str(infra_id).zfill(6), vehicle_file_name=str(vehicle_id).zfill(6))
    infra_bbox_list, vehicle_bbox_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_lists_vehicle_coordinate()
    oIoU_data = {}
    oDistance_data = {}
    extrinsic_unit = {}
    key_list = ['x', 'y', 'yaw']  # 限制为 x, y, yaw
    identity6 = np.identity(6)
    for i, key in enumerate(key_list):
        oIoU_data[key] = []
        oDistance_data[key] = []
        extrinsic_unit[key] = identity6[i] * 0.1

    for i in range(-cnt, cnt + 1):
        for key in key_list:
            converted_infra_bbox_list = implement_T_3dbox_object_list(convert_6DOF_to_T(extrinsic_unit[key] * i), infra_bbox_list)
            oIoU = CorrespondingDetector(converted_infra_bbox_list, vehicle_bbox_list, core_similarity_component='iou').get_Yscore()
            oDistance = (CorrespondingDetector(converted_infra_bbox_list, vehicle_bbox_list, core_similarity_component='centerpoint_distance').get_Yscore() +
                         CorrespondingDetector(converted_infra_bbox_list, vehicle_bbox_list, core_similarity_component='vertex_distance').get_Yscore()) / 2
            oIoU_data[key].append(oIoU)
            oDistance_data[key].append(oDistance)

    # 定义色彩
    colors = {'x': 'darkorange', 'y': 'royalblue', 'yaw': 'forestgreen'}

    # 分别绘制每个键的数据
    for key in key_list:
        fig, ax = plt.subplots(figsize=(10, 6))
        bias_levels = np.arange(-cnt, cnt + 1) * step
        ax.plot(bias_levels, oIoU_data[key], label=f'oIoU {key.capitalize()}', color=colors[key], linestyle='--')
        ax.plot(bias_levels, oDistance_data[key], label=f'oDistance {key.capitalize()}', color=colors[key], linestyle='-')

        ax.set_xlabel('Bias')
        ax.set_ylabel('Metrics')
        ax.set_title(f'oIoU and oDistance for Motion {key.capitalize()}')
        ax.legend(loc='upper right')
        ax.axvline(0, color='black', linestyle='--')

        plt.tight_layout()
        plt.show()
        


def plot_curve_oIoU_under_trueT_different_scene(frame_cnt = 100):
    batch_reader = CooperativeBatchingReader(path_data_info=r'/home/massimo/vehicle_infrastructure_calibration/data/cooperative-vehicle-infrastructure/cooperative/sorted_data_info.json')
    
    oIoU_list = {}
    key_list = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    for i, key in enumerate(key_list):
        oIoU_list[key] = []

    cnt = 0

    for infra_file_name, vehicle_file_name, infra_bboxes_object_list, vehicle_bboxes_object_list, T_i2v in batch_reader.generate_infra_vehicle_bboxes_object_list(start_idx=0, end_idx=-1):
        if cnt >= frame_cnt:
            break
        cnt += 1

        for key in key_list:
            converted_infra_bbox_list = implement_T_3dbox_object_list(T_i2v, infra_bboxes_object_list)
            oIoU = CorrespondingDetector(converted_infra_bbox_list, vehicle_bboxes_object_list, core_similarity_component='iou').get_Yscore()
            oIoU_list[key].append(oIoU)

    # Create two figures for separate plots
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    # Define colors for better visibility
    colors_xyz = ['blue', 'green', 'red']
    colors_rpy = ['cyan', 'magenta', 'yellow']

    # Plot x, y, z with 'm' unit
    for key, color in zip(['x', 'y', 'z'], colors_xyz):
        bias_levels = [i for i in range(frame_cnt)]
        ax1.plot(bias_levels, oIoU_list[key], label=key.capitalize(), color=color)

    # Plot roll, pitch, yaw with '°' unit
    for key, color in zip(['roll', 'pitch', 'yaw'], colors_rpy):
        bias_levels = [i for i in range(frame_cnt)]
        ax2.plot(bias_levels, oIoU_list[key], label=key.capitalize(), color=color)

    # Add legends
    ax1.legend()
    ax2.legend()

    # Add titles and labels
    ax1.set_title('oIoU for Translational Motion Under True T in consecutive frames')
    ax1.set_xlabel('Count')
    ax1.set_ylabel('oIoU')

    ax2.set_title('oIoU for Rotational Motion Under True T in consecutive frames')
    ax2.set_xlabel('Count')
    ax2.set_ylabel('oIoU')

    # Show the plots
    plt.show()


# Define the function to plot 3D curves
def plot_3d_curve_oIoU_under_trueT_with_bias_different_scene(infra_ids, vehicle_ids, cnt, step):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')

    colors_xyz = ['blue', 'green', 'red']
    colors_rpy = ['cyan', 'magenta', 'yellow']

    identity6 = np.identity(6)
    key_back_links = {'x': 0, 'y': 1, 'z': 2, 'roll': 3, 'pitch': 4, 'yaw': 5}

    # Loop over different scenes, represented by different infra_ids and vehicle_ids
    for scene_index, (infra_id, vehicle_id) in enumerate(zip(infra_ids, vehicle_ids)):
        cooperative_reader = CooperativeReader(infra_file_name=str(infra_id).zfill(6), 
                                               vehicle_file_name=str(vehicle_id).zfill(6))
        infra_bbox_list, vehicle_bbox_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_lists_vehicle_coordinate()
        oIoU_list = {key: [] for key in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']}
        

        for i in range(-cnt, cnt + 1):
            for key in oIoU_list.keys():
                bias = identity6[key_back_links[key]] * i * step
                converted_infra_bbox_list = implement_T_3dbox_object_list(convert_6DOF_to_T(bias), infra_bbox_list)
                oIoU = CorrespondingDetector(converted_infra_bbox_list, vehicle_bbox_list).get_Yscore()
                oIoU_list[key].append(oIoU)

        # Plotting translational motion for this scene
        for key, color in zip(['x', 'y', 'z'], colors_xyz):
            ax1.plot([scene_index] * (2 * cnt + 1), range(-cnt, cnt + 1), oIoU_list[key], color=color)

        # Plotting rotational motion for this scene
        for key, color in zip(['roll', 'pitch', 'yaw'], colors_rpy):
            ax2.plot([scene_index] * (2 * cnt + 1), range(-cnt, cnt + 1), oIoU_list[key], color=color)

    ax1.set_xticks(range(len(infra_ids)))
    ax2.set_xticks(range(len(infra_ids)))

    ax1.set_xlabel('Scene')
    ax1.set_ylabel('Bias (m)')
    ax1.set_zlabel('oIoU')
    ax1.set_title('3D Plot of oIoU for Translational Motion Under True T with Bias Across Scenes')

    ax2.set_xlabel('Scene')
    ax2.set_ylabel('Bias (°)')
    ax2.set_zlabel('oIoU')
    ax2.set_title('3D Plot of oIoU for Rotational Motion Under True T with Bias Across Scenes')

    plt.show()


def plot_3d_curve_oIoU_under_trueT_with_bias_different_scene_test(infra_ids, vehicle_ids, cnt, step):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')

    colors_xyz = ['blue', 'green', 'red']
    colors_rpy = ['cyan', 'magenta', 'yellow']
    identity6 = np.identity(6)
    key_back_links = {'x': 0, 'y': 1, 'z': 2, 'roll': 3, 'pitch': 4, 'yaw': 5}

    # Turn off the grid
    ax1.grid(False)
    ax2.grid(False)

    # Loop over different scenes, represented by different infra_ids and vehicle_ids
    for scene_index, (infra_id, vehicle_id) in enumerate(zip(infra_ids, vehicle_ids)):
        cooperative_reader = CooperativeReader(infra_file_name=str(infra_id).zfill(6), 
                                               vehicle_file_name=str(vehicle_id).zfill(6))
        infra_bbox_list, vehicle_bbox_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_lists_vehicle_coordinate()
        oIoU_list = {key: [] for key in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']}
        
        for i in range(-cnt, cnt + 1):
            for key in oIoU_list.keys():
                bias = identity6[key_back_links[key]] * i * step
                converted_infra_bbox_list = implement_T_3dbox_object_list(convert_6DOF_to_T(bias), infra_bbox_list)
                oIoU = CorrespondingDetector(converted_infra_bbox_list, vehicle_bbox_list).get_Yscore()
                oIoU_list[key].append(oIoU)

        # Plotting translational motion for this scene
        for key, color in zip(['x', 'y', 'z'], colors_xyz):
            bias_levels = np.array([step * i for i in range(-cnt, cnt + 1)])
            ax1.plot(np.full(bias_levels.shape, scene_index + 1), bias_levels, oIoU_list[key], color=color)
            if scene_index == 0:  # Only add label once
                ax1.set_xlabel('Scene')
                ax1.set_ylabel('Bias (m)')
                ax1.set_zlabel('oIoU')
                ax1.set_title('3D Plot of oIoU for Translational Motion Under True T with Bias Across Scenes')
        
        # Draw bias=0 line for each scene in the x-z plane
        ax1.plot(np.full(2, scene_index + 1), [0, 0], [min(min(oIoU_list.values())), max(max(oIoU_list.values()))], 'k--')

        # Plotting rotational motion for this scene
        for key, color in zip(['roll', 'pitch', 'yaw'], colors_rpy):
            bias_levels = np.array([step * i for i in range(-cnt, cnt + 1)])
            ax2.plot(np.full(bias_levels.shape, scene_index + 1), bias_levels, oIoU_list[key], color=color)
            if scene_index == 0:  # Only add label once
                ax2.set_xlabel('Scene')
                ax2.set_ylabel('Bias (°)')
                ax2.set_zlabel('oIoU')
                ax2.set_title('3D Plot of oIoU for Rotational Motion Under True T with Bias Across Scenes')

        # Draw bias=0 line for each scene in the x-z plane
        ax2.plot(np.full(2, scene_index + 1), [0, 0], [min(min(oIoU_list.values())), max(max(oIoU_list.values()))], 'k--')

    # Setting the scene numbers as x-ticks
    scene_numbers = list(range(1, len(infra_ids) + 1))
    ax1.set_xticks(scene_numbers)
    ax2.set_xticks(scene_numbers)

    plt.show()


# plot_curve_oIoU_under_trueT_with_bias('003920', '020092', 100, 0.1)

# plot_curve_overall_distance_under_trueT_with_bias('003920', '020092', 100, 0.1)

# plot_combined_motion_and_metrics('003920', '020092', 100, 0.1)
plot_separate_motion_metrics('003920', '020092', 100, 0.1)

# plot_curve_oIoU_under_trueT_different_scene()

# plot_3d_curve_oIoU_under_trueT_with_bias_different_scene(infra_ids=[3920, 7489, 5298], vehicle_ids=[20092, 289, 1374], cnt=10, step=0.1)

# plot_3d_curve_oIoU_under_trueT_with_bias_different_scene_test(infra_ids=[3920, 7489, 5298], vehicle_ids=[20092, 289, 1374], cnt=10, step=0.1)


