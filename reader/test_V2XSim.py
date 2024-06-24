import os
import copy
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from collections import OrderedDict
import pickle
import numpy as np
import os
from BBox3d import BBox3d
import sys
sys.path.append('/home/massimo/vehicle_infrastructure_calibration/process/utils')
sys.path.append('/home/massimo/vehicle_infrastructure_calibration/visualize')
from extrinsic_utils import implement_T_3dbox_object_list, convert_T_to_6DOF, convert_6DOF_to_T, implement_T_3dbox_n_8_3
import noise_utils
from BBoxVisualizer_open3d_standardized import BBoxVisualizer_open3d_standardized
import signal


class SimpleDataset(Dataset):
    """Simple dataset to read the generated pickle files for V2X-Sim dataset
    """
    def __init__(self,
                 root_dir, from_scene, to_scene):
        """constructor

        Args:
            root_dir (string): the root directory where the pickle files live
            from_scene (int): load pickle files starting from which scene
            to_scene (int): load pickle files until which scene (inclusive)
        """

        self.keyframe_database = OrderedDict()
        self.len_record = 0

        for scene_idx in range(from_scene, to_scene + 1):
            pickle_file = os.path.join(root_dir, f'{scene_idx}.pkl')
            if not os.path.exists(pickle_file):
                raise Exception(f'No pickle file found for scene {scene_idx}.')
        
            with open(pickle_file, 'rb') as f:
                dataset_infos = pickle.load(f)  # dataset_infos is a list 

            self.max_cav = 5

            self.len_record += len(dataset_infos)
            agent_start = eval(min([i[-1] for i in dataset_infos[0].keys() if i.startswith("lidar_pose")]))
            self.agent_start = agent_start

            # loop over all keyframe.
            # data_info is one sample.
            for (i, data_info) in enumerate(dataset_infos):
                frame_idx = i + (scene_idx - from_scene) * 100
                self.keyframe_database.update({frame_idx: OrderedDict()})

                # at least 1 cav should show up
                cav_num = data_info['agent_num']
                assert cav_num > 0

                # in one keyframe, loop all agent
                for cav_id in range(agent_start, cav_num+agent_start):

                    self.keyframe_database[frame_idx][cav_id] = OrderedDict()
                    self.keyframe_database[frame_idx][cav_id]['lidar'] = data_info[f'lidar_path_{cav_id}']  # maybe add camera in the future
                    self.keyframe_database[frame_idx][cav_id]['params'] = OrderedDict()
                    self.keyframe_database[frame_idx][cav_id]['params']['lidar_pose'] = data_info[f"lidar_pose_{cav_id}"]

                    if cav_id == agent_start:
                        # let ego load the gt box, gt box is [x,y,z,dx,dy,dz,w,a,b,c]
                        self.keyframe_database[frame_idx][cav_id]['params']['vehicles'] = data_info['gt_boxes_global']
                        self.keyframe_database[frame_idx][cav_id]['params']['object_ids'] = data_info['gt_object_ids'].tolist()
                        self.keyframe_database[frame_idx][cav_id]['params']['sample_token'] = data_info['token']


    ### rewrite __len__ ###
    def __len__(self):
        return self.len_record

    ### rewrite retrieve_base_data ###
    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
            lidar_np: (N, 4)
        """
        # we loop the accumulated length list to see get the scenario index
        keyframe = self.keyframe_database[idx]

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in keyframe.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['params'] = cav_content['params'] # lidar_pose, vehicles(gt_boxes), object_id(token)

            # load the corresponding data into the dictionary
            nbr_dims = 4 # x,y,z,intensity
            scan = np.fromfile(cav_content['lidar'], dtype='float32')
            points = scan.reshape((-1, 5))[:, :nbr_dims] 
            data[cav_id]['lidar_np'] = points

        return data

    def __getitem__(self, index):
        return self.retrieve_base_data(index)


def generate_object_corners_v2x(cav_contents,
                               reference_lidar_pose):
        """
        v2x-sim dataset

        Retrieve all objects (gt boxes)

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            In fact, only the ego vehile needs to generate object center

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (n, 8, 3).
        object_ids : list
            Length is number of bbx in current sample.
        """
        # from opencood.data_utils.datasets import GT_RANGE
        max_num = 200
        gt_boxes = cav_contents[0]['params']['vehicles'] # notice [N,10], 10 includes [x,y,z,dx,dy,dz,w,a,b,c]
        object_ids = cav_contents[0]['params']['object_ids']
        
        object_dict = {"gt_boxes": gt_boxes, "object_ids":object_ids}

        output_dict = {}
        lidar_range = (-32,-32,-3,32,32,2)
        x_min, y_min, z_min, x_max, y_max, z_max = lidar_range
        
        gt_boxes = object_dict['gt_boxes']
        object_ids = object_dict['object_ids']
        for i, object_content in enumerate(gt_boxes):
            x,y,z,dx,dy,dz,w,a,b,c = object_content

            q = Quaternion([w,a,b,c])
            T_world_object = q.transformation_matrix
            T_world_object[:3,3] = object_content[:3]

            T_world_lidar = reference_lidar_pose

            object2lidar = np.linalg.solve(T_world_lidar, T_world_object) # T_lidar_object

            # shape (3, 8)
            x_corners = dx / 2 * np.array([ 1,  1, -1, -1,  1,  1, -1, -1]) # (8,)
            y_corners = dy / 2 * np.array([-1,  1,  1, -1, -1,  1,  1, -1])
            z_corners = dz / 2 * np.array([-1, -1, -1, -1,  1,  1,  1,  1])

            bbx = np.vstack((x_corners, y_corners, z_corners)) # (3, 8)

            # bounding box under ego coordinate shape (4, 8)
            bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

            # project the 8 corners to world coordinate
            bbx_lidar = np.dot(object2lidar, bbx).T # (8, 4)
            bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0) # (1, 8, 3)

            bbox_corner = copy.deepcopy(bbx_lidar)

            center = np.mean(bbox_corner, axis=1)[0]

            if (center[0] > x_min and center[0] < x_max and 
               center[1] > y_min and center[1] < y_max and 
               center[2] > z_min and center[2] < z_max) or i==3:
                output_dict.update({object_ids[i]: bbox_corner})

        object_np = np.zeros((max_num, 8, 3))
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            object_ids.append(object_id)

        # should not appear repeated items
        object_np = object_np[:len(object_ids)]

        return object_np, object_ids


class V2XSim_Reader:
    '''

    1 'params' : { 'lidar_pose' : [4,4] numpy array, 'vehicles' : [N,10] numpy array, 'object_ids' : int, 'sample_token' : str, lidar_np : [N,4] numpy array }
    2-5 'params' : { 'lidar_pose' : [4,4] numpy array, 'lidar_np' : [N,4] numpy array } 

    '''

    def __init__(self, root_dir = '/home/massimo/coperception/v2xsim2_info'):
        self.dataset = SimpleDataset(root_dir=root_dir, from_scene=0, to_scene=0)

    def get_3dbbox_object_list(self, bboxes_list):
        bbox3d_object_list = []
        for bbox in bboxes_list:
            bbox3d_object_list.append(BBox3d('car', bbox))
        return bbox3d_object_list

    def get_noised_3dbbox_object_list(self, T, bboxes_object_list, noise_type='gaussian', noise={'pos_std':0.2, 'rot_std':0.2, 'pos_mean':0, 'rot_mean':0}):
        
        noised_bbox3d_object_list = []
        for bbox3d_object in bboxes_object_list:
            if noise_type == 'gaussian':
                noise_6DOF = noise_utils.generate_noise(noise['pos_std'], noise['rot_std'], noise['pos_mean'], noise['rot_mean'])
            elif noise_type == 'laplace':
                pos_b = noise['pos_std'] / np.sqrt(2)
                rot_b = noise['rot_std'] / np.sqrt(2)
                noise_6DOF = noise_utils.generate_noise_laplace(pos_b, rot_b, noise['pos_mean'], noise['rot_mean'])
            elif noise_type == 'von_mises':
                noise_6DOF = noise_utils.generate_noise_von_mises(noise['pos_std'], noise['rot_std'], noise['pos_mean'], noise['rot_mean'])

            T_noised_6DOF = convert_T_to_6DOF(T) + noise_6DOF
            noised_bbox3d_object_list.append(BBox3d('car', implement_T_3dbox_n_8_3(convert_6DOF_to_T(T_noised_6DOF), bbox3d_object.get_bbox3d_8_3()), bbox3d_object.get_bbox2d_4(), bbox3d_object.get_occluded_state(), bbox3d_object.get_truncated_state(), bbox3d_object.get_alpha(), bbox3d_object.get_confidence()))

        # print('noise', noise)
        # print('noise_6DOF', noise_6DOF)
        # print(f'T_6DOF: {convert_T_to_6DOF(T)}, noise_6DOF: {noise_6DOF}')
        # print('T_noised_6DOF: ', T_noised_6DOF)

        # return implement_T_3dbox_object_list(convert_6DOF_to_T(T_noised_6DOF), bboxes_object_list)
        return noised_bbox3d_object_list


    def generate_vehicle_vehicle_bboxes_object_list(self, noise_type='gaussian', noise={'pos_std':0.2, 'rot_std':0.2, 'pos_mean':0, 'rot_mean':0}):

        def signal_handler(sig, frame):
            print("Exiting visualization...")
            global should_exit
            should_exit = True

        signal.signal(signal.SIGINT, signal_handler)
        global should_exit
        should_exit = False

        for frame_idx in range(len(self.dataset)):

            if should_exit:
                break

            base_data_dict = self.dataset[frame_idx]
            cav_content = base_data_dict[1]
            # T_lidar_world = cav_content['params']['lidar_pose']
            T_world_lidar = cav_content['params']['lidar_pose']
            bounding_boxes_lidar, _ = generate_object_corners_v2x([cav_content], T_world_lidar)
            bbox3d_object_list_lidar = self.get_3dbbox_object_list(bounding_boxes_lidar)

            cav_ids = list(base_data_dict.keys())

            # print('car_ids: ', cav_ids)

            for cav_id1 in range(len(cav_ids) - 1):

                if cav_id1 == 1: # 多车端在不限制每辆车视野的情况下没有意义，只取第一辆车 ，这样scene0 有100个场景，每个场景1次用掉
                    break

                if should_exit:
                    break

                cav_id2 = cav_id1 + 1
                cav_content1 = base_data_dict[cav_ids[cav_id1]]
                cav_content2 = base_data_dict[cav_ids[cav_id2]]
                T_world_lidar1 = np.array(cav_content1['params']['lidar_pose'])
                T_world_lidar2 = np.array(cav_content2['params']['lidar_pose'])

                T_lidar1_lidar = np.linalg.inv(np.linalg.solve(T_world_lidar, T_world_lidar1))
                T_lidar2_lidar = np.linalg.inv(np.linalg.solve(T_world_lidar, T_world_lidar2))
                T_lidar2_lidar1 = np.dot(T_lidar2_lidar, np.linalg.inv(T_lidar1_lidar))

                if noise_type == None:
                    bbox3d_object_list_lidar1 = implement_T_3dbox_object_list(T_lidar1_lidar, bbox3d_object_list_lidar)
                    bbox3d_object_list_lidar2 = implement_T_3dbox_object_list(T_lidar2_lidar, bbox3d_object_list_lidar)
                else:
                    bbox3d_object_list_lidar1 = self.get_noised_3dbbox_object_list(T_lidar1_lidar, bbox3d_object_list_lidar, noise_type, noise)
                    bbox3d_object_list_lidar2 = self.get_noised_3dbbox_object_list(T_lidar2_lidar, bbox3d_object_list_lidar, noise_type, noise)
                
                # converted_bbox3d_object_list_lidar1 = implement_T_3dbox_object_list(T_lidar2_lidar1, bbox3d_object_list_lidar1)

                # BBoxVisualizer_open3d_standardized('compare_original_converted', ['original', 'converted'], 2).visualize_matches_under_dual_true_predicted_scene(
                #     [bbox3d_object_list_lidar1, bbox3d_object_list_lidar2], [converted_bbox3d_object_list_lidar1, bbox3d_object_list_lidar2], [], [], {}, {})

                yield frame_idx, cav_ids[cav_id1], bbox3d_object_list_lidar1, bbox3d_object_list_lidar2, T_lidar2_lidar1
                

    def generate_gt_and_noised_bboxes_object_list(self, noise_type='gaussian', noise={'pos_std':0, 'rot_std':0, 'pos_mean':0, 'rot_mean':0}):

        for frame_idx in range(len(self.dataset)):
            base_data_dict = self.dataset[frame_idx]
            cav_content = base_data_dict[1]
            T_world_lidar = cav_content['params']['lidar_pose']
            bounding_boxes_lidar, _ = generate_object_corners_v2x([cav_content], T_world_lidar)
            bbox3d_object_list_lidar = self.get_3dbbox_object_list(bounding_boxes_lidar)
            bbox3d_object_list_world = implement_T_3dbox_object_list(T_world_lidar, bbox3d_object_list_lidar)
            # if noise['pos_std'] == 0 and noise['rot_std'] == 0:
            #     noised_bbox3d_object_list_world = bbox3d_object_list_world.copy()
            # else:
            #     noised_bbox3d_object_list_world = self.get_noised_3dbbox_object_list(T_world_lidar, bbox3d_object_list_lidar, noise_type, noise)
            noised_bbox3d_object_list_world = self.get_noised_3dbbox_object_list(T_world_lidar, bbox3d_object_list_lidar, noise_type, noise)
            # BBoxVisualizer_open3d_standardized('compare_gt_noised', ['gt', 'noised'], 2).visualize_matches_under_dual_true_predicted_scene(
            #     [bbox3d_object_list_world, noised_bbox3d_object_list_world], [bbox3d_object_list_world, noised_bbox3d_object_list_world], [], [], {}, {})
            
            cav_ids = list(base_data_dict.keys())
            yield frame_idx, cav_ids[0], bbox3d_object_list_world, noised_bbox3d_object_list_world


if __name__ == "__main__":
    reader = V2XSim_Reader() 
    # reader.generate_vehicle_vehicle_bboxes_object_list()
    reader.generate_gt_and_noised_bboxes_object_list()

