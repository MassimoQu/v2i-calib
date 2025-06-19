import os
import copy
import numpy as np
from pyquaternion import Quaternion
from collections import OrderedDict
import pickle
import numpy as np
import signal
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
from v2x_calib.utils import implement_T_3dbox_object_list, convert_T_to_6DOF, convert_6DOF_to_T, implement_T_3dbox_n_8_3, implement_T_to_3dbox_with_own_center
from v2x_calib.reader.BBox3d import BBox3d
import noise_utils
import json
from scipy.spatial.transform import Rotation as R
from visualize import BBoxVisualizer_open3d
# from visualize import BBoxVisualizer_open3d_standardized



class SimpleDataset():
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

            noised_bbox3d_8_3 = implement_T_to_3dbox_with_own_center(convert_6DOF_to_T(noise_6DOF), bbox3d_object.get_bbox3d_8_3())

            noised_bbox3d_object_list.append(BBox3d('car', implement_T_3dbox_n_8_3(T, noised_bbox3d_8_3), bbox3d_object.get_bbox2d_4(), bbox3d_object.get_occluded_state(), bbox3d_object.get_truncated_state(), bbox3d_object.get_alpha(), bbox3d_object.get_confidence()))

        # print('noise', noise)
        # print('noise_6DOF', noise_6DOF)
        # print(f'T_6DOF: {convert_T_to_6DOF(T)}, noise_6DOF: {noise_6DOF}')
        # print('T_noised_6DOF: ', T_noised_6DOF)

        # return implement_T_3dbox_object_list(convert_6DOF_to_T(T_noised_6DOF), bboxes_object_list)
        return noised_bbox3d_object_list


    def generate_vehicle_vehicle_bboxes_object_list(self, noise_type='gaussian', noise={'pos_std':0.2, 'rot_std':0.2, 'pos_mean':0, 'rot_mean':0}):

        for frame_idx in range(len(self.dataset)):

            base_data_dict = self.dataset[frame_idx]
            cav_content = base_data_dict[1]
            # T_lidar_world = cav_content['params']['lidar_pose']
            T_world_lidar = cav_content['params']['lidar_pose']
            bounding_boxes_lidar, _ = generate_object_corners_v2x([cav_content], T_world_lidar)
            bbox3d_object_list_lidar = self.get_3dbbox_object_list(bounding_boxes_lidar)

            cav_ids = list(base_data_dict.keys())


            for cav_id1 in range(len(cav_ids) - 1):

                if cav_id1 == 1: # 多车端在不限制每辆车视野的情况下没有意义，只取第一辆车 ，这样scene0 有100个场景，每个场景1次用掉
                    break

                for cav_id2 in range(cav_id1 + 1, len(cav_ids)):

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
                        # bbox3d_object_list_lidar2 = self.get_noised_3dbbox_object_list(T_lidar2_lidar, bbox3d_object_list_lidar, noise_type, {'pos_std':0.2, 'rot_std':0.2, 'pos_mean':0, 'rot_mean':0})             
                
                    yield frame_idx, cav_ids[cav_id2], bbox3d_object_list_lidar1, bbox3d_object_list_lidar2, T_lidar2_lidar1
                

   
    def generate_vehicle_vehicle_pointcloud(self, noise_type='gaussian', noise={'pos_std':0.2, 'rot_std':0.2, 'pos_mean':0, 'rot_mean':0}):

        for frame_idx in range(len(self.dataset)):

            base_data_dict = self.dataset[frame_idx]
            cav_content = base_data_dict[1]
            # T_lidar_world = cav_content['params']['lidar_pose']
            T_world_lidar = cav_content['params']['lidar_pose']
            cav_ids = list(base_data_dict.keys())

            for cav_id1 in range(len(cav_ids) - 1):

                if cav_id1 == 1: # 多车端在不限制每辆车视野的情况下没有意义，只取第一辆车 ，这样scene0 有100个场景，每个场景1次用掉
                    break
                
                for cav_id2 in range(cav_id1 + 1, len(cav_ids)):

                    cav_content1 = base_data_dict[cav_ids[cav_id1]]
                    cav_content2 = base_data_dict[cav_ids[cav_id2]]
                    T_world_lidar1 = np.array(cav_content1['params']['lidar_pose'])
                    T_world_lidar2 = np.array(cav_content2['params']['lidar_pose'])

                    T_lidar1_lidar = np.linalg.inv(np.linalg.solve(T_world_lidar, T_world_lidar1))
                    T_lidar2_lidar = np.linalg.inv(np.linalg.solve(T_world_lidar, T_world_lidar2))
                    T_lidar2_lidar1 = np.dot(T_lidar2_lidar, np.linalg.inv(T_lidar1_lidar))

                    pointcloud1 = cav_content1['lidar_np']
                    pointcloud2 = cav_content2['lidar_np']

                    yield frame_idx, cav_ids[cav_id2], pointcloud1[:,:3], pointcloud2[:,:3], T_lidar2_lidar1


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


# Utility function (can be outside the class or a static method)
def pose_to_matrix(pose_vec):
    """
    Converts a pose vector [x, y, z, roll, pitch, yaw] (angles in degrees)
    to a 4x4 SE(3) transformation matrix.
    """
    translation = np.array(pose_vec[:3])
    # Assuming roll, pitch, yaw are in degrees.
    # Standard sequence for many systems is 'xyz' extrinsic or 'ZYX' intrinsic.
    # If your system uses a different order, adjust 'xyz'.
    rotation_obj = R.from_euler('xyz', pose_vec[3:6], degrees=True)
    rotation_matrix = rotation_obj.as_matrix()
    
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation
    return matrix




class V2XSim_detected_Reader:
    def __init__(self, json_file_path="/mnt/ssd_gw/v2i-calib/data/DAIR-V2X/detected/given_boxes_val.json", json_data_str=None):
        if json_file_path:
            with open(json_file_path, 'r') as f:
                self.dataset_json = json.load(f)
        elif json_data_str:
            self.dataset_json = json.loads(json_data_str)
        else:
            raise ValueError("Either json_file_path or json_data_str must be provided.")

    def get_3dbbox_object_list(self, bboxes_corners_list_raw):
        """
        Converts a list of raw bounding box corners (list of lists of lists)
        into a list of BBox3d objects.
        Args:
            bboxes_corners_list_raw (list): A list where each element is an 8x3 list/array of corners.
        Returns:
            list: A list of BBox3d objects.
        """
        bbox3d_object_list = []
        for corners_raw in bboxes_corners_list_raw:
            # The BBox3d constructor expects bbox_type and bbox_8_3 (corners)
            # We'll use 'car' as a default type and pass other BBox defaults.
            bbox3d_object_list.append(BBox3d(bbox_type='car', bbox_8_3=np.array(corners_raw)))
        return bbox3d_object_list

    def get_noised_3dbbox_object_list(self, T_transform_target_from_base, base_bbox_object_list, 
                                      noise_type, noise_params):
        """
        Transforms a list of BBox3d objects from a base frame to a target frame,
        and then applies noise to the corners of these transformed objects.
        Args:
            T_transform_target_from_base (np.ndarray): 4x4 matrix to transform objects from base frame to target frame.
            base_bbox_object_list (list): List of BBox3d objects in the base frame.
            noise_type (str): Type of noise (e.g., 'gaussian').
            noise_params (dict): Parameters for noise generation (e.g., 'pos_std', 'pos_mean').
        Returns:
            list: List of new BBox3d objects in the target frame, with noise applied.
        """
        # Step 1: Transform objects to the target CAV's frame using the global function
        # This function returns a NEW list of NEW BBox3d objects with transformed corners.
        transformed_bboxes_list = implement_T_3dbox_object_list(T_transform_target_from_base, base_bbox_object_list)

        if noise_type != 'gaussian' or noise_params is None: # Or no noise_type
            return transformed_bboxes_list

        # Step 2: Apply noise to the corners of the already transformed BBox3d objects
        noised_final_bboxes_list = []
        pos_std = noise_params.get('pos_std', 0.0)
        pos_mean = noise_params.get('pos_mean', 0.0)
        # Note: Applying rotational noise directly to corners is complex. 
        # True rotational noise would typically be applied to the object's orientation parameters
        # before corners are derived. Here, we only apply positional noise to corners.

        for bbox_obj in transformed_bboxes_list: # bbox_obj is already a transformed copy
            noisy_corners = bbox_obj.get_bbox3d_8_3() + np.random.normal(
                pos_mean, 
                pos_std, 
                bbox_obj.get_bbox3d_8_3().shape
            )
            # Create a new BBox3d object with the noised corners, preserving other attributes
            # by using the copy method of the transformed bbox_obj and then updating corners.
            noised_bbox = bbox_obj.copy() # Preserves type, 2d_bbox, confidence etc.
            noised_bbox.bbox3d_8_3 = noisy_corners 
            noised_final_bboxes_list.append(noised_bbox)
        
        return noised_final_bboxes_list

    def generate_vehicle_vehicle_bboxes_object_list(self, 
                                                    noise_type=None, 
                                                    noise={'pos_std':0.2, 'rot_std':0.2, 'pos_mean':0, 'rot_mean':0}):
        """
        Generates pairs of bounding box lists for pairs of CAVs from the loaded JSON data.
        The first CAV in each frame's list is used as the reference (CAV1). Its detected objects
        (in its own LiDAR frame) are considered the "base scene". These base objects are then
        transformed (and potentially noised) into the view of other CAVs (CAV2, CAV3, etc.)
        to form pairs (CAV1_objects, CAV2_objects_transformed_from_CAV1).
        """
        for frame_idx_str in self.dataset_json.keys():
            frame_data = self.dataset_json[frame_idx_str]
            cav_id_str_list = frame_data['cav_id_list']
            
            if len(cav_id_str_list) < 1: # Need at least one CAV to be the reference
                continue

            # --- Reference CAV (CAV1) Setup ---
            ref_cav_list_idx = 0 # Index of the reference CAV in the frame's lists
            
            # Pose of the reference CAV (CAV1) in the world
            T_world_ref_lidar = pose_to_matrix(frame_data['lidar_pose_clean_np'][ref_cav_list_idx])
            
            # Base objects are detections from the reference CAV, in its own LiDAR frame.
            base_objects_corners_in_ref_frame = frame_data['pred_corner3d_np_list'][ref_cav_list_idx]
            base_bbox3d_list_in_ref_frame = self.get_3dbbox_object_list(base_objects_corners_in_ref_frame)

            # Objects for CAV1 (the reference CAV).
            # T_cav1_from_ref is Identity because CAV1 is the reference.
            T_cav1_from_ref = np.eye(4)
            if noise_type is None:
                # Create copies even if no noise, to ensure distinct lists per yield
                bbox3d_object_list_lidar1 = implement_T_3dbox_object_list(T_cav1_from_ref, base_bbox3d_list_in_ref_frame)
            else:
                bbox3d_object_list_lidar1 = self.get_noised_3dbbox_object_list(
                    T_cav1_from_ref, 
                    base_bbox3d_list_in_ref_frame, 
                    noise_type, 
                    noise
                )

            # --- Pair with other CAVs (CAV2) ---
            # The outer loop in user's example code (with `if cav_id1 == 1: break`)
            # effectively means CAV1 is always the first CAV (index 0 here).
            for cav2_list_idx in range(ref_cav_list_idx + 1, len(cav_id_str_list)):
                T_world_lidar2 = pose_to_matrix(frame_data['lidar_pose_clean_np'][cav2_list_idx])

                # Transformation matrix to bring objects from ref_lidar_frame to lidar2_frame:
                # X_lidar2 = T_lidar2_world @ X_world
                # X_world = T_world_ref_lidar @ X_ref_lidar
                # So, X_lidar2 = (T_lidar2_world @ T_world_ref_lidar) @ X_ref_lidar
                # T_lidar2_world = inv(T_world_lidar2)
                T_cav2_from_ref = np.linalg.inv(T_world_lidar2) @ T_world_ref_lidar
                
                if noise_type is None:
                     bbox3d_object_list_lidar2 = implement_T_3dbox_object_list(
                         T_cav2_from_ref, 
                         base_bbox3d_list_in_ref_frame
                     )
                else:
                    bbox3d_object_list_lidar2 = self.get_noised_3dbbox_object_list(
                        T_cav2_from_ref,
                        base_bbox3d_list_in_ref_frame,
                        noise_type,
                        noise
                    )
                
                # Relative pose of lidar2_frame w.r.t lidar1_frame (CAV1 is ref_lidar)
                # T_lidar2_in_lidar1_frame = inv(T_world_lidar1) @ T_world_lidar2
                # Since T_world_lidar1 is T_world_ref_lidar:
                T_lidar2_in_lidar1_frame = np.linalg.inv(T_world_lidar2) @  T_world_ref_lidar
                
                current_frame_int_idx = int(frame_idx_str)
                cav_id_str_cav2 = cav_id_str_list[cav2_list_idx]

                yield current_frame_int_idx, cav_id_str_cav2, \
                      bbox3d_object_list_lidar1, bbox3d_object_list_lidar2, \
                      T_lidar2_in_lidar1_frame


if __name__ == "__main__":
    # reader = V2XSim_Reader() 
    # # reader.generate_vehicle_vehicle_bboxes_object_list()
    # reader.generate_gt_and_noised_bboxes_object_list()
    cnt = 0
    for id1, id2, infra_boxes_object_list, vehicle_boxes_object_list, T_true in V2XSim_detected_Reader().generate_vehicle_vehicle_bboxes_object_list():
        if cnt ==1:
            print(f"id1: {id1}, id2: {id2}, infra_boxes_object_list: {len(infra_boxes_object_list)}, vehicle_boxes_object_list: {len(vehicle_boxes_object_list)}, T_true: {T_true}")
            T_true_converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, infra_boxes_object_list)
            # BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([T_true_converted_infra_boxes_object_list, vehicle_boxes_object_list], [], [(1, 0, 0), (0, 1, 0)], 'true T')
            BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([infra_boxes_object_list, vehicle_boxes_object_list], [], [(1, 0, 0), (0, 1, 0)], 'true T')
            break
        else:
            cnt += 1
            continue
        # print(f"id1: {id1}, id2: {id2}, infra_boxes_object_list: {len(infra_boxes_object_list)}, vehicle_boxes_object_list: {len(vehicle_boxes_object_list)}, T_true: {T_true}")
        # T_true_converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, infra_boxes_object_list)
        # BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([T_true_converted_infra_boxes_object_list, vehicle_boxes_object_list], [], [(1, 0, 0), (0, 1, 0)], 'true T')
        # BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([infra_boxes_object_list, vehicle_boxes_object_list], [], [(1, 0, 0), (0, 1, 0)], 'true T')
        