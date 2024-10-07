import open3d as o3d
from matplotlib import pyplot as pyplot
import numpy as np
import copy
from scipy import spatial 
import random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from time import perf_counter
from config.config import Logger
from v2x_calib.utils import get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF
from v2x_calib.reader.CooperativeBatchingReader import CooperativeBatchingReader
from v2x_calib import V2XSim_Reader
import math

#Kabsch Algorithm
def compute_transformation(source,target):
    #Normalization
    number = len(source)
    #the centroid of source points
    cs = np.zeros((3,1))
    #the centroid of target points
    ct = copy.deepcopy(cs)
    cs[0] = np.mean(source[:][0]);cs[1]=np.mean(source[:][1]);cs[2]=np.mean(source[:][2])
    ct[0] = np.mean(target[:][0]);cs[1]=np.mean(target[:][1]);cs[2]=np.mean(target[:][2])
    #covariance matrix
    cov = np.zeros((3,3))
    #translate the centroids of both models to the origin of the coordinate system (0,0,0)
    #subtract from each point coordinates the coordinates of its corresponding centroid
    for i in range(number):
        sources = source[i].reshape(-1,1)-cs
        targets = target[i].reshape(-1,1)-ct
        cov = cov + np.dot(sources,np.transpose(targets))
    #SVD (singular values decomposition)
    u,w,v = np.linalg.svd(cov)
    #rotation matrix
    R = np.dot(u,np.transpose(v))
    #Transformation vector
    T = ct - np.dot(R,cs)
    return R, T

#compute the transformed points from source to target based on the R/T found in Kabsch Algorithm
def _transform(source,R,T):
    points = []
    for point in source:
        points.append(np.dot(R,point.reshape(-1,1)+T))
    return points

#compute the root mean square error between source and target
def compute_rmse(source,target,R,T):
    rmse = 0
    number = len(target)
    points = _transform(source,R,T)
    for i in range(number):
        error = target[i].reshape(-1,1)-points[i]
        rmse = rmse + math.sqrt(error[0]**2+error[1]**2+error[2]**2)
    return rmse

def draw_registrations(source, target, transformation = None, recolor = False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if(recolor): # recolor the points
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    if(transformation is not None): # transforma source to targets
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def pc2array(pointcloud):
    return np.asarray(pointcloud.points)

def registration_RANSAC(source,target,source_feature,target_feature,ransac_n=3,max_iteration=100,max_validation=100):
    #the intention of RANSAC is to get the optimal transformation between the source and target point cloud
    s = pc2array(source) #(4760,3)
    t = pc2array(target)
    #source features (33,4760)
    sf = np.transpose(source_feature.data)
    tf = np.transpose(target_feature.data)
    #create a KD tree
    tree = spatial.KDTree(tf)
    corres_stock = tree.query(sf)[1]
    for i in range(max_iteration):
        #take ransac_n points randomly
        idx = [random.randint(0,s.shape[0]-1) for j in range(ransac_n)]
        corres_idx = corres_stock[idx]
        source_point = s[idx,...]
        target_point = t[corres_idx,...]
        #estimate transformation
        #use Kabsch Algorithm
        R, T = compute_transformation(source_point,target_point)
        #calculate rmse for all points
        source_point = s
        target_point = t[corres_stock,...]
        rmse = compute_rmse(source_point,target_point,R,T)
        #compare rmse and optimal rmse and then store the smaller one as optimal values
        if not i:
            opt_rmse = rmse
            opt_R = R
            opt_T = T
        else:
            if rmse < opt_rmse:
                opt_rmse = rmse
                opt_R = R
                opt_T = T
    return opt_R, opt_T

#used for downsampling
voxel_size = 1
#this is to get the fpfh features, just call the library
def get_fpfh(cp):
    cp = cp.voxel_down_sample(voxel_size)
    cp.estimate_normals()
    return cp, o3d.pipelines.registration.compute_fpfh_feature(cp, o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=100))

def numpy_to_point_cloud(points):
    """
    Convert a numpy array to an Open3D PointCloud object.
    
    Parameters:
        points (np.ndarray): A Nx3 numpy array of xyz positions.
    
    Returns:
        o3d.geometry.PointCloud: An Open3D point cloud object.
    """
    # 创建一个空的 PointCloud 对象
    pcd = o3d.geometry.PointCloud()

    # 将 numpy array 中的数据转换成 Open3D 所需的格式
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


if __name__ == "__main__":

    # logger = Logger("dair-v2x_demo_ransac_fpfh")
    logger = Logger("v2x-sim_ransac_fpfh")

    # for inf_id, veh_id, inf_pc, veh_pc, T_true in CooperativeBatchingReader(path_data_info = f'./data/cooperative-vehicle-infrastructure/cooperative/selected_data_info.json').generate_infra_vehicle_pointcloud():
    for frome_id, cav_id, inf_pc, veh_pc, T_true in V2XSim_Reader().generate_vehicle_vehicle_pointcloud():
        
        inf_pc_ = numpy_to_point_cloud(inf_pc)
        veh_pc_ = numpy_to_point_cloud(veh_pc)
        
        t1 = perf_counter()
        r1, f1 = get_fpfh(inf_pc_)
        r2, f2 = get_fpfh(veh_pc_)
        R, t = registration_RANSAC(r1,r2,f1,f2)
        T = np.vstack((np.hstack((np.float64(R), np.float64(t))), np.array([0,0,0,1])))
        t2 = perf_counter()
        
        RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(convert_T_to_6DOF(T), convert_T_to_6DOF(T_true))
        # logger.info(f"inf_id: {inf_id}, veh_id: {veh_id}, RE: {RE}, TE: {TE}, time: {t2 - t1}")
        logger.info(f"frame_id: {frome_id}, cav_id: {cav_id}, RE: {RE}, TE: {TE}, time: {t2 - t1}")

