import numpy as np
import os
import time
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from reader import CooperativeBatchingReader
from reader import CooperativeReader
from preprocess import Filter3dBoxes
from reader import V2XSim_Reader
# import corresponding.BoxesMatch_cpp as BoxesMatch_cpp # type: ignore
from corresponding import CorrespondingDetector
from corresponding import BoxesMatch
from search import Matches2Extrinsics
from utils import implement_T_3dbox_object_list, implement_T_points_n_3, get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF, convert_6DOF_to_T, get_extrinsic_from_two_3dbox_object
from config.config import cfg, cfg_from_yaml_file, Logger



if __name__ == '__main__':
    
    reader = CooperativeReader()

    infra_boxes_object_list, vehicle_boxes_object_list = reader.get_cooperative_infra_vehicle_boxes_object_list()
    T_i2v = reader.get_cooperative_camera_T_i2v()

    converted_infra_boxes_object_list = implement_T_3dbox_object_list(infra_boxes_object_list, T_i2v)

    