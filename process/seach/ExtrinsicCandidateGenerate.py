import numpy as np

import sys
sys.path.append('./process/utils')
from PairwiseFunctionGenerator import PairwiseFunctionGenerator
from extrinsic_utils import get_extrinsic_from_two_3dbox_object, convert_T_to_6DOF


class ExtrinsicCandidateGenerator(PairwiseFunctionGenerator):
    def __init__(self, infra_boxes_object_list, vehicle_boxes_object_list):
        super().__init__(infra_boxes_object_list, vehicle_boxes_object_list, get_extrinsic_from_two_3dbox_object,
                         constraint=lambda infra_boxes_object, vehicle_boxes_object: infra_boxes_object.get_bbox_type() == vehicle_boxes_object.get_bbox_type())
        
    def get_whole_candidate6DOF_list(self):
        candidate6DOF_list = []
        for extrinsic_candidate in self.generate():
            candidate6DOF_list.append(convert_T_to_6DOF(extrinsic_candidate))

        return np.array(candidate6DOF_list)
    
    def get_whole_candidateT_list(self):
        candidateT_list = []
        for extrinsic_candidate in self.generate():
            candidateT_list.append(extrinsic_candidate)

        return np.array(candidateT_list)
    

if __name__ == "__main__":
    pass

