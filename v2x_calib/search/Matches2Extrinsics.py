import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import get_extrinsic_from_two_3dbox_object, get_extrinsic_from_two_mixed_3dbox_object_list, get_extrinsic_from_two_mixed_3dbox_object_list_svd_without_match, convert_T_to_6DOF#, optimize_extrinsic_from_two_mixed_3dbox_object_list

class Matches2Extrinsics:
    
    def __init__(self, infra_boxes_object_list, vehicle_boxes_object_list, true_T_6DOF = None, matches_score_list = [], verbose = False, svd_strategy = 'svd_with_match'):
        
        self.matches_score_list = matches_score_list
        self.infra_boxes_object_list = infra_boxes_object_list
        self.vehicle_boxes_object_list = vehicle_boxes_object_list
        self.svd_strategy = svd_strategy
        self.threshold = max(matches_score_list[0][1] * 0.8, matches_score_list[0][1] - 1) if len(matches_score_list) >= 1 else 0

        self.true_T_6DOF_format = true_T_6DOF

        if verbose:
            
            if self.true_T_6DOF_format is not None:
                print('true_T_6DOF: ', true_T_6DOF)
            
            print('self.threshold: ', self.threshold)

            adequate_num = len([match[0] for match in self.matches_score_list if match[1] >= self.threshold])
            print('len(adequete_matches): ', adequate_num)

            cnt = 0

            for match, score in self.matches_score_list:
                print(cnt)
                infra_box_object = self.infra_boxes_object_list[match[0]]
                vehicle_box_object = self.vehicle_boxes_object_list[match[1]]
                extrinsic = get_extrinsic_from_two_3dbox_object(infra_box_object, vehicle_box_object)
                print('- score: ', score)
                print('- extrinsic: ', extrinsic)
                cnt += 1
                if score < self.threshold:
                    print('below threshold')


    def get_combined_extrinsic(self, matches2extrinsic_strategies = 'weightedSVD'):

        infra_boxes_object_list = [self.infra_boxes_object_list[match[0]] for match, _ in self.matches_score_list]
        vehicle_boxes_object_list = [self.vehicle_boxes_object_list[match[1]] for match, _ in self.matches_score_list]
        weights = [score for _, score in self.matches_score_list]
        
        if self.svd_strategy == 'svd_with_match':
            if matches2extrinsic_strategies == 'evenSVD':
                resultT = get_extrinsic_from_two_mixed_3dbox_object_list(infra_boxes_object_list, vehicle_boxes_object_list)
            elif matches2extrinsic_strategies == 'weightedSVD':
                resultT = get_extrinsic_from_two_mixed_3dbox_object_list(infra_boxes_object_list, vehicle_boxes_object_list, weights)
            # elif matches2extrinsic_strategies == 'ndt':
            #     resultT = optimize_extrinsic_from_two_mixed_3dbox_object_list(infra_boxes_object_list, vehicle_boxes_object_list)
            else:
                raise ValueError(f'matches2extrinsic_strategies={matches2extrinsic_strategies}, matches2extrinsic_strategies should be svd8point or ndt')
        elif self.svd_strategy == 'svd_without_match':
            if matches2extrinsic_strategies == 'evenSVD':
                resultT = get_extrinsic_from_two_mixed_3dbox_object_list_svd_without_match(infra_boxes_object_list, vehicle_boxes_object_list)
            elif matches2extrinsic_strategies == 'weightedSVD':
                resultT = get_extrinsic_from_two_mixed_3dbox_object_list_svd_without_match(infra_boxes_object_list, vehicle_boxes_object_list, weights)
            else:
                raise ValueError(f'matches2extrinsic_strategies={matches2extrinsic_strategies}, matches2extrinsic_strategies should be svd8point or ndt')
        else:
            raise ValueError(f'svd_strategy={self.svd_strategy}, svd_strategy should be svd_with_match or svd_without_match')

        return convert_T_to_6DOF(resultT)
    
    

