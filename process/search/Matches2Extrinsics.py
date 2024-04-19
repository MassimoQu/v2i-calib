
from extrinsic_utils import get_extrinsic_from_two_3dbox_object, get_extrinsic_from_two_mixed_3dbox_object_list, convert_T_to_6DOF, optimize_extrinsic_from_two_mixed_3dbox_object_list


class Matches2Extrinsics:
    
    def __init__(self, infra_boxes_object_list, vehicle_boxes_object_list, true_T_6DOF = None, matches_score_list = [], verbose = False, true_matches = []):
        
        self.matches_score_list = matches_score_list
        self.infra_boxes_object_list = infra_boxes_object_list
        self.vehicle_boxes_object_list = vehicle_boxes_object_list
        self.threshold = max(matches_score_list[0][1] * 0.8, matches_score_list[0][1] - 1)
        self.true_matches = true_matches
        
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

    def cal_adequete_matches_threshold(self):
        return [match[0] for match in self.matches_score_list if match[1] >= self.threshold]

    def cal_adequete_matches_true_value(self):
        return [match[0] for match in self.matches_score_list if match[0] in self.true_matches]

    def get_combined_extrinsic(self, matches_filter_strategy = 'trueT', optimization_strategy = 'svd'):
        if matches_filter_strategy == 'trueT':
            adequete_matches = self.cal_adequete_matches_true_value()
        elif matches_filter_strategy == 'threshold':
            adequete_matches = self.cal_adequete_matches_threshold()
        elif matches_filter_strategy == 'top match':
            adequete_matches = [self.matches_score_list[0][0]]
        else:
            raise ValueError('matches_filter_strategy should be trueT, threshold or top match')
        
        infra_boxes_object_list = [self.infra_boxes_object_list[match[0]] for match in adequete_matches]
        vehicle_boxes_object_list = [self.vehicle_boxes_object_list[match[1]] for match in adequete_matches]
        
        if optimization_strategy == 'svd':
            resultT = get_extrinsic_from_two_mixed_3dbox_object_list(infra_boxes_object_list, vehicle_boxes_object_list)
        elif optimization_strategy == 'ndt':
            resultT = optimize_extrinsic_from_two_mixed_3dbox_object_list(infra_boxes_object_list, vehicle_boxes_object_list)
        else:
            raise ValueError('optimization_strategy should be svd or ndt')
        
        return convert_T_to_6DOF(resultT)
    
    
