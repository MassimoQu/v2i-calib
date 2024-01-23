
from extrinsic_utils import get_extrinsic_from_two_3dbox_object, get_extrinsic_from_two_mixed_3dbox_object_list, convert_T_to_6DOF

class Matches2Extrinsics:
    
    def __init__(self, infra_boxes_object_list, vehicle_boxes_object_list, true_T_6DOF = None, matches_score_list = [], verbose = False, max_matches_num = 10):
        
        self.matches_score_list = matches_score_list
        self.infra_boxes_object_list = infra_boxes_object_list
        self.vehicle_boxes_object_list = vehicle_boxes_object_list
        self.threshold = max(matches_score_list[0][1] * 0.8, matches_score_list[0][1] - 1)
        
        self.true_T_6DOF_format = true_T_6DOF
        
        self.max_matches_num = max_matches_num 

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


    # def get_combined_extrinsic(self):
    #     # adequete_matches = [match[0] for match in self.matches_score_list if match[1] >= self.threshold]
    #     cnt = 0
    #     adequete_matches = []
    #     for match, _ in self.matches_score_list:
    #         if cnt < self.max_matches_num:
    #             adequete_matches.append(match)
    #         cnt += 1

    #     infra_boxes_object_list = [self.infra_boxes_object_list[match[0]] for match in adequete_matches]
    #     vehicle_boxes_object_list = [self.vehicle_boxes_object_list[match[1]] for match in adequete_matches]
        
    #     return convert_T_to_6DOF(get_extrinsic_from_two_mixed_3dbox_object_list(infra_boxes_object_list, vehicle_boxes_object_list))


    def get_combined_extrinsic(self):
        adequete_matches = [match[0] for match in self.matches_score_list if match[1] >= self.threshold]
        infra_boxes_object_list = [self.infra_boxes_object_list[match[0]] for match in adequete_matches]
        vehicle_boxes_object_list = [self.vehicle_boxes_object_list[match[1]] for match in adequete_matches]
        
        return convert_T_to_6DOF(get_extrinsic_from_two_mixed_3dbox_object_list(infra_boxes_object_list, vehicle_boxes_object_list))
