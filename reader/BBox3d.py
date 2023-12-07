import copy
import numpy as np
from BBox import BBox

class BBox3d(BBox):
    def __init__(self, bbox_type, bbox_8_3, bbox_4 = [0, 0, 0, 0], occluded_state = 0, truncated_state = 0, alpha = 0.0):
        super().__init__(bbox_type, bbox_4, occluded_state, truncated_state, alpha)
        self.bbox3d_8_3 = bbox_8_3

    def __eq__(self, other):
        if not isinstance(other, BBox3d):
            return False
        return (super().__eq__(other) and
                np.array_equal(self.bbox3d_8_3, other.bbox3d_8_3))

    def get_bbox3d_8_3(self):
        return self.bbox3d_8_3
    
    def copy(self):
        return BBox3d(self.bbox_type, copy.deepcopy(self.bbox3d_8_3), self.occluded_state, self.truncated_state)