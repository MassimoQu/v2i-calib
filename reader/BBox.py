class BBox:
    # represent both 2d bbox and 3d bbox
    def __init__(self, bbox_type, occluded_state = 0, truncated_state = 0):
        self.bbox_type = bbox_type
        self.occluded_state = occluded_state
        self.truncated_state = truncated_state
        # self.bbox2d_4 = bbox_4

    def get_bbox_type(self):
        return self.bbox_type
    
    def get_occluded_state(self):
        return self.occluded_state
    
    def get_truncated_state(self):
        return self.truncated_state
    
    