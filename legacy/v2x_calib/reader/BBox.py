class BBox:
    # represent both 2d bbox and 3d bbox
    def __init__(self, bbox_type, bbox_4 = [0, 0, 0, 0], occluded_state = 0, truncated_state = 0, alpha = 0.0, confidence = 1.0):
        self.bbox_type = bbox_type
        self.occluded_state = occluded_state
        self.truncated_state = truncated_state
        self.alpha = alpha
        self.bbox2d_4 = bbox_4
        self.confidence = confidence

    def __eq__(self, other):
        if not isinstance(other, BBox):
            return False
        return (self.bbox_type == other.bbox_type and
                self.occluded_state == other.occluded_state and
                self.truncated_state == other.truncated_state)

    def get_bbox_type(self):
        return self.bbox_type.lower()
    
    def get_bbox2d_4(self):
        return self.bbox2d_4
    
    def get_occluded_state(self):
        return self.occluded_state
    
    def get_truncated_state(self):
        return self.truncated_state
    
    def get_alpha(self):
        return self.alpha
    
    def get_confidence(self):
        return self.confidence

    def set_confidence(self, confidence):
        self.confidence = confidence

    def copy(self):
        return BBox(self.bbox_type, self.bbox2d_4, self.occluded_state, self.truncated_state, self.alpha, self.confidence)
    