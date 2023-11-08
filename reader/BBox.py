class BBox:
    # represent both 2d bbox and 3d bbox
    def __init__(self, bbox_type):
        self.bbox_type = bbox_type
        self.occluded_state = 0
        self.truncated_state = 0
        # self.bbox2d_4 = bbox_4

