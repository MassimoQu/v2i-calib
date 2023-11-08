from BBox import BBox

class BBox3d(BBox):
    def __init__(self, bbox_type, bbox_8_3):
        super().__init__(bbox_type)
        self.bbox3d_8_3 = bbox_8_3
        