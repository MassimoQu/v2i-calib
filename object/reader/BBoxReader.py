import os.path as osp

from reader.Reader import Reader
from object import Reader


class BBoxReader(Reader):
    def __init__(self, folder_name, file_name):

        self.file_path = osp.join(folder_name, file_name)
        
        super().__init__(self.file_path)
    
    def read(self):
        with open(self.file_path, 'r') as f:
            return f.read()
        