from reader.Reader import Reader
from object import Reader


class BBoxReader(Reader):
    def __init__(self, file_path):
        super().__init__(file_path)
    
    def read(self):
        with open(self.file_path, 'r') as f:
            return f.read()
        