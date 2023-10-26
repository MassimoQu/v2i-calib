from object.reader import Reader

class Task:
    def __init__(self, input_file_path = None) -> None:
        self.reader = None
        if input_file_path is not None:
            self.reader = Reader(input_file_path)
        


    def execute(self):
        pass
