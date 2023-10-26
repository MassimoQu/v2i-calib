from object.reader import Reader

class TaskExecutor:
    def __init__(self, task):
        self.task = task
        self.reader = Reader(task.file_path)

    def execute(self):
        self.task.execute()
        