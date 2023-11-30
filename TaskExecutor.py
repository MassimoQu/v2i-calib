from process.Task import Task

class TaskExecutor:
    def __init__(self, task):
        self.task = task
        
    def execute(self):
        self.task.execute()
        