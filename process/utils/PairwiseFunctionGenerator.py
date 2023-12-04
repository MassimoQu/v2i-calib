class PairwiseFunctionGenerator:
    def __init__(self, list1, list2, func, constraint=None):
        self.list1 = list1
        self.list2 = list2
        self.func = func
        self.constraint = constraint

    def generate(self):
        for item1 in self.list1:
            for item2 in self.list2:
                if self.constraint is not None:
                    if self.constraint(item1, item2):
                        yield self.func(item1, item2)