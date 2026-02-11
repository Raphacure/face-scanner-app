class BaseMetric:
    name = "base"

    def calculate(self, context):
        raise NotImplementedError
