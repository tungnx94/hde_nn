class WFException(Exception):

    def __init__(self, message, name=None):
        self.message = message
        self.name = name

    def __str__(self):
        if (self.name is not None):
            return self.name + ": " + self.message
        else:
            return self.message