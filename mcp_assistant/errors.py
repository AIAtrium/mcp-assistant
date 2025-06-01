class UninitializedSession(Exception):
    def __init__(self, message="Perhaps you forgot to initialize the session"):
        super().__init__(message)


class EmptyOutput(Exception):
    pass


class InvalidToolArgsType(Exception):
    pass
