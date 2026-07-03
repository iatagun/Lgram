class CohesionError(Exception):
    pass


class ModelNotAvailableError(CohesionError):
    pass


class SerializationError(CohesionError):
    pass


class StreamError(CohesionError):
    pass


class InsufficientDataError(CohesionError):
    pass
