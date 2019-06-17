class ReNomIMGError(Exception):
    code = "IMG0000"
    message = "Unkown error was occured."


class ForbiddenError(ReNomIMGError):
    message = "forbidden."


class NotFoundError(ReNomIMGError):
    message = "not found."


class MethodNotAllowedError(ReNomIMGError):
    message = "not allowed."


class ServiceUnavailableError(ReNomIMGError):
    message = "temporary service down."


class MissingRequestParamError(ReNomIMGError):
    message = "required param is missing."


class InvalidRequestParamError(ReNomIMGError):
    message = "param is invalid."


class TaskNotFoundError(ReNomIMGError):
    message = "Task not found."


class DatasetNotFoundError(ReNomIMGError):
    message = "Dataset not found."


class ModelNotFoundError(ReNomIMGError):
    message = "Model not found."


class WeightNotFoundError(ReNomIMGError):
    message = "Weight not found."


class ModelRunningError(ReNomIMGError):
    message = "Model is running."


class MemoryOverflowError(ReNomIMGError):
    message = "Memory overflow."


class DirectoryNotFound(ReNomIMGError):
    message = "directory not found."
