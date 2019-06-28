class ReNomIMGServerError(Exception):
    code = "IMG-0202-ER-B999-99"

    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message


class ForbiddenError(ReNomIMGServerError):
    code = "IMG-0202-ER-B001-01"

    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message


class NotFoundError(ReNomIMGServerError):
    code = "IMG-0202-ER-B001-02"

    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message


class MethodNotAllowedError(ReNomIMGServerError):
    code = "IMG-0202-ER-B001-03"

    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message


class ServiceUnavailableError(ReNomIMGServerError):
    code = "IMG-0202-ER-B001-04"

    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message


class DirectoryNotFound(ReNomIMGServerError):
    code = "IMG-0202-ER-B002-01"

    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message


class TaskNotFoundError(ReNomIMGServerError):
    code = "IMG-0202-ER-B002-02"

    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message


class DatasetNotFoundError(ReNomIMGServerError):
    code = "IMG-0202-ER-B002-03"

    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message


class ModelNotFoundError(ReNomIMGServerError):
    code = "IMG-0202-ER-B002-04"

    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message


class WeightNotFoundError(ReNomIMGServerError):
    code = "IMG-0202-ER-B002-05"

    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message


class ModelRunningError(ReNomIMGServerError):
    code = "IMG-0202-ER-B003-01"

    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message


class InvalidRequestParamError(ReNomIMGServerError):
    code = "IMG-0202-ER-B004-01"

    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message


class MemoryOverflowError(ReNomIMGServerError):
    code = "IMG-0202-ER-B005-01"

    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message
