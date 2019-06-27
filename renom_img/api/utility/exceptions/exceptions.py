class ReNomIMGError(Exception):
    def __init__(self, msg=None):
        self.message = msg+"\n" if not msg is None else ""

    def _message(self):
        return self.message

    def _append_message(self,msg):
        self.message += msg
        self.message += '\n'

    def _status_code(self):
        return self.status_code

class UnknownError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-01"
        super(UnknownError,self).__init__(msg)

class MissingInputError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-02"
        super(MissingInputError,self).__init__(msg)

class InvalidInputTypeError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-03"
        super(InvalidInputTypeError,self).__init__(msg)

class InvalidInputValueError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-04"
        super(InvalidInputValueError,self).__init__(msg)

class WeightNotFoundError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-05"
        super(WeightNotFoundError,self).__init__(msg)

class WeightLoadError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-06"
        super(WeightLoadError,self).__init__(msg)

class WeightURLOpenError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-07"
        super(WeightURLOpenError, self).__init__(msg)

class WeightRetrieveError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-08"
        super(WeightRetrieveError, self).__init__(msg)

class InvalidLearningRateError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-09"
        super(InvalidLearningRateError,self).__init__(msg)

class InvalidOptimizerError(ReNomIMGError):
    def __init__(self,msg=None):
        self.status_code = "IMG-0202-ER-R001-10"
        super(InvalidOptimizerError,self).__init__(msg)

class InvalidValueError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-11"
        super(InvalidValueError,self).__init__(msg)

class InvalidLossValueError(ReNomIMGError):
    def __init__(self,msg=None):
        self.status_code = "IMG-0202-ER-R001-12"
        super(InvalidLossValueError,self).__init__(msg) 

class OutOfMemoryError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-13"
        super(OutOfMemoryError,self).__init__(msg)

class FunctionNotImplementedError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-14"
        super(FunctionNotImplementedError,self).__init__(msg)

class ServerConnectionError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-15"
        super(ServerConnectionError,self).__init__(msg)

