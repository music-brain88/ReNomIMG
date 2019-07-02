class ReNomIMGError(Exception):
    def __init__(self, msg=None):
        self.message = msg+"\n" if not msg is None else ""

    def _message(self):
        return self.message

    def _append_message(self,msg):
        self.message += msg
        self.message += '\n'

    def _code(self):
        return self.code


class UnknownError(ReNomIMGError):
    code = "IMG-0202-ER-R001-01"
    
    def __init__(self, msg=None):
        super(UnknownError,self).__init__(msg)


class MissingInputError(ReNomIMGError):
    code = "IMG-0202-ER-R001-02"

    def __init__(self, msg=None):
        super(MissingInputError,self).__init__(msg)


class InvalidInputTypeError(ReNomIMGError):
    code = "IMG-0202-ER-R001-03"

    def __init__(self, msg=None):
        super(InvalidInputTypeError,self).__init__(msg)


class InvalidInputValueError(ReNomIMGError):
    code = "IMG-0202-ER-R001-04"
    
    def __init__(self, msg=None):
        super(InvalidInputValueError,self).__init__(msg)


class WeightNotFoundError(ReNomIMGError):
    code = "IMG-0202-ER-R001-05"

    def __init__(self, msg=None):
        super(WeightNotFoundError,self).__init__(msg)


class WeightLoadError(ReNomIMGError):
    code = "IMG-0202-ER-R001-06"

    def __init__(self, msg=None):
        super(WeightLoadError,self).__init__(msg)


class WeightURLOpenError(ReNomIMGError):
    code = "IMG-0202-ER-R001-07"

    def __init__(self, msg=None):
        super(WeightURLOpenError, self).__init__(msg)


class WeightRetrieveError(ReNomIMGError):
    code = "IMG-0202-ER-R001-08"

    def __init__(self, msg=None):
        super(WeightRetrieveError, self).__init__(msg)


class InvalidLearningRateError(ReNomIMGError):
    code = "IMG-0202-ER-R001-09"

    def __init__(self, msg=None):
        super(InvalidLearningRateError,self).__init__(msg)


class InvalidOptimizerError(ReNomIMGError):
    code = "IMG-0202-ER-R001-10"

    def __init__(self,msg=None):
        super(InvalidOptimizerError,self).__init__(msg)


class InvalidValueError(ReNomIMGError):
    code = "IMG-0202-ER-R001-11"

    def __init__(self, msg=None):
        super(InvalidValueError,self).__init__(msg)


class InvalidLossValueError(ReNomIMGError):
    code = "IMG-0202-ER-R001-12"

    def __init__(self,msg=None):
        super(InvalidLossValueError,self).__init__(msg) 


class OutOfMemoryError(ReNomIMGError):
    code = "IMG-0202-ER-R001-13"

    def __init__(self, msg=None):
        super(OutOfMemoryError,self).__init__(msg)


class FunctionNotImplementedError(ReNomIMGError):
    code = "IMG-0202-ER-R001-14"

    def __init__(self, msg=None):
        super(FunctionNotImplementedError,self).__init__(msg)


class ServerConnectionError(ReNomIMGError):
    code = "IMG-0202-ER-R001-15"

    def __init__(self, msg=None):
        super(ServerConnectionError,self).__init__(msg)

