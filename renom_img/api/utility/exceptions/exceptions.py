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
        
class MissingParamError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-02"
        super(MissingParamError,self).__init__(msg)
        
class InvalidParamError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-03"
        super(InvalidParamError,self).__init__(msg)

class MissingDataError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-04"
        super(MissingDataError,self).__init__(msg)
        
class InvalidDataError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-05"
        super(InvalidDataError,self).__init__(msg)
        
class OutOfMemoryError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-06"
        super(OutOfMemoryError,self).__init__(msg)
        
class WeightNotFoundError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-07"
        super(WeightNotFoundError,self).__init__(msg)

class WeightLoadError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-08"
        super(WeightLoadError,self).__init__(msg)
        
class ParamValueError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-09"
        super(ParamValueError,self).__init__(msg)
        
class DataValueError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-10"
        super(DataValueError, self).__init__(msg)

class WeightURLOpenError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-11"
        super(WeightURLOpenError, self).__init__(msg)

class WeightRetrieveError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-12"
        super(WeightRetrieveError, self).__init__(msg)

class LearningRateError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "IMG-0202-ER-R001-13"
        super(LearningRateError,self).__init__(msg)

class OptimizerError(ReNomIMGError):
    def __init__(self,msg=None):
        self.status_code = "IMG-0202-ER-R001-14"
        super(OptimizerError,self).__init__(msg)

class LossError(ReNomIMGError):
    def __init__(self,msg=None):
        self.status_code = "IMG-0202-ER-R001-15"
        super(LossError,self).__init__(msg) 
