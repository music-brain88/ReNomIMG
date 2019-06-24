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
        self.status_code = "API0000"
        super(UnknownError,self).__init__(msg)
        
class MissingParamError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "API0001"
        super(MissingParamError,self).__init__(msg)
        
class InvalidParamError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "API0002"
        super(InvalidParamError,self).__init__(msg)

class MissingDataError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "API0003"
        super(MissingDataError,self).__init__(msg)
        
class InvalidDataError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "API0004"
        super(InvalidDataError,self).__init__(msg)
        
class OutOfMemoryError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "API0005"
        super(OutOfMemoryError,self).__init__(msg)
        
class WeightNotFoundError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "API0006"
        super(WeightNotFoundError,self).__init__(msg)

class WeightLoadError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "API0007"
        super(WeightLoadError,self).__init__(msg)
        
class ParamValueError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "API0008"
        super(ParamValueError,self).__init__(msg)
        
class DataValueError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "API0009"
        super(DataValueError, self).__init__(msg)

class WeightURLOpenError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "API0010"
        super(WeightURLOpenError, self).__init__(msg)

class WeightRetrieveError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "API0011"
        super(WeightRetrieveError, self).__init__(msg)

class LearningRateError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "API0012"
        super(LearningRateError,self).__init__(msg)

class OptimizerError(ReNomIMGError):
    def __init__(self,msg=None):
        self.status_code = "API0013"
        super(OptimizerError,self).__init__(msg)

class LossError(ReNomIMGError):
    def __init__(self,msg=None):
        self.status_code = "API0014"
        super(LossError,self).__init__(msg) 
