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
        self.status_code = "000"
        super(UnknownError,self).__init__(msg)
        
class MissingParamError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "001"
        super(MissingParamError,self).__init__(msg)
        
class InvalidParamError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "002"
        super(InvalidParamError,self).__init__(msg)

class MissingDataError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "003"
        super(MissingDataError,self).__init__(msg)
        
class InvalidDataError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "004"
        super(InvalidDataError,self).__init__(msg)
        
class OutOfMemoryError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "005"
        super(OutOfMemoryError,self).__init__(msg)
        
class WeightNotFoundError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "006"
        super(WeightNotFoundError,self).__init__(msg)

class WeightLoadError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "007"
        super(WeightLoadError,self).__init__(msg)
        
class ParamValueError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "008"
        super(ParamValueError,self).__init__(msg)
        
class DataValueError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "009"
        super(DataValueError, self).__init__(msg)

class WeightURLOpenError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "010"
        super(WeightURLOpenError, self).__init__(msg)

class WeightRetrieveError(ReNomIMGError):
    def __init__(self, msg=None):
        self.status_code = "011"
        super(WeightRetrieveError, self).__init__(msg)

