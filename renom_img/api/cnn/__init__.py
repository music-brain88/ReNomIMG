import renom as rm
from renom_img.api.utility.exceptions.exceptions import MissingInputError, FunctionNotImplementedError


class CnnBase(rm.Model):

    SERIALIZED = ("output_size", "train_whole")

    def __init__(self):
        self.train_whole = None
        self.output_size = None
        self.has_bn = True

    def __call__(self, *args, **kwargs):
        if (None in [self.train_whole, self.output_size]):
            raise MissingInputError(
                "Please set attributes `train_whole` and 'output_size' before running __call__.")
        return super(CnnBase, self).__call__(*args, **kwargs)

    def set_output_size(self, output_size):
        raise FunctionNotImplementedError("This function has not been implemented.")

    def set_train_whole(self, whole):
        self.train_whole = whole

    def reset_deeper_layer(self):
        pass

    def load_pretrained_weight(self, path):
        raise FunctionNotImplementedError("This function has not been implemented.")
