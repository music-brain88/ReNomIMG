import renom as rm


class CnnBase(rm.Model):

    SERIALIZED = ("output_size", "train_whole")

    def __init__(self):
        self.train_whole = None
        self.output_size = None

    def __call__(self, *args, **kwargs):
        assert not ([self.train_whole, self.output_size] in None), \
            "Please set attributes `train_whole` and 'output_size' before running __call__."
        super(CnnBase, self).__call__(*args, **kwargs)

    def set_output_size(self, output_size):
        raise NotImplemented

    def set_train_whole(self, whole):
        self.train_whole = whole
