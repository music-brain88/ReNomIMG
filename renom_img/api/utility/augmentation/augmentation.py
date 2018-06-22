
import numpy as np
from renom_img.api.utility.augmentation.process import MODE


class Augmentation(object):
    """
    This class is for applying augmentation to images.
    Augmentation is called only when training process.
    You could choose transform function from Process module.

    Attributes:
        process_list (list): list of augmentation methods.

    Example:
        >>> from renom_img.api.utility.precess import Flip, Shift, Rotate, WhiteNoise
        >>> from renom_img.api.utility.augmentation import Augmentation
        >>> from renom_img.api.utility.distributor import ImageDistributor
        >>> aug = Augmentation([
        ...     Shift(40, 40),
        ...     Rotate(),
        ...     Flip(),
        ...     WhiteNoise()
        ... ])
        >>> distributor = ImageDistributor(
        ...     img_path_list,
        ...     label_list,
        ...     builder,
        ...     aug,
        ...     num_worker
        ... )

    """
    def __init__(self, process_list):
        self._process_list = process_list

    def __call__(self, x, y=None, mode="classification"):
        return self.transform(x, y, mode)

    def transform(self, x, y=None, mode="classification"):
        """
        This function is for applying augmentation to ImageDistributor

        Args:
            x (list of numpy.ndarray):
            y (list of str): list of label for x. It is only used when prediction.

        Returns:

        """
        assert_msg = "{} is not supported transformation mode. {} are available."
        assert mode in MODE, assert_msg.format(mode, MODE)
        for process in self._process_list:
            if np.random.rand() >= 0.9:
                continue
            x, y = process(x, y, mode)
        return x, y
