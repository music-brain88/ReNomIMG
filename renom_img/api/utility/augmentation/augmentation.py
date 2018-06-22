
import numpy as np
from renom_img.api.utility.augmentation.process import MODE


class Augmentation(object):
    """This class is for applying augmentation to images.
    Instance of augmentation is passed to ImageDistributor module,
    and is called only when training process is runnning.
    You could choose augmentation methods from Process module.

    Attributes:
        process_list (list of Process modules): list of Process modules.
        You could choose from Flip, Shift, Rotate and WhiteNoise

    Example:
        >>> from renom_img.api.utility.augmentation.process import Flip, Shift, Rotate, WhiteNoise
        >>> from renom_img.api.utility.augmentation.augmentation import Augmentation
        >>> from renom_img.api.utility.distributor.distributor import ImageDistributor
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
        """This function is for applying augmentation to images.

        Args:
            x (list of numpy.ndarray): List of images.
            y (list of dict): List of annotation results.

        Returns:
            x (list of numpy.ndarray): List of transformed images.
            y (list of dict): List of annotation results.

        """
        return self.transform(x, y, mode)

    def transform(self, x, y=None, mode="classification"):
        assert_msg = "{} is not supported transformation mode. {} are available."
        assert mode in MODE, assert_msg.format(mode, MODE)
        for process in self._process_list:
            if np.random.rand() >= 0.9:
                continue
            x, y = process(x, y, mode)
        return x, y
