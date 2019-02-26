import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.augmentation import Augmentation
from renom_img.api.utility.augmentation.process import Shift, Rotate


class Builder:

    def builder(self, img_path_list, annotation_list, augmentation, nth):
        img_list = []
        for p in img_path_list:
            img = Image.open(p)
            img = np.asarray(img).transpose(2, 0, 1).copy()[None]
            img, _ = augmentation(img, mode="classification")
            img_list.append(img)
        return np.concatenate(img_list)


def test_distributor():
    root_dir = 'test_files'

    # Create test images.
    os.makedirs(root_dir, exist_ok=True)
    for i in range(8):
        img = Image.fromarray(np.random.randint(0, 255, size=(224, 224, 3)).astype(np.uint8))
        img.save(os.path.join(root_dir, '{:03d}.png'.format(i)))

    # Provides function to distributor.
    # Argumentation.
    builder = Builder().builder
    aug = Augmentation([Shift(), Rotate()])
    dist = ImageDistributor(list([os.path.join(root_dir, p) for p in os.listdir(root_dir)]),
                            augmentation=aug, target_builder=builder, num_worker=4)

    for x in dist.batch(batch_size=4):
        assert x.shape == (4, 3, 224, 224)

    # Remove files.
    shutil.rmtree(root_dir)
