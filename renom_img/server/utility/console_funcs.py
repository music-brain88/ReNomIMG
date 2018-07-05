import os
import shutil
import numpy as np


def divide_datasets(r=0.8):
    """This method creates directories, 'train_set' and 'valid_set'.
    And automatically copies images and labels into them.
    """
    cwd = os.getcwd()
    assert os.path.exists('label'), \
        "The `label` folder not found. Please confirm `label` folder exists in current directory."
    assert os.path.exists('dataset'), \
        "The `dataset` folder not found. Please confirm `dataset` folder exists in current directory."

    ext_candidates = ['.jpg', '.jpeg', '.png']
    data_path_list = [p for p in sorted(os.listdir("dataset"))
                      if os.path.splitext(p)[1] in ext_candidates]
    print("# {} data found.".format(len(data_path_list)))

    train_set_img_path = os.path.join('train_set', 'img')
    train_set_label_path = os.path.join('train_set', 'label')
    valid_set_img_path = os.path.join('valid_set', 'img')
    valid_set_label_path = os.path.join('valid_set', 'label')

    for p in [train_set_img_path, train_set_label_path, valid_set_img_path, valid_set_label_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    perm = np.random.permutation(len(data_path_list))

    if not len(perm):
        return

    train_perm, valid_perm = np.split(perm, [int(len(data_path_list) * r)])
    print("# Data will be divided according following number.")
    print("# | Train data size | Valid data size |")
    print("# |{: 17}|{: 17}|".format(len(train_perm), len(valid_perm)))

    train_count = 0
    valid_count = 0
    error_count = 0

    for p in train_perm:
        path, ext = os.path.splitext(data_path_list[p])
        label_src = os.path.join('label', path + '.xml')
        label_dist = os.path.join(train_set_label_path, path) + '.xml'
        img_src = os.path.join('dataset', path + ext)
        img_dist = os.path.join(train_set_img_path, path + ext)
        if os.path.exists(img_src) and os.path.exists(label_src):
            shutil.copy(img_src, img_dist)
            shutil.copy(label_src, label_dist)
            train_count += 1
        else:
            error_count += 1

    for p in valid_perm:
        path, ext = os.path.splitext(data_path_list[p])
        label_src = os.path.join('label', path + '.xml')
        label_dist = os.path.join(valid_set_label_path, path) + '.xml'
        img_src = os.path.join('dataset', path + ext)
        img_dist = os.path.join(valid_set_img_path, path + ext)
        if os.path.exists(img_src) and os.path.exists(label_src):
            shutil.copy(img_src, img_dist)
            shutil.copy(label_src, label_dist)
            valid_count += 1
        else:
            error_count += 1

    print("# Number of accutual copied data")
    print("# | Train data size | Valid data size | Not copied |")
    print("# |{: 17}|{: 17}|{: 12}|".format(train_count, valid_count, error_count))
