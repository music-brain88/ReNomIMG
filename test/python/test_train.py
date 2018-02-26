
import os
import sys
import renom as rm
import numpy as np
from renom.cuda import set_cuda_active
sys.path.append('../../python')

from train import Train
from model.yolo_detector import build_truth, Yolo, apply_nms, box_iou

def test_train():
    set_cuda_active(True)
    project_id = 0 
    model_id = 0 
    dataset_id = 0 
    total_epoch = 10 
    seed = 0 
    algorithm = 0 
    hyper_parameter = {
      'image_width': 448,
      'image_height': 448,
      'batch_size': 64,
      'additional_params':{
          'horizontal_cells':7,
          'vertical_cells':7,
          'bounding_box':2
        }
    }
    os.chdir('../../')
    print('Current directory is {}'.format(os.getcwd()))
    train = Train(project_id, model_id, dataset_id, total_epoch, seed, algorithm, hyper_parameter)
    train.execute()
    train.model.save('test_learned.yolo')


if __name__ == '__main__':
    test_train()
