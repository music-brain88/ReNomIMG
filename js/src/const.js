export const TASK_ID = {
  CLASSIFICATION: 0,
  DETECTION: 1,
  SEGMENTATION: 2,
}

export const ALGORITHM = {
  CLASSIFICATION: {
    ResNet: {
      id: 0,
      key: 'resnet',
      title: 'ResNet',
      params: {
        TRAIN_WHOLE: {
          title: 'Train Whole Network',
          key: 'train_whole', // Must be same as back-end.
          type: 'checkbox',
          default: false,
        },
      }
    },
    DenseNet: 2,
    Vgg16: 3,
    Vgg19: 4,
    Inception1: 5,
    Inception2: 6,
    Inception3: 7,
    Inception4: 8,
  },
  DETECTION: {
    YOLOv1: {
      id: 10, // Must be same as back-end.
      key: 'yolov1', // Must be same as back-end.
      title: 'Yolo v1',
      params: {
        TRAIN_WHOLE: {
          title: 'Train Whole Network',
          key: 'train_whole', // Must be same as back-end.
          type: 'checkbox',
          default: false,
        },
        BATCH_SIZE: {
          title: 'Batch Size',
          key: 'batch_size',
          type: 'number',
          default: 64,
          min: 4,
          max: 128,
        },
        IMAGE_WIDTH: {
          title: 'Image Width',
          key: 'imsize_w',
          type: 'number',
          default: 448,
          min: 64,
          max: 512,
        },
        IMAGE_HEIGHT: {
          title: 'Image Height',
          key: 'imsize_h',
          type: 'number',
          default: 448,
          min: 64,
          max: 512,
        },
        TOTAL_EPOCH: {
          title: 'Total Epoch',
          key: 'total_epoch',
          disabled: false,
          type: 'number',
          default: 160,
        },
        CELL: {
          title: 'Num Cell',
          key: 'cell',
          disabled: false,
          type: 'number',
          default: 7,
        },
        BOX: {
          title: 'Num Box',
          key: 'box',
          disabled: false,
          type: 'number',
          default: 2,
        },
      }
    },
    YOLOv2: {
      id: 11,
      key: 'yolov2',
      title: 'Yolo v2',
      params: {
        ANCHOR: {
          title: 'Anchor',
          key: 'anchor',
          type: 'number',
          default: 5,
        },
        TRAIN_WHOLE: {
          title: 'Train Whole Network',
          key: 'train_whole',
          type: 'checkbox',
          default: false,
        },
        BATCH_SIZE: {
          title: 'Batch Size',
          key: 'batch_size',
          type: 'number',
          default: 32,
          min: 4,
          max: 128,
        },
        IMAGE_WIDTH: {
          title: 'Image Width',
          key: 'imsize_w',
          type: 'number',
          default: 448,
          min: 64,
          max: 512,
        },
        IMAGE_HEIGHT: {
          title: 'Image Height',
          key: 'imsize_h',
          type: 'number',
          default: 448,
          min: 64,
          max: 512,
        },
      }
    },
    SSD: {
      id: 12,
      key: 'ssd',
      title: 'SSD',
      params: {
        TRAIN_WHOLE: {
          title: 'Train Whole Network',
          key: 'train_whole',
          disabled: false,
          type: 'checkbox',
          default: false,
        },
        BATCH_SIZE: {
          title: 'Batch Size',
          key: 'batch_size',
          disabled: false,
          type: 'number',
          default: 32,
          min: 4,
          max: 128,
        },
        IMAGE_WIDTH: {
          title: 'Image Width',
          key: 'imsize_w',
          disabled: true,
          type: 'number',
          default: 300,
          min: 64,
          max: 512,
        },
        IMAGE_HEIGHT: {
          title: 'Image Height',
          key: 'imsize_h',
          disabled: true,
          type: 'number',
          default: 300,
          min: 64,
          max: 512,
        },
        TOTAL_EPOCH: {
          title: 'Total Epoch',
          key: 'total_epoch',
          disabled: false,
          type: 'number',
          default: 160,
        },

      }
    },
  },
  SEGMENTATION: {
    Unet: {
      key: 'unet',
    },
    Fcn: 22,
    TernousNet: 23,
  }
}

export const SORTBY = {
  CLASSIFICATION: {
    MODEL_ID: {
      id: 0,
      key: 'model_id',
      title: 'Model ID',
    },
    VALID_RECALL: {
      id: 1,
      key: 'valid_recall',
      title: 'Valid Recall'
    },
    VALID_PRECISION: {
      id: 2,
      key: 'valid_precision',
      title: 'Valid Precision'
    },
    VALID_F1: {
      id: 3,
      key: 'valid_f1',
      title: 'Valid F1'
    },
    VALID_LOSS: {
      id: 4,
      key: 'valid_loss',
      title: 'Valid Loss'
    }
  },
  DETECTION: {
    MODEL_ID: {
      id: 10,
      key: 'model_id',
      title: 'Model ID',
    },
    VALID_MAP: {
      id: 11,
      key: 'valid_mAP',
      title: 'Valid mAP'
    },
    VALID_IOU: {
      id: 12,
      key: 'valid_iou',
      title: 'Valid IOU'
    },
    VALID_LOSS: {
      id: 13,
      key: 'valid_loss',
      title: 'Valid Loss'
    }

  },
  SEGMENTATION: {
  },
}

export const STATE = {
  CREATED: 0,
  RESERVED: 1,
  STARTED: 2,
  STOPPED: 3,
}

export const RUNNING_STATE = {
  STARTING: 0,
  TRAINING: 1,
  VALIDATING: 2,
  STOPPING: 3,
}

export const FILTER = {
  CLASSIFICATION: {
    VALID_RECALL: {
      key: 'valid_recall',
      title: 'Valid Recall',
      type: 'condition'
    },
    VALID_PRECISION: {
      key: 'valid_precision',
      title: 'Valid Precision',
      type: 'condition'
    },
    VALID_F1: {
      key: 'valid_precision',
      title: 'Valid Precision',
      type: 'condition'
    },
    VALID_LOSS: {
      key: 'valid_precision',
      title: 'Valid Precision',
      type: 'CONDITION'
    },
    ALGORITHM_NAME: {
      key: 'algorithm',
      title: 'Algorithm',
      type: 'SELECT_ALGORITHM'
    }
  },
}

export const FILTER_CONDITION = {
  CONDITION: {
    LESS_THAN: 0,
    EQUAL: 1,
    GRATER_THAN: 2,
  },
}

export function getKeyByValue (object, value) {
  return Object.keys(object).find(key => object[key] === value)
}

export function getKeyByValueIncludes (object, value) {
  return Object.keys(object).find(key => (Object.values(object[key]).some(v => value === v)))
}
