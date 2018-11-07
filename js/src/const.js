export const TASK_ID = {
  CLASSIFICATION: 0,
  DETECTION: 1,
  SEGMENTATION: 2,
}

export const ALGORITHM = {
  CLASSIFICATION: {
    ResNet: {
      key: 'resnet'
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
      key: 'yolov1',
      title: 'Yolo v1',
      params: {
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
      }
    },
    YOLOv2: {
      key: 'yolov2',
      title: 'Yolo v2',
      params: {
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
      }
    },
  },
  SEGMENTATION: {
    Unet: 21,
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
  },
  SEGMENTATION: {
  },
}

export const STATE = {
  CREATED: 0,
  RESERVED: 1,
  RUNNING: 2,
  STOPPED: 3,
}

export const RUNNING_STATE = {
  STARTING: 0,
  TRAINING: 1,
  VALIDATING: 2,
  STOPPING: 3,
}

export function getKeyByValue (object, value) {
  return Object.keys(object).find(key => object[key] === value)
}

export function getKeyByValueIncludes (object, value) {
  return Object.keys(object).find(key => (Object.values(object[key]).some(v => value === v)))
}
