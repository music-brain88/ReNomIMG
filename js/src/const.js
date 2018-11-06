export const TASK_ID = {
  CLASSIFICATION: 0,
  DETECTION: 1,
  SEGMENTATION: 2,
}

export const ALGORITHM_ID = {
  CLASSIFICATION: {
    ResNet: 1,
    DenseNet: 2,
    Vgg16: 3,
    Vgg19: 4,
    Inception1: 5,
    Inception2: 6,
    Inception3: 7,
    Inception4: 8,
  },
  DETECTION: {
    YOLOv1: 11,
    YOLOv2: 12,
    YOLOv3: 13,
    SSD: 14,
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
