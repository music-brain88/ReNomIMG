export const TASK_ID = {
  CLASSIFICATION: 0,
  DETECTION: 1,
  SEGMENTATION: 2,
}

export const PAGE_ID = {
  DEBUG: -1,
  TRAIN: 0,
  PREDICT: 1,
  DATASET: 2,
}

const COMMON_PARAMS = {
  BATCH_SIZE: {
    title: 'Batch Size',
    key: 'batch_size',
    type: 'number',
    default: 16,
    min: 4,
    max: 128,
  },
  TOTAL_EPOCH: {
    title: 'Total Epoch',
    key: 'total_epoch',
    disabled: false,
    type: 'number',
    default: 160,
    min: 1,
    max: 1000,
  },
  IMAGE_WIDTH: {
    title: 'Image Width',
    key: 'imsize_w',
    type: 'number',
    disabled: false,
    default: 224,
    min: 64,
    max: 512,
  },
  IMAGE_HEIGHT: {
    title: 'Image Height',
    key: 'imsize_h',
    type: 'number',
    disabled: false,
    default: 224,
    min: 64,
    max: 512,
  },
  LOAD_PRETRAINED_WEIGHT: {
    title: 'Load pretrain weight',
    key: 'load_pretrained_weight',
    disabled: false,
    default: true,
    type: 'checkbox'
  },
  TRAIN_WHOLE: {
    title: 'Train Whole Network',
    key: 'train_whole', // Must be same as back-end.
    type: 'checkbox',
    default: false,
  },
}

function override (key, params) {
  return {[key]: {...Object.assign({...COMMON_PARAMS[key]}, params)}}
}

export const ALGORITHM = {
  CLASSIFICATION: {
    ResNet: {
      id: 1,
      key: 'resnet',
      title: 'ResNet',
      params: {
        ...COMMON_PARAMS,
        PLATEAU: {
          title: 'Plateau',
          key: 'plateau',
          type: 'checkbox',
          default: true
        },
        LAYER: {
          title: 'Number of Layers',
          key: 'layer',
          type: 'select',
          default: '34',
          options: ['18', '34', '50', '101', '152']
        }
      }
    },
    ResNext: {
      id: 2,
      key: 'resnext',
      title: 'ResNeXt',
      params: {
        ...COMMON_PARAMS,
        PLATEAU: {
          title: 'Plateau',
          key: 'plateau',
          type: 'checkbox',
          default: true
        },
        LAYER: {
          title: 'Number of Layers',
          key: 'layer',
          type: 'select',
          default: '50',
          options: ['50', '101']
        }
      }
    },

    /* Not available in v2.0
    DenseNet: {
      id: 3,
      key: 'densenet',
      title: 'DenseNet',
      params: {
        ...COMMON_PARAMS,
        LAYER: {
          title: 'Number of Layers',
          key: 'layer',
          type: 'select',
          default: "121",
          options: ["121", "169", "201"]
        }
      }
    }, */
    VGG: {
      id: 4,
      key: 'Vgg',
      title: 'VGG',
      params: {
        ...COMMON_PARAMS,
        LAYER: {
          title: 'Number of Layers',
          key: 'layer',
          type: 'select',
          default: '16',
          options: ['11', '16', '19']
        }
      }
    },
    /* Not available in v2.0
    Inception: {
      id: 5,
      key: 'Inception',
      title: 'Inception',
      params: {
        ...COMMON_PARAMS,
        VERSION: {
          title: 'Version',
          key: 'version',
          type: 'select',
          default: "1",
          options: ["1", "2", "3", "4"]
        }
      }
    },
    */
  },
  DETECTION: {
    YOLOv1: {
      id: 30, // Must be same as back-end.
      key: 'yolov1', // Must be same as back-end.
      title: 'Yolo v1',
      params: {
        ...COMMON_PARAMS,
        ...override('IMAGE_WIDTH', {
          default: 224,
        }),
        ...override('IMAGE_HEIGHT', {
          default: 224,
        }),
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
      id: 31,
      key: 'yolov2',
      title: 'Yolo v2',
      params: {
        ...COMMON_PARAMS,
        ...override('IMAGE_WIDTH', {
          disabled: true,
          default: 448,
        }),
        ...override('IMAGE_HEIGHT', {
          disabled: true,
          default: 448,
        }),
        ANCHOR: {
          title: 'Anchor',
          key: 'anchor',
          type: 'number',
          default: 5,
        },
      }
    },
    SSD: {
      id: 32,
      key: 'ssd',
      title: 'SSD',
      params: {
        ...COMMON_PARAMS,
        ...override('IMAGE_WIDTH', {
          disabled: true,
          default: 300,
        }),
        ...override('IMAGE_HEIGHT', {
          disabled: true,
          default: 300,
        })
      }
    },
  },
  SEGMENTATION: {
    Unet: {
      id: 60,
      key: 'unet',
      title: 'U-Net',
      params: {
        ...COMMON_PARAMS,
        ...override('IMAGE_WIDTH', {
          disabled: false,
          default: 512,
        }),
        ...override('IMAGE_HEIGHT', {
          disabled: false,
          default: 512,
        }),
        ...override('LOAD_PRETRAINED_WEIGHT', {
          title: 'Load pretrain weight',
          key: 'load_pretrained_weight',
          disabled: true,
          default: false,
          type: 'checkbox'
        }),
        ...override('TRAIN_WHOLE', {
          title: 'Train Whole Network',
          key: 'train_whole',
          type: 'checkbox',
          default: true,
        }),
      }
    },
    Fcn: {
      id: 61,
      key: 'fcn',
      title: 'FCN',
      params: {
        ...COMMON_PARAMS,
        ...override('LOAD_PRETRAINED_WEIGHT', {
          title: 'Load pretrain weight',
          key: 'load_pretrained_weight',
          disabled: false,
          default: false,
          type: 'checkbox'
        }),
        ...override('TRAIN_WHOLE', {
          title: 'Train Whole Network',
          key: 'train_whole',
          type: 'checkbox',
          default: true,
        }),
        ...override('BATCH_SIZE', {
          title: 'Batch Size',
          key: 'batch_size',
          type: 'number',
          default: 1,
          min: 1,
          max: 4,
        }),
        LAYER: {
          title: 'Number of Layers',
          key: 'layer',
          type: 'select',
          default: '16',
          options: ['8', '16', '32']
        }
      }
    },
    TernousNet: {
      id: 64,
      key: 'ternousnet',
      title: 'TernousNet',
      params: {
        ...COMMON_PARAMS,
        ...override('IMAGE_WIDTH', {
          disabled: false,
          default: 512,
        }),
        ...override('IMAGE_HEIGHT', {
          disabled: false,
          default: 512,
        })
      }
    },
  }
}

export const SORTBY = {
  ID: 0,
  ALG: 1,
  LOSS: 2,
  M1: 3,
  M2: 4,
}

export const SORT_DIRECTION = {
  ASCENDING: 0,
  DESCENDING: 1,
}

export const STATE = {
  CREATED: 0,
  RESERVED: 1,
  STARTED: 2,
  STOPPED: 3,

  PRED_CREATED: 4,
  PRED_RESERVED: 5,
  PRED_STARTED: 6,
}

export const RUNNING_STATE = {
  PREPARING: 0,
  TRAINING: 1,
  VALIDATING: 2,
  PREDICTING: 3,
  STARTING: 4,
  STOPPING: 5,
  WEIGHT_DOWNLOADING: 6,
}

export const FILTER = {
  CLASSIFICATION: {
    VALID_RECALL: {
      // Must be equal to the response of server key. "best_epoch_valid_result.key"
      key: 'recall',
      title: 'Valid Recall',
      type: 'condition',
      min: 0,
      max: 1,
    },
    VALID_PRECISION: {
      // Must be equal to the response of server key. "best_epoch_valid_result.key"
      key: 'precision',
      title: 'Valid Precision',
      type: 'condition',
      min: 0,
      max: 1,
    },
    /*
    VALID_F1: {
      // Must be equal to the response of server key. "best_epoch_valid_result.key"
      key: 'f1',
      title: 'Valid F1',
      type: 'condition',
      min: 0,
      max: 1,
    },
    */
    VALID_LOSS: {
      // Must be equal to the response of server key. "best_epoch_valid_result.key"
      key: 'loss',
      title: 'Valid Loss',
      type: 'condition',
      min: 0,
      max: 100,
    },
    ALGORITHM_NAME: {
      key: 'algorithm',
      title: 'Algorithm',
      type: 'select',
      options: Object.values(ALGORITHM.CLASSIFICATION)
    }
  },
  DETECTION: {
    VALID_MAP: {
      // Must be equal to the response of server key. "best_epoch_valid_result.key"
      key: 'mAP',
      title: 'Valid mAP',
      type: 'condition',
      min: 0,
      max: 1,
    },
    VALID_IOU: {
      // Must be equal to the response of server key. "best_epoch_valid_result.key"
      key: 'IOU',
      title: 'Valid IOU',
      type: 'condition',
      min: 0,
      max: 1,
    },
    VALID_LOSS: {
      // Must be equal to the response of server key. "best_epoch_valid_result.key"
      key: 'loss',
      title: 'Valid Loss',
      type: 'condition',
      min: 0,
      max: 100,
    },
    ALGORITHM_NAME: {
      key: 'algorithm',
      title: 'Algorithm',
      type: 'select',
      options: Object.values(ALGORITHM.DETECTION)
    }
  },
  SEGMENTATION: {
    VALID_RECALL: {
      // Must be equal to the response of server key. "best_epoch_valid_result.key"
      key: 'recall',
      title: 'Valid Recall',
      type: 'condition',
      min: 0,
      max: 1,
    },
    VALID_PRECISION: {
      // Must be equal to the response of server key. "best_epoch_valid_result.key"
      key: 'precision',
      title: 'Valid Precision',
      type: 'condition',
      min: 0,
      max: 1,
    },
    /*
    VALID_F1: {
      // Must be equal to the response of server key. "best_epoch_valid_result.key"
      key: 'f1',
      title: 'Valid F1',
      type: 'condition',
      min: 0,
      max: 1,
    },
    */
    VALID_LOSS: {
      // Must be equal to the response of server key. "best_epoch_valid_result.key"
      key: 'loss',
      title: 'Valid Loss',
      type: 'condition',
      min: 0,
      max: 100,
    },
    ALGORITHM_NAME: {
      key: 'algorithm',
      title: 'Algorithm',
      type: 'select',
      options: Object.values(ALGORITHM.SEGMENTATION)
    }
  }
}

export const GROUPBY = {
  NONE: {
    key: 'NONE',
    title: '-'
  },
  ALGORITHM: {
    key: 'ALGORITHM',
    title: 'Algorithm'
  },
  DATASET: {
    key: 'DATASET',
    title: 'Dataset',
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
