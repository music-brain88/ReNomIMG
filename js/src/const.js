export default {
  // Tasks
  TASK: {
    CLASSIFICATION: 0,
    DETECTION: 1,
    SEGMENTATION: 2,
  },

  // Algorithms
  ALG: {
    CLASSIFICATION: {
      ResNet: 0,
      DenseNet: 1,
      Vgg16: 2,
      Vgg19: 3,
      Inception1: 4,
      Inception2: 5,
      Inception3: 6,
      Inception4: 7,
    },
    DETECTION: {
      YOLOv1: 0,
      YOLOv2: 1,
      YOLOv3: 2,
      SSD: 3,
    },
    SEGMENTATION: {
      Unet: 0,
      Fcn: 1,
      TernousNet: 2,
    }
  },

  // Model states.
  STATE: {
    CREATED: 0,
    RESERVED: 1,
    RUNNING: 2,
    STOPPED: 3,
  },

  // If the state is RUNNING, following are the more detailed states.
  RUNNING_STATE: {
    STARTING: 0,
    TRAINING: 1,
    VALIDATING: 2,
    STOPPING: 3,
  },

  SORTBY: {
    CLASSIFICATION: {
      VARID: {
        RECALL: 0,
        PRECISION: 1,
        F1: 2,
        LOSS: 3
      },
      TEST: {
        RECALL: 4,
        PRECISION: 5,
        F1: 6,
      }
    },
    DETECTION: {
      VARID: {
        MAP: 7,
        IOU: 8,
        LOSS: 9,
      },
      TEST: {
        MAP: 10,
        IOU: 11,
      }
    },
    SEGMENTATION: {
      VARID: {
        RECALL: 0,
        PRECISION: 1,
        F1: 2,
        LOSS: 3
      },
      TEST: {
        RECALL: 4,
        PRECISION: 5,
        F1: 6,
      }
    },
  }
}
