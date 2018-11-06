export default {
  // Tasks
  TASK: {
    CLASSIFICATION: 0,
    DETECTION: 1,
    SEGMENTATION: 2,
  },

  // Algorithms
  ALG: {
    YOLOv1: 0,
    YOLOv2: 1,
    YOLOv3: 2,
    SSD: 3,
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
  }
}
