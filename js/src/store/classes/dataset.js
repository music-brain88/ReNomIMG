import { TASK_ID, STATE, RUNNING_STATE } from '@/const.js'

export class Dataset {
  constructor (task_id, name, ratio, description, test_dataset_id) {
    this.id = undefined
    this.task_id = task_id
    this.name = name
    this.description = description
    this.test_dataset_id = test_dataset_id

    // Followings will be loaded from server.
    this.valid_data = {}
    this.class_map = []

    // Cache page division.
    this.page = []

    /**
      This is a dictionary.
      {
        // 'class_map' is a list of class name. This is same to this.class_map
        class_map: ['dog', 'cat', ...],

        // 'class_ratio' is a ratio of each number of
        // each class object to total number of object.
        // So Sum(class_ratio) must be 1.
        class_ratio: [0.23, 0.12],

        // 'train_ratio' is a list.
        // This contains each class's train image ratio.
        train_ratio: [],

        // 'valid_ratio' is a list.
        // This contains each class's valid image ratio.
        valid_ratio: [],

        // 'test_ratio' is a list.
        // This contains each class's test image ratio.
        // So for each 'i',
        // (train_ratio[i] + valid_ratio[i] + test_ratio[i]) must be 1.
        test_ratio: [],

        // Number of train images.
        train_img_num: (int)

        // Number of valid images.
        valid_img_num: (int)

        // Number of test images.
        test_img_num: (int)
      }
    */
    this.class_info = {}
  }
  getValidTarget (index) {
    const task = this.task_id
    const vd = this.valid_data
    if (!vd) return
    if (task === TASK_ID.CLASSIFICATION) {
      return vd.target[index]
    } else if (task === TASK_ID.DETECTION) {
      const size_list = vd.size[index]
      let box_list = vd.target[index]

      box_list = box_list.map((b, index) => {
        const ow = size_list[0]
        const oh = size_list[1]
        const x = b.box[0] / ow
        const y = b.box[1] / oh
        const w = b.box[2] / ow
        const h = b.box[3] / oh
        return Object.assign({box: [
          x, y, w, h
        ]}, Object.keys(b).reduce((obj, k) => {
          if (k === 'box') {
            return obj
          } else {
            return Object.assign(obj, {[k]: b[k]})
          }
        }, {}))
      })

      return box_list
    } else if (task === TASK_ID.SEGMENTATION) {
      return {
        size: vd.size[index],
        name: vd.img[index],
      }
    }
  }
}

export class TestDataset {
  constructor (task_id, name, ratio, description) {
    this.id = undefined
    this.task_id = task_id
    this.name = name
    this.description = description

    // Followings will be loaded from server.
    this.test_data = {}
    this.class_map = []
  }
}
