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
