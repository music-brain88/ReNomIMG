
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
