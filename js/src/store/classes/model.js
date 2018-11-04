export default class Model {
  constructor (id, task, hyper_parameters, dataset_id, children, parents) {
    // (Integer) ID of Model. This will defined by database.
    this.id = id

    this.total_epoch = 0
    this.last_epoch = 0

    this.total_batch = 0
    this.last_batch = 0

    this.loss_list = {
      'train': [],
      'valid': [],
    }

    this.best_epoch = 0
    this.best_valid_result = {}

    this.model_list = {
      'child': children,
      'parent': parents
    }
  }
  get child () {
    return this.model_list.child
  }
  get parent () {
    return this.model_list.parent
  }
}
