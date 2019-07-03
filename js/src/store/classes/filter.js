export default class Filter {
  constructor (item, condition = null, threshold = null) {
    this.item = item
    this.condition = condition
    this.threshold = threshold
  }
  filter (task_filtered_model_list) {
    // / /// model list needs to be filtered by task.

    const item = this.item
    const condition = this.condition
    const threshold = this.threshold

    function filter_func (m) {
      const key = item.key
      if (item.type === 'condition') {
        const model_value = m.best_epoch_valid_result
        // 表示が四捨五入されているため、表示と合わせて比較処理を行います。
        const value = Number(model_value[key]).toFixed(2)
        // console.log('***[model_value[key] of filter_func]：', model_value[key])
        // console.log('***[value of filter_func]：', value)
        // console.log('***[parseFloat(threshold) of filter_func]：', parseFloat(threshold))

        if (condition === '>=') { // Less equal than
          if (model_value) {
            if (value) return value >= parseFloat(threshold)
            else return false
          } else {
            return false
          }
        } else if (condition === '==') { // Equan
          if (model_value) {
            if (value) return value === parseFloat(threshold).toFixed(2) // toFixedをして型を揃えています
            else return false
          } else {
            return false
          }
        } else if (condition === '<=') { // Grater equal than
          if (model_value) {
            if (value) return value <= parseFloat(threshold)
            else return true
          } else {
            return false
          }
        } else {
          throw new Error('Do not reach here. - Filter.condition')
        }
      } else if (item.type === 'select') {
        if (key === 'algorithm') {
          return threshold.id === m.algorithm_id
        }
      } else {
        throw new Error('Do not reach here. - Filter')
      }
    }
    return task_filtered_model_list.filter(filter_func)
  }
}
