import Project from './classes/project'
import {TASK, ALG} from '../const.js'

export default {
  setCurrentTask(state, payload) {
    const task = payload.task
    assert(task in Object.values(TASK))
    state.current_task = task
  }
}
