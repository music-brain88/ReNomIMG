import Model from './model.js'

export default class Project {
  constructor(project_id, project_name, project_comment) {
    this.project_id = project_id;
    this.project_name = project_name;
    this.project_comment = project_comment;
    this.deploy_model_id = undefined;
  }
}
