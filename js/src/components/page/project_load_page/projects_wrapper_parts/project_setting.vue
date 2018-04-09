<template>
  <div id="project-setting">
    <div id="modal-menu">
      <form>
        <legend>Project Setting</legend>
        <div class="input-group vertical">
          <label for="project_name">Project Name</label>
          <input type="text" id="project-name" v-model="project_name">
        </div>
  
        <div class="input-group vertical">
          <label for="project_comment">Project Comment</label>
          <input type="text" id="project-comment" v-model="project_comment">
        </div>
  
        <div class="input-group vertical">
          <label for="thumbnail_path">Thumbnail Image Name</label>
          <input type="text" id="thumbnail-path" v-model="thumbnail_path">
        </div>
        <button v-on:click="createProject">Submit</button>
        <button v-on:click="closeProjectSetting">Cancel</button>
      </form>
    </div>
  </div>
</template>

<script>
  export default {
    name: 'ProjectSetting',
    data: function () {
      return {
        project_name: '',
        project_comment: '',
        thumbnail_path: ''
      }
    },
    methods: {
      createProject () {
        let self = this
        let ret = this.$store.dispatch('createProject', {
          'project_name': this.project_name,
          'project_comment': this.project_comment,
          'thumbnail_path': this.thumbnail_path
        })
        ret.then(function (responce) {
          self.$router.push({ path: '/detection_page' })
        })
      },
      closeProjectSetting () {
        this.$parent.showAddProject()
      }
    }
  }
</script>

<style lang="scss">
  #project-setting {
    display: flex;
    position:fixed;
    top:0;
    left:0;
    width:100%;
    height:120%;
    z-index:1;
    background-color:rgba(0,0,0,0.75);
  }

  #modal-menu {
    width: 80%;
    margin: 10% auto;  
    z-index:2;
  }
</style>
