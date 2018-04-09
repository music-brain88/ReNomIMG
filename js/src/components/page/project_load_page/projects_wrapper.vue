<template>
  <div id="projects-wrapper">
    <project-add-panel class="load-panel"
      @showAddProject="showAddProject"></project-add-panel>
    <div class="row panel-row">
      <project-panel class="col-lg-3"></project-panel>
      <project-panel class="col-lg-3"></project-panel>
      <project-panel class="col-lg-3"></project-panel>
      <project-panel class="col-lg-3"></project-panel>
      <project-panel class="col-lg-3"></project-panel>
      <project-panel class="col-lg-3"></project-panel>

      <!--
        <project-panel v-if="projects" v-for="(p, index) in projects"
          :key="index" :project="p" class="col-md-3 col-lg-3 load-panel">
        </project-panel>
      -->

    </div>
    <project-setting v-if="show_project_setting"
      @showAddProject="showAddProject"></project-setting>
  </div>
</template>

<script>
import ProjectAddPanel from './projects_wrapper_parts/project_add_panel.vue'
import ProjectPanel from './projects_wrapper_parts/project_panel.vue'
import ProjectSetting from './projects_wrapper_parts/project_setting.vue'

export default {
  name: 'ProjectsWrapper',
  components: {
    'project-add-panel': ProjectAddPanel,
    'project-panel': ProjectPanel,
    'project-setting': ProjectSetting
  },
  data: function () {
    return {
      show_project_setting: false
    }
  },
  created: function () {
    // this.$store.dispatch("loadProjects");
  },
  computed: {
    projects () {
      return this.$store.getters.getProjects
    }
  },
  methods: {
    showAddProject: function () {
      this.show_project_setting = !this.show_project_setting
    }
  }
}
</script>

<style lang="scss">

.panel-row {
  padding: 0px 0px 0px 20px;
}

</style>
