<template>
  <div class="outer-add-project-panel">
  <div id="project-add-panel"
    class="project-panel"
    v-on:click="showAddProject">
      <i class="fa fa-plus" aria-hidden="true" style="font-size:2rem;"></i>
  </div>
  </div>
</template>

<script>
  export default {
    name: "ProjectPlus",
    methods: {
      showAddProject: function() {
        // Fire parent(ProjectsWrapper vue)'s event.
        this.$emit('showAddProject')
      },
    },  
  }
</script>

<style lang="scss">

  .outer-add-project-panel {
    height: 100px;
    width: 200px;
    margin: 20px 0px 0px 20px;
    padding: 0px 0px 0px 0px;
    box-sizing: border-box;

    .project-panel {
      color: #4d4d4d;
      height: 100%;
      padding: 15px;
      border: 2px solid #eee;
      box-sizing:border-box;
      background-color: #ffffff;
    }

    .project-panel:hover {
      border: 2px solid #2d3e50;
    }
  }



</style>
