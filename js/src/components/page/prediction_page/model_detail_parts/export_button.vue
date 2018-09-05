<template>
  <div id="export-button">
    <div class="export-button" :disabled="!predictModelId" @click="exportCSV">
      <span>
        <i class="fa fa-file-o icon" aria-hidden="true"></i>Export
      </span>
      prediction results as csv
    </div>
  </div>
</template>

<script>
export default {
  name: 'ExportButton',
  computed: {
    predictModelId: function () {
      if (this.$store.state.project) {
        return this.$store.state.project.deploy_model_id
      }
    }
  },
  methods: {
    exportCSV: function () {
      let url = '/api/renom_img/v1/projects/1/models/' + this.$store.state.project.deploy_model_id + '/export_csv/' + this.$store.state.csv
      window.open(url, '__blank')
    }
  }
}
</script>

<style lang="scss" scoped>
#export-button {
  $button-height: 32px;
 // position: absolute;
 // top: 200px;
  margin-left:20px;


  .icon{
    margin-right:10px;
  }

  .export-button {
    height: $button-height;
    margin: 0;
    padding-bottom: $content-bottom-padding;
    color:$panel-bg-color;
    // text-align: center;
    line-height: $button-height;
    cursor: pointer;
  }
}
</style>
