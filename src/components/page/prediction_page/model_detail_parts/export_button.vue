<template>
  <div id="export-button">
    <div class="export-button" :disabled="!predictModelId" @click="exportCSV">
      <span class="icon">
        <i class="fa fa-file-o" aria-hidden="true"></i>
      </span>
      Export csv
    </div>
  </div>
</template>

<script>
export default {
  name: "ExportButton",
  computed: {
    predictModelId: function() {
      return this.$store.state.predict_model_id;
    },
    csv: function() {
      return this.$store.state.csv;
    }
  },
  methods: {
    exportCSV: function(){
      let url = "/api/obj_detector/v1/projects/1/models/" + this.$store.state.predict_model_id + "/export_csv/" + this.csv
      window.open(url, "__blank");
    }
  }
}
</script>

<style lang="scss" scoped>
#export-button {
  $button-height: 32px;
  $button-border-color: #7F9DB5;

  position: absolute;
  bottom: 0;
  width: 100%;

  .export-button {
    width: 96%;
    height: $button-height;
    margin: 0;
    margin-left: 4%;

    background-color: #ffffff;
    color: $button-border-color;
    text-align: center;
    line-height: $button-height;
    border: 1px solid $button-border-color;
    border-radius: 4px;
    cursor: pointer;
  }
}
</style>

