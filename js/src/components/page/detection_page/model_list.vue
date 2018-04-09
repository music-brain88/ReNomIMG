<template>
  <div id="model-list">
    <div class="title-selectbox">
      <div class="title">
        Model List
      </div>

      <select class="sort-selectbox" v-model="selected" @change="sortModels">
        <option disabled value="">-- sort by --</option>
        <option value="0">Model ID</option>
        <option value="1">IoU</option>
        <option value="2">mAP</option>
        <option value="3">Validation Loss</option>
      </select>
    </div>

    <div class="add-panel" @click="showAddModelModal">
      <i class="fa fa-plus" aria-hidden="true"></i> Add New Model
    </div>

    <div class="model-item-area">
      <model-list-item v-for="(s,index) in $store.state.models" :key="index" :model="s"></model-list-item>
    </div>
  </div>
</template>

<script>
import ModelListItem from './model_list_parts/model_list_item.vue'

export default {
  name: 'ModelList',
  components: {
    'model-list-item': ModelListItem
  },
  data: function () {
    return {
      selected: ''
    }
  },
  methods: {
    sortModels: function () {
      this.$store.commit('sortModels', {'sort_by': this.selected})
    },
    showAddModelModal: function () {
      this.$store.commit('setAddModelModalShowFlag', {
        'add_model_modal_show_flag': true
      })
    }
  }
}
</script>

<style lang="scss" scoped>
#model-list {
  $component-margin-top: 32px;

  $border-width: 2px;
  $border-color: #006699;

  $title-height: 44px;
  $title-font-size: 15pt;
  $font-weight-medium: 500;

  $content-padding: 24px;

  $content-bg-color: #ffffff;
  $content-border-color: #cccccc;

  $selectbox-height: 24px;
  $selectbox-margin-top: 12px;
  $selectbox-font-size: 12px;

  $add-panel-height: 86px;
  $add-panel-bg-color: #7F9DB5;
  $add-panel-bg-color-hover: #7590A5;

  height: 100%;
  margin: 0;
  margin-top: $component-margin-top;
  border-top: $border-width solid $border-color;

  .title-selectbox {
    display: flex;
    display: -webkit-flex;
    -webkit-justify-content: space-between;
    justify-content: space-between;
  }

  .title {
    line-height: $title-height;
    font-size: $title-font-size;
    font-weight: $font-weight-medium;
  }

  .sort-selectbox {
    height: $selectbox-height;
    margin: 0;
    margin-top: $selectbox-margin-top;
    padding: 0;
    padding-left: 4px;
    padding-right: 16px;
    font-size: $selectbox-font-size;
  }

  .model-item-area {
    height: calc(100% - #{$title-height} - #{$add-panel-height});
    margin-top: 12px;
    overflow: auto;
  }

  .add-panel {
    height: calc(#{$add-panel-height}*0.5);
    background-color: $add-panel-bg-color;
    color: #ffffff;
    text-align: center;
    line-height: calc(#{$add-panel-height}*0.5);
    border-radius: 4px;
    cursor: pointer;
  }
  .add-panel:hover {
    background-color: $add-panel-bg-color-hover;
  }
}
</style>

