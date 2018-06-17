<template>
  <div id="tag-list">
    <div class="title">
      Tags
    </div>

    <div class="content">
      <div class='tag-item' v-for="(name, id) in selectedModelTags">
        <span> {{ name }} </span>
        <div class='box' v-bind:style="{backgroundColor: color_list[id%4]}"><span>{{id}}</span></div>
      </div>
    </div>
  </div>
</template>

<script>

import { mapState } from 'vuex'
export default {
  name: 'TagList',
  data: function () {
    return {
      color_list: ['#f19f36', '#53b05f', '#536cff', '#f86c8e']
    }
  },
  computed: {
    ...mapState([
      'dataset_defs',
      'selected_model_id'
    ]),
    selectedModelTags () {
      if (this.selected_model_id === undefined) {
        return
      }
      let model = this.$store.getters.getSelectedModel
      let dataset_id = model.dataset_id
      let selected_dataset_index = 0
      for (let index in this.dataset_defs) {
        if (this.dataset_defs[index].id === dataset_id) {
          selected_dataset_index = index
          break
        }
      }
      return this.dataset_defs[selected_dataset_index].class_map
    }
  }
}
</script>

<style lang="scss" scoped>
#tag-list {
  $component-margin-top: 32px;

  $border-width: 2px;
  $border-color: #006699;

  $title-height: 44px;
  $title-font-size: 15pt;
  $font-weight-medium: 500;

  $content-padding-top: 36px;
  $content-padding-horizontal: 32px;
  $content-padding-bottom: 36px;

  $content-bg-color: #ffffff;
  $content-border-color: #cccccc;

  margin: 0;
  margin-top: $component-margin-top;
  border-top: $border-width solid $border-color;

  .title {
    line-height: $title-height;
    font-size: $title-font-size;
    font-weight: $font-weight-medium;
  }

  .content {
    padding: $content-padding-top $content-padding-horizontal $content-padding-bottom;
    background-color: $content-bg-color;
    border: 1px solid $content-border-color;
    border-radius: 4px;

    .tag-item{
      display: flex;
      border-bottom: 1px solid #c3c3c3;
      box-sizing:border-box;
      font-size: 0.8rem;
      color: #5a5a5a;
      padding-top: 2px;
      padding-bottom: 2px;
      .box {
        display: flex;
        margin-left: auto;
        margin-right: 8px;
        border-radius: 2px;
        align-self: center;
        height: 1rem;
        width: 25px;
        color: #ffffff;
        span {
          width: 100%;
          text-align: center;
          align-self: center;
        }
      }

    }
  }
}
</style>

