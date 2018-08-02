<template>
  <div id="tag-list">
    <div class="row">
      <div class="col-md-12 col-sm-12">
        <div class="title">
          <div class="title-text">
            Tags
          </div>
        </div>
        <div class="content">
          <div class='tag-item' v-for="(name, id) in selectedModelTags">
            <div class="item-name">
              <span> {{ name }} </span>
            </div>
            <div class='box' v-bind:style="{backgroundColor: color_list[id%4]}">
              <span>{{id}}</span>
            </div>
          </div>
        </div>
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
      let dataset_def_id = model.dataset_def_id
      let selected_dataset_index = 0
      for (let index in this.dataset_defs) {
        if (this.dataset_defs[index].id === dataset_def_id) {
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
  margin-left: $content-parts-margin;

  .title {
    height: $content-top-header-hight;
    background: $header-color;
    font-size: $content-top-header-font-size;
    font-family: $content-top-header-font-family;
    color:$font-color;
    .title-text{
      margin-left: $content-top-heder-horizonral-margin;
      line-height: $content-top-header-hight;
    }
  }

  .content {
    font-family: $content-inner-box-font-family;
    display: flex;
    height: $content-taglist-hegiht;
    margin-top: $content-top-margin;
    padding: $content-padding-top $content-padding-horizontal $content-padding-bottom;
    background-color: $content-bg-color;
    border: 1px solid $content-border-color;

    .tag-item{
      display: flex;
      float:left;
      margin-left:$content-parts-margin;
      box-sizing:border-box;
      font-size: $content-inner-box-font-size;
      color: $content-taglist-tagbox-font-colot;
      padding-top: 2px;
      padding-bottom: 2px;
      width:100px;
      height: 1rem;
      .item-name{
        height: 1rem;
        display: inline-flex;
        align-self: center;
        line-height: 1rem;
        span{
          width: 100%;
          text-align: center;
          align-self: center;
        }
      }
      .box {
        font-family: $content-taglist-tagbox-font-family;
        font-size: $content-taglist-tagbox-font-size;
        display: inline-flex;
        margin-left:$content-parts-margin;
        margin-right: 8px;
        border-radius: 2px;
        align-self: center;
        height: 1.5rem;
        width: 30px;
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
