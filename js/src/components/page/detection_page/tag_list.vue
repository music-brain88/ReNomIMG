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
          <!-- tag-item -->
          <div class='tag-item' v-for="(name, id) in selectedModelTags">
            <div class="item-name">
                <span> &nbsp;{{ name }} </span>
            </div>
            <div class="box" v-bind:style="{backgroundColor: color_list[id%4]}">
              <span>{{id}}</span>
            </div>
          </div>
          <!-- tag-item -->

          <!-- tag-item -->
          <!-- <div class='tag-item' v-for="i in 10000">
            <div class="item-name">
                <span> &nbsp;{{ i }} </span>
            </div>
            <div class="box" v-bind:style="{backgroundColor: color_list[i%10]}">
              <span>{{i}}</span>
            </div>
          </div> -->
          <!-- tag-item -->
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
      color_list: ['#E94C33', '#E7009A', '#9F14C1',
        '#582396', '#0A20C4', '#3E9AAF', '#13894B',
        '#8BAA1A', '#FFCC33', '#EF8200'
      ]
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

  margin: 0;
  margin-top: $component-margin-top;
  // margin-left: $content-parts-margin;

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
    flex-wrap: wrap;
    // height: $content-taglist-hegiht;
    margin-top: $content-top-margin;
    padding: $content-top-padding $content-horizontal-padding $content-bottom-padding;
    background-color: $content-bg-color;
    border: 1px solid $content-border-color;

    .tag-item{
      display: inline-flex;
      box-sizing:border-box;
      font-size: $content-taglist-tagbox-font-size;
      color: $content-taglist-tagbox-font-color;
      padding-top: 2px;
      padding-bottom: 2px;
      width:165px;
      border-left: 1px solid $content-taglist-tagbox-font-color;
      .item-name{
        height: 1rem;
        display: inline-flex;
        align-self: center;
        line-height: 1rem;
        span{
          width: 100%;
          align-self: center;
        }
      }
      .box {
        font-family: $content-taglist-tagbox-font-family;
        font-size: $content-taglist-tagbox-font-size;
        display: inline-flex;
        align-self: center;
        height: 18px;
        width: 35px;
        color: #ffffff;
        margin-left: auto;
        margin-right: 4px;
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
