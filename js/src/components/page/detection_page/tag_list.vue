<template>
  <div id="tag-list">
    <div class="row">
      <div class="col-md-12 col-sm-12">
        <div class="title">
          <div class="title-text">
            Tags
          </div>
        </div>
        <div v-if="selected_model_id" class="content">
          <!-- tag-item -->
          <div class='tag-item' v-for="(name, id) in class_names">
            <div class="item-name">
                <span> &nbsp;{{ name }} </span>
            </div>
            <div class="box" v-bind:style="{backgroundColor: color_list[id%10]}">
              <span>{{id}}</span>
            </div>
          </div>
          <!-- tag-item -->
        </div>


        <div v-else class="content">
        </div>


      </div>
    </div>
  </div>
</template>

<script>

import { mapState, mapGetters } from 'vuex'
export default {
  name: 'TagList',
  data: function () {
    return {
      color_list: ['#E7009A', '#0A20C4', '#3E9AAF',
        '#FFCC33', '#EF8200', '#9F14C1', '#582396',
        '#8BAA1A', '#13894B', '#E94C33'
      ]
      // color_list: ['#E94C33', '#E7009A', '#0A20C4', '#3E9AAF',
      //   '#FFCC33', '#EF8200', '#9F14C1', '#582396',
      //   '#8BAA1A', '#13894B'
      // ]
    }
  },
  computed: {
    ...mapState([
      'dataset_defs',
      'selected_model_id',
      'class_names'
    ]),
    ...mapGetters([
      'getTagName'
    ])
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
    color: $font-color;
    .title-text{
      font-family: $content-top-header-font-family;
      font-size: $content-top-header-font-size;
      margin-left: $content-top-heder-horizonral-margin;
      line-height: $content-top-header-hight;
    }
  }

  .content {
    font-family: $content-inner-box-font-family;
    flex-wrap: wrap;
    margin-top: $content-top-margin;
    padding-top: $content-top-padding;
    padding-left: $content-horizontal-padding;
    padding-bottom:$content-bottom-padding;
    background-color: $content-bg-color;
    border: 1px solid $content-border-color;

    .FirstCol{
      border-left: 1px solid $content-taglist-tagbox-font-color;
    }
    .tag-item{
      display: inline-flex;
      box-sizing:border-box;
      font-size: $content-taglist-tagbox-font-size;
      color: $content-taglist-tagbox-font-color;
      padding-top: 2px;
      padding-bottom: 2px;
      width:177.5px;
      .item-name{
        height: 1rem;
        display: inline-flex;
        align-self: center;
        line-height: 1rem;
        border-left: 1px solid $content-taglist-tagbox-font-color;
        span{
          font-size: $content-taglist-tagbox-font-size;
          margin-left: calc(#{$content-taglist-tagbox-font-size}/2);
          align-self: center;
          width:calc(#{$content-taglist-tagbox-font-size}*6.5);
          overflow: hidden;
          text-overflow: ellipsis;
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
        margin-right: $content-horizontal-padding;
        span {
          width: 100%;
          text-align: center;
          align-self: center;
          // margin-right: 10px;
        }
      }
    }
  }
}
</style>
