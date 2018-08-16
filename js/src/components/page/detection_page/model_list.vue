<template>
  <div id="model-list" class="list-child">
    <div class="add-panel" @click="showAddModelModal">
      <!-- <i class="fa fa-plus" aria-hidden="true"></i>  -->
      <img class="plus" :src="url">Add New Model
    </div>
    <div class="title-selectbox">
      <div class="title">
        Model List
      </div>
      <div class="select-wrapper">
        <select class="sort-selectbox" v-model="selected" @change="sortModels">
          <option disabled value="">sort by</option>
          <option class="select-values" value="0">Model ID</option>
          <option class="select-values" value="1">IoU</option>
          <option class="select-values" value="2">mAP</option>
          <option class="select-values" value="3">Validation Loss</option>
        </select>
      </div>
    </div>

    <div class="model-item-area">
      <model-list-item v-for="(s,index) in $store.state.models.filter(model => model.state !== 3)" :key="index" :model="s"></model-list-item>
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
      url: require('../../../../static/img/plus.png'),
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

  $selectbox-height: 20px;

  height: 650px;
  margin: 0;
  margin-top: $component-margin-top;
  // border-top: $border-width solid $border-color;
  z-index: 0;
  position: fixed;
  font-family: $content-top-header-font-family;
  font-size: $content-top-header-font-size;

  div::-webkit-scrollbar{
    width: 6px;
  }
  div::-webkit-scrollbar-track{
    background: $body-color;
    border: none;
    border-radius: 6px;
  }
  div::-webkit-scrollbar-thumb{
    background: #aaa;
    border-radius: 6px;
    box-shadow: none;
  }

  .title-selectbox {
    display: flex;
    display: -webkit-flex;
    -webkit-justify-content: space-between;
    justify-content: space-between;
    color: $font-color;
    background: $header-color;
    margin-top:$content-top-margin;
  }

  .title {
    line-height: $content-top-header-hight;
    font-family:$content-top-header-font-family;
    font-size:$content-top-header-font-size;
    margin-left: $content-top-heder-horizonral-margin;
  }

  .select-wrapper{
    position: relative;
    display: inline-block;
    margin-left:$content-parts-margin;
  }
  .select-wrapper::before{
    z-index: 1;
    position: absolute;
    right:25%;
    top: 0;
    font-family: "FontAwesome";
    content: '\f107';
    line-height: $content-top-header-hight;
    color: $font-color;
    pointer-events: none;
  }

  select {
    text-align: left;
    // text-align-last: center;
    -webkit-appearance: none;
	  -moz-appearance: none;
	  appearance: none;
    background: transparent;
    position: relative;
    /* webkit*/
  }


  .sort-selectbox {
    height: $selectbox-height;
    margin: 0;
    margin-top: calc(#{$content-top-header-hight}*0.225);
    margin-right: $content-top-heder-horizonral-margin;
    padding: 0;
    padding-left: 4px;
    padding-right: 16px;
    font-family: $content-modellist-font-family;
    font-size: $content-modellist-font-size;
    background: $header-color;
    color:$font-color;
  }

  .model-item-area {
    height: calc(100% - #{$content-top-header-hight} - #{$panel-height});
    width:100%;
    margin-top: 12px;
    overflow-y: scroll;
  }

  .add-panel {
    height: $panel-height;
    background-color: $panel-bg-color;
    color: $font-color;
    font-size: $content-top-header-font-size;
    text-align: center;
    line-height: $panel-height;
    cursor: pointer;
  }
  .add-panel:hover {
    background-color: $panel-bg-color-hover;
  }
  .plus{
    margin-right: 10px;
    vertical-align: middle;
    height:$content-top-header-font-size;
  }
}
</style>
