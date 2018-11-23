<template>
  <component-frame :width-weight="2" :height-weight="10"
    :style="{position: 'sticky', top: 10 + 'px'}" class="model-list">
    <template slot="header-slot">
      Model List
      <div id="add-button" @click="showModal({add_both: true})">
        <i class="fa fa-plus" aria-hidden="true"></i>&nbsp;New
      </div>
    </template>

    <div id="model-filter">
      Model Filter
      <i id="add-filter" class="fa fa-ellipsis-h"
        aria-hidden="true" @click="showModal({add_filter: true})"></i>
    </div>

    <!--
    <div id="model-groupby">
      Group by
      <select v-on:change="setGoupingCategory" v-model="groupby">
        <option :value="item.key" v-for='item of getGroupTitles'>{{item.title}}</option>
      </select>
    </div>
    -->

    <div id="model-titles">
      <span class="title-row">
        <div class="title selected">ID
          <i class="fa fa-sort-desc" aria-hidden="true"></i>
          &nbsp;&nbsp;&nbsp;
        </div>
        <div class="title">Alg
          <i class="fa fa-sort-desc" aria-hidden="true"></i>
        </div>
      </span>
      <span class="title-row">
        <div class="title">Loss
          <i class="fa fa-sort-desc" aria-hidden="true"></i>
        </div>
        <div class="title">/ mAP
          <i class="fa fa-sort-desc" aria-hidden="true"></i>
        </div>
        <div class="title">/ IOU
          <i class="fa fa-sort-desc" aria-hidden="true"></i>
        </div>
      </span>
    </div>

    <div id="model-list" class="scrollbar-container">
      <model-item :model="getDeployedModel"/>
      <model-item v-for="(model, index) in getFilteredAndGroupedModelList" :model="model"
         v-if="model !== getDeployedModel"/>
    </div>

  </component-frame>
</template>

<script>
import { mapGetters, mapMutations} from 'vuex'
import { GROUPBY } from '@/const.js'
import ComponentFrame from '@/components/common/component_frame.vue'
import ModelItem from '@/components/page/train_page/model_item.vue'

export default {
  name: 'ComponentModelList',
  components: {
    'component-frame': ComponentFrame,
    'model-item': ModelItem,
  },
  data: function () {
    return {
      groupby: '-'
    }
  },
  computed: {
    ...mapGetters(['getFilteredModelList', 'getFilteredAndGroupedModelList',
      'getSortTitle', 'getDeployedModel',
      'getGroupTitles']),
  },
  created: function () {

  },
  methods: {
    ...mapMutations(['setSortOrder', 'showModal', 'setGoupBy']),
    setGoupingCategory: function () {
      this.setGoupBy(this.groupby)
    }
  }
}
</script>

<style lang='scss'>

#add-button {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  width: 35%;
  background-color: $component-header-sub-color;
  cursor: pointer;
}

#model-filter,#model-groupby,#model-titles {
  display: flex;
  align-items: center;
  justify-content: space-between;

  height: calc(#{$model-filter-height}*0.6);
  min-height: calc(#{$model-filter-min-height}*0.6);
  width: 100%;

  padding: $model-list-margin;
  margin-bottom: $model-list-margin;
  font-size: 90%;
  color: gray;
  background-color: white;

  #add-filter {
    width: 20%;
    font-size: 110%;
    cursor: pointer;
  }
  #add-filter:hover {
    color: lightgray;
  }
}

#model-groupby {
  select {
    border: none;
    background: transparent;
    border: solid 1px #eeeeee;
    height: 100%;
    appearance: none;
    line-height: 120%;
    cursor: pointer;
  }
}

#model-titles {
  font-size: 70%;
  color: gray;
  display: block;
  height: calc(#{$model-filter-height}*0.8);
  min-height: calc(#{$model-filter-min-height}*0.8);

  .selected {
    color:  $component-header-sub-color;
    font-weight: bold;
  }

  .title-row {
    padding-left: 5px;
    display: flex;
    width: 100%;
    height: 50%;
  }
  .title {
    height: 100%;
    margin-left: 3px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
  }
  .title:hover {
    color: lightgray;
  }
  i {
    font-size: 150%;
    text-align: center;
    position: relative;
    top: -3px;
    margin-left: 3px;
  }
}

#model-list {
  display: flex; 
  align-content: flex-start;
  flex-wrap: wrap;
  width: 100%;
  height: 100%;
  overflow: visible scroll;
}

.model-list {
  #component-header {
    display: flex; 
    justify-content:space-between;
  }
}
.model-list#component-frame {
  #frame-content {
    background-color: transparent;
  }
}
</style>
