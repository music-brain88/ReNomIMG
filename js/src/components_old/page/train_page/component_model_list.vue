<template>
  <!--TODO: Fix this top value-->
  <component-frame
    :width-weight="2"
    :height-weight="12"
    :style="{position: 'sticky', top: '55px', bottom: 'calc(100vh - 72px)'}"
    class="model-list">
    <template slot="header-slot">
      Model List
      <div
        id="add-button"
        @click="showModal({add_both: true})">
        <i
          class="fa fa-plus"
          aria-hidden="true"/>&nbsp;New
      </div>
    </template>

    <div id="model-filter">
      Model Filter
      <i
        id="add-filter"
        class="fa fa-ellipsis-h"
        aria-hidden="true"
        @click="showModal({add_filter: true})"/>
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
        <div
          :class="{selected: isSortBy('ID')}"
          class="title"
          @click="setOrder('ID')">
          ID
          <div class="sort-icon">
            <i
              v-if="isDescending && isSortBy('ID')"
              class="fa fa-sort-desc"
              aria-hidden="true"/>
            <i
              v-else-if="isSortBy('ID')"
              class="fa fa-sort-asc"
              aria-hidden="true"/>
          </div>
          &nbsp;&nbsp;&nbsp;
        </div>
        <div
          :class="{selected: isSortBy('ALG')}"
          class="title"
          @click="setOrder('ALG')">Alg
          <div class="sort-icon">
            <i
              v-if="isDescending && isSortBy('ALG')"
              class="fa fa-sort-desc"
              aria-hidden="true"/>
            <i
              v-else-if="isSortBy('ALG')"
              class="fa fa-sort-asc"
              aria-hidden="true"/>
          </div>
        </div>
      </span>
      <span class="title-row">
        <div
          :class="{selected: isSortBy('LOSS')}"
          class="title"
          @click="setOrder('LOSS')">Loss
          <div class="sort-icon">
            <i
              v-if="isDescending && isSortBy('LOSS')"
              class="fa fa-sort-desc"
              aria-hidden="true"/>
            <i
              v-else-if="isSortBy('LOSS')"
              class="fa fa-sort-asc"
              aria-hidden="true"/>
          </div>
        </div>
        <div
          :class="{selected: isSortBy('M1')}"
          class="title"
          @click="setOrder('M1')">
          /&nbsp;&nbsp;&nbsp;{{ getTitleMetric1 }}
          <div class="sort-icon">
            <i
              v-if="isDescending && isSortBy('M1')"
              class="fa fa-sort-desc"
              aria-hidden="true"/>
            <i
              v-else-if="isSortBy('M1')"
              class="fa fa-sort-asc"
              aria-hidden="true"/>
          </div>
        </div>
        <div
          :class="{selected: isSortBy('M2')}"
          class="title"
          @click="setOrder('M2')">
          /&nbsp;&nbsp;&nbsp; {{ getTitleMetric2 }}
          <div class="sort-icon">
            <i
              v-if="isDescending && isSortBy('M2')"
              class="fa fa-sort-desc"
              aria-hidden="true"/>
            <i
              v-else-if="isSortBy('M2')"
              class="fa fa-sort-asc"
              aria-hidden="true"/>
          </div>
        </div>
      </span>
    </div>

    <div id="deployed-model-area">
      <model-item
        v-if="getDeployedModel"
        :model="getDeployedModel"/>
      <div
        v-else
        id="empty">
        No model deployed
      </div>
    </div>
    <hr>
    <div
      id="model-list"
      class="scrollbar-container"
      tabindex="0"
      @keyup.right="selectNextModel"
      @keyup.left="selectPrevModel">
      <model-item
        v-for="(model, index) in getFilteredAndGroupedModelList"
        v-if="model !== getDeployedModel"
        :key="index"
        :model="model"/>
    </div>

  </component-frame>
</template>

<script>
import { mapGetters, mapMutations, mapState } from 'vuex'
import { SORTBY, SORT_DIRECTION } from '@/const.js'
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
    ...mapState([
      'sort_order',
      'sort_order_direction'
    ]),
    ...mapGetters([
      'getFilteredModelList',
      'getFilteredAndGroupedModelList',
      'getSortTitle',
      'getDeployedModel',
      'getGroupTitles',
      'getTitleMetric1',
      'getTitleMetric2'
    ]),
    isDescending: function () {
      return this.sort_order_direction === SORT_DIRECTION.DESCENDING
    },
  },
  created: function () {

  },
  methods: {
    ...mapMutations([
      'setSortOrder',
      'showModal',
      'setGoupBy',
      'selectPrevModel',
      'selectNextModel',
      'toggleSortOrder'
    ]),
    setGoupingCategory: function () {
      this.setGoupBy(this.groupby)
    },
    setOrder: function (key) {
      if (this.isSortBy(key)) {
        this.toggleSortOrder()
      } else {
        this.setSortOrder(key)
      }
    },
    isSortBy: function (key) {
      return this.sort_order === SORTBY[key]
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
  &:hover {
    background-color: $component-header-sub-color-hover;
  }
}

#model-filter,#model-groupby,#model-titles {
  display: flex;
  align-items: center;
  justify-content: space-between;

  height: calc(#{$model-filter-height}*0.6);
  min-height: calc(#{$model-filter-min-height}*0.6);
  width: 100%;

  padding: $model-list-margin;
  padding-left: 17px;
  margin-bottom: $model-list-margin;
  font-size: 90%;
  color: gray;
  background-color: white;

  #add-filter {
    width: 15%;
    height: 100%;
    font-size: 110%;
    cursor: pointer;
    position: relative;
    top: 3px;
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
    display: flex;
    width: 100%;
    height: 50%;
  }
  .title {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
  }
  .title:hover {
    color: lightgray;
  }
  .selected:hover {
    color:  $component-header-sub-color;
  }
  .sort-icon {
    width: 10px;
    height: 100%;
    i {
      font-size: 150%;
      text-align: center;
      margin-left: 2px;
    }
    .fa-sort-desc {
      position: relative;
      top: -3.5px;
    }
    .fa-sort-asc {
      position: relative;
      top: 3.5px;
    }
  }
}

#model-list {
  display: flex;
  align-content: flex-start;
  flex-wrap: wrap;
  width: 100%;
  max-height: calc(100% - #{$model-filter-height}*1.4 - #{$model-item-height} - 20px - 1rem);
  height: calc(100% - #{$model-filter-min-height}*1.4 - #{$model-item-height-min} - 20px - 1rem);
  overflow: auto;
}

#deployed-model-area {
  width: 100%;
  height: $model-item-height;
  margin-bottom: calc(#{$model-item-margin});
  #empty {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    border: dashed 2px lightgray;
    color: gray;
  }
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
