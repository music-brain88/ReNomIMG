<template>
  <rnc-title-frame
    :width-weight="widthWeight"
    :height-weight="heightWeight"
    :style="{position: 'sticky', top: '50px', bottom: 'calc(100vh - 72px)'}"
    class="model-list"
  >
    <template slot="header-slot">
      Model List
      <div
        id="add-button"
        @click="showModal({add_both: true})"
      >
        <i
          class="fa fa-plus"
          aria-hidden="true"
        />
        &nbsp;New
      </div>
    </template>

    <template slot="content-slot">
      <div id="model-filter">
        Model Filter
        <i
          id="add-filter"
          class="fa fa-ellipsis-h"
          aria-hidden="true"
          @click="showModal({add_filter: true})"
        />
      </div>

      <!--
      TODO muraishi : not using currently
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
            @click="setOrder('ID')"
          >
            ID
            <div class="sort-icon">
              <i
                v-if="isDescending && isSortBy('ID')"
                class="fa fa-sort-desc"
                aria-hidden="true"
              />
              <i
                v-else-if="isSortBy('ID')"
                class="fa fa-sort-asc"
                aria-hidden="true"
              />
            </div>
            &nbsp;&nbsp;&nbsp;
          </div>
          <div
            :class="{selected: isSortBy('ALG')}"
            class="title"
            @click="setOrder('ALG')"
          >
            Alg
            <div class="sort-icon">
              <i
                v-if="isDescending && isSortBy('ALG')"
                class="fa fa-sort-desc"
                aria-hidden="true"
              />
              <i
                v-else-if="isSortBy('ALG')"
                class="fa fa-sort-asc"
                aria-hidden="true"
              />
            </div>
          </div>
        </span>
        <span class="title-row">
          <div
            :class="{selected: isSortBy('LOSS')}"
            class="title"
            @click="setOrder('LOSS')"
          >
            Loss
            <div class="sort-icon">
              <i
                v-if="isDescending && isSortBy('LOSS')"
                class="fa fa-sort-desc"
                aria-hidden="true"
              />
              <i
                v-else-if="isSortBy('LOSS')"
                class="fa fa-sort-asc"
                aria-hidden="true"
              />
            </div>
          </div>
          <div
            :class="{selected: isSortBy('M1')}"
            class="title"
            @click="setOrder('M1')"
          >
            /&nbsp;&nbsp;&nbsp;{{ getTitleMetric1 }}
            <div class="sort-icon">
              <i
                v-if="isDescending && isSortBy('M1')"
                class="fa fa-sort-desc"
                aria-hidden="true"
              />
              <i
                v-else-if="isSortBy('M1')"
                class="fa fa-sort-asc"
                aria-hidden="true"
              />
            </div>
          </div>
          <div
            :class="{selected: isSortBy('M2')}"
            class="title"
            @click="setOrder('M2')"
          >
            /&nbsp;&nbsp;&nbsp; {{ getTitleMetric2 }}
            <div class="sort-icon">
              <i
                v-if="isDescending && isSortBy('M2')"
                class="fa fa-sort-desc"
                aria-hidden="true"
              />
              <i
                v-else-if="isSortBy('M2')"
                class="fa fa-sort-asc"
                aria-hidden="true"
              />
            </div>
          </div>
        </span>
      </div>

      <!-- TODO muraishi: not using RncList Molecule intentionally -->
      <div id="deployed-model-area">
        <rnc-model-list-item
          v-if="getDeployedModel"
          :model="deployedModelListItem.Model"
          :last-batch-loss="deployedModelListItem.LastBatchLoss"
          :result-of-metric1="deployedModelListItem.ResultOfMetric1"
          :result-of-metric2="deployedModelListItem.ResultOfMetric2"
          :selected-model-id="deployedModelListItem.SelectedModelId"
          :is-deployed-model="true"
          @clicked-model-item="updateSelectedModel($event)"
        />
        <!-- CHANGE muraishi -->
        <!-- @clicked-model-item="setSelectedModel($event)" -->

        <div
          v-else
          id="empty"
        >
          No model deployed
        </div>
      </div>

      <hr>

      <div
        id="model-list"
        class="scrollbar-container"
        tabindex="0"
        @keyup.right="selectNextModel"
        @keyup.left="selectPrevModel"
      >

        <!-- TODO muraishi: not using RncList Molecule intentionally -->
        <rnc-model-list-item
          v-for="(item, index) in ModelListItemArray"
          v-if="(item.Model.id !== deployedModelListItem.Model.id)"
          :key="index"

          :model="item.Model"
          :last-batch-loss="item.LastBatchLoss"
          :result-of-metric1="item.ResultOfMetric1"
          :result-of-metric2="item.ResultOfMetric2"
          :selected-model-id="item.SelectedModelId"

          @rm-model="rmModel($event.id)"
          @clicked-model-item="updateSelectedModel($event)"
        />
        <!-- CHANGE muraishi -->
        <!-- @clicked-model-item="setSelectedModel($event)" -->

      </div>
    </template>
  </rnc-title-frame>
</template>

<script>
import { mapActions, mapGetters, mapMutations, mapState } from 'vuex'
import RncTitleFrame from './../../Molecules/rnc-title-frame/rnc-title-frame.vue'
import RncList from './../../Molecules/rnc-list/rnc-list.vue'
import RncModelListItem from './../../Molecules/rnc-model-list-item/rnc-model-list-item.vue'
import { SORTBY, SORT_DIRECTION } from './../../../const.js'

export default {
  name: 'RncTrainPanelModelList',
  components: {
    'rnc-title-frame': RncTitleFrame,
    'rnc-list': RncList,
    'rnc-model-list-item': RncModelListItem
  },
  props: {
    widthWeight: {
      type: Number,
      default: 2.5,
    },
    heightWeight: {
      type: Number,
      default: 12,
    }
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
      'getSelectedModel',
      'getGroupTitles',
      'getTitleMetric1',
      'getTitleMetric2',
      'getModelById'
    ]),
    isDescending: function () {
      return this.sort_order_direction === SORT_DIRECTION.DESCENDING
    },
    deployedModelListItem: function () {
      // TODO: console.log('getDeployedModel in modelList', this.getDeployedModel)
      const ret = this.makeModelListItem(this.getDeployedModel)

      return ret
    },
    ModelListItemArray: function () {
      // TODO: console.log('getFilteredAndGroupedModelList in modelList', this.getFilteredAndGroupedModelList)
      const ret = this.getFilteredAndGroupedModelList.map((model) => {
        // TODO: console.log('loop')
        return this.makeModelListItem(model)
      })
      // TODO: console.log('***** ret:', ret)
      return ret
    },
  },

  mounted: function () {
    if (this.ModelListItemArray[0]) {
      this.updateSelectedModel(this.ModelListItemArray[0].Model)
    }
  },

  methods: {
    ...mapMutations([
      'setSortOrder',
      'showModal',
      'setSelectedModel',
      'showConfirm',
      'setGoupBy',
      'selectPrevModel',
      'selectNextModel',
      'toggleSortOrder',
    ]),
    // ADD muraishi
    ...mapActions([
      'removeModel',
      'updateSelectedModel',
      'loadModelsOfCurrentTaskDetail',
      'loadDatasetsOfCurrentTaskDetail']),

    // TODO muraishi : not using currently
    // setGoupingCategory: function () {
    //   this.setGoupBy(this.groupby)
    // },
    clickedModelItem: function (model) {
      this.loadModelsOfCurrentTaskDetail(model.id)
      this.loadDatasetsOfCurrentTaskDetail(model.dataset_id)

      // set selected_model form updated state.models
      const selected_model = this.getModelById(model.id)
      this.setSelectedModel(selected_model)
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
    },
    rmModel: function (model_id) {
      const func = this.removeModel
      this.showConfirm({
        message: "Are you sure you want to <span style='color: #f00;}'>remove</span> this model?",
        callback: function () { func(model_id) }
      })
    },
    makeModelListItem: function (model) {
      // TODO LastBatchLoss -> BestLoss ??
      let model_item = {}
      model_item = {
        'Model': {
          'algorithm_id': undefined,
          'id': undefined,
          'state': undefined
        },
        'LastBatchLoss': '-',
        'ResultOfMetric1': '-',
        'ResultOfMetric2': '-',
        'SelectedModelId': '-'
      }

      if (model) {
        // console.log("【model in makeModelListItem】")
        // console.log(model)
        model_item.Model = model

        model_item.LastBatchLoss = this.getLastBatchLoss(model)
        model_item.ResultOfMetric1 = model.getResultOfMetric1().value
        model_item.ResultOfMetric2 = model.getResultOfMetric2().value
        model_item.SelectedModelId = this.ensureSelectedModelId(this.getSelectedModel)
      }
      // console.log('【model_item in makeModelListItem】')
      // console.log(model_item)
      return model_item
    },
    getLastBatchLoss (model) {
      if (model.getBestLoss()) {
        return model.getBestLoss().toFixed(2)
      } else {
        return '-'
      }
    },
    ensureSelectedModelId (model) {
      let ret = 0
      if (model) {
        ret = model.id
      }
      return ret
    }
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

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
  min-height: $model-item-height-min;
  margin-bottom: $margin-micro;
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
  .component-header {
    display: flex;
    justify-content:space-between;
  }
}
.model-list {
  .frame-content {
    background-color: transparent;
  }
}
</style>
