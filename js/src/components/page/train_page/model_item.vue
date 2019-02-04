<template>
  <div
    v-if="model"
    id="model-item"
    :class="{ isSelected: model === getSelectedModel}">
    <div
      id="model-color"
      :class="getColorClass(model)"/>
    <div
      v-if="isAddButton"
      id="model-add-button"
      @click="showModal({add_both: true})">
      ADD
    </div>
    <div
      v-else
      id="model-id"
      @click="setSelectedModel(model)">
      <div class="info-row">
        <span class="info-title">ID:</span>
        <span :class="{'sorted-item': isSortBy('ID')}">{{ model.id }}</span>
        <span class="info-title">&nbsp;&nbsp;Alg:</span>
        <span :class="{'sorted-item': isSortBy('ALG')}">
          {{ getAlgorithmTitleFromId(model.algorithm_id) }}
        </span>
      </div>
      <div class="info-row">
        <span :class="{'sorted-item': isSortBy('LOSS')}">{{ getLastBatchLoss }}</span>
        <span class="info-title">&nbsp;/&nbsp;</span>
        <span :class="{'sorted-item': isSortBy('M1')}">{{ model.getResultOfMetric1().value }}</span>
        <span class="info-title">&nbsp;/&nbsp;</span>
        <span :class="{'sorted-item': isSortBy('M2')}">{{ model.getResultOfMetric2().value }}</span>
      </div>
    </div>
    <div id="model-buttons">
      <i
        class="fa fa-cog"
        aria-hidden="true"/>
      <i
        v-if="!isDeployedModel"
        class="fa fa-times"
        aria-hidden="true"
        @click="rmModel(model.id)"/>
      <div
        v-else
        id="deploy-icon">
        Deployed
      </div>
    </div>
    <div id="child-model">
      <model-item
        v-for="item in getChildModelList"
        :model="item"
        :hierarchy="hierarchy+1"/>
    </div>
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapActions, mapState } from 'vuex'
import { GROUPBY, SORTBY, SORT_DIRECTION } from '@/const.js'

export default {
  name: 'ModelItem',
  components: {

  },
  props: {
    model: {
      type: Object,
      require: true
    },
    isAddButton: {
      type: Boolean,
      default: false,
    },
    isOpenChildModelList: {
      type: Boolean,
      type: false,
    },
    hierarchy: 0,
  },
  computed: {
    ...mapState([
      'sort_order'
    ]),
    ...mapGetters([
      'getCurrentTask',
      'getModelResultTitle',
      'getAlgorithmTitleFromId',
      'getColorClass',
      'getSelectedModel',
      'getDeployedModel'
    ]),
    getChildModelList: function () {
      if (this.isAddButton || this.hierarchy > 0) {
        return []
      } else {
        return this.model.model_list
      }
    },
    getLastBatchLoss () {
      if (this.model.getBestLoss()) {
        return this.model.getBestLoss().toFixed(2)
      } else {
        return '-'
      }
    },
    isDeployedModel () {
      return this.model === this.getDeployedModel
    }
  },
  created: function () {

  },
  methods: {
    ...mapMutations([
      'showModal',
      'setSelectedModel',
      'showConfirm',
    ]),
    ...mapActions(['removeModel']),
    isSortBy: function (key) {
      return this.sort_order === SORTBY[key]
    },
    rmModel: function (model) {
      const func = this.removeModel
      this.showConfirm({
        message: "Are you sure to <span style='color: #f00;}'>remove</span> this model?",
        callback: function () { func(model) }
      })
    }
  }
}
</script>

<style lang='scss'>
.isSelected#model-item {
  border: solid 1px $component-header-sub-color;
}
#model-item:hover {
  background-color: $model-item-hover-color;
}

#model-item {
  width: 100%;
  height: $model-item-height;
  min-height: $model-item-height-min;
  margin-bottom: $model-item-margin;
  display: flex;
  flex-wrap: wrap;
  background-color: white;
  cursor: pointer;
  #model-color {
    width: 3%;
    height: 100%;
  }
  #model-id {
    width: calc(72% - 5px);
    height: 100%;
    margin-left: 5px;
    padding-left: 5px;
    padding-top: 5px;
    padding-bottom: 5px;
    color: gray;
    font-size: 0.9rem;
    .info-row {
      height: 50%;
      display: flex;
      align-items: center;
      justify-content: flex-start;
    }
    span {
      font-size: 85%;
    }
    .info-title {
      height: 50%;
      color: #999;
      padding-right: 5px;
    }
    #trush {
      align: right;
    }
    .sorted-item {
      color: black;
    }
  }
  #model-buttons {
    width: 25%;
    height: 100%;
    display: flex;
    justify-content: space-around;
    align-items: flex-end;
    flex-direction: column;
    padding-right: 5px;
    i {
      color: lightgray;
    }
    .fa-cog:hover {
      color: black;
      cursor: pointer;
    }
    .fa-times:hover {
      color: $model-item-remove-button-color;
      cursor: pointer;
    }
    .fa:active {
      color: gray;
    }
    #deploy-icon {
      font-size: 70%;
      font-weight: bold;
      color: $component-header-sub-color;
    }
  }
  #child-model {
    display: flex;
    flex-wrap: wrap;
    margin-top: $model-item-margin;
    width: 100%;
    font-size: 85%;
    #model-item {
      width: 90%;
    }
  }
}
</style>
