<template>
  <div
    v-if="model && model.id"
    id="model-item"
    :class="{ isSelected: model.id === SelectedModelId}"
  >
    <div
      id="model-color"
      :class="getColorClass(model)"
    />
    <!--
    TODO muraishi: not using currently
    <div
      v-if="isAddButton"
      ref="addButton2"
      id="model-add-button"
      @click="$emit('show-modal')"
    >
      ADD
    </div> -->

    <div
      id="model-id"
      @click="$emit('clicked-model-id', model)"
    >
      <div class="info-row">
        <rnc-key-value
          :key-text="'ID:'"
          :value-text="model.id"
        />
        <span class="info-title">
          &nbsp;&nbsp;
        </span>
        <rnc-key-value
          :key-text="'Alg:'"
          :value-text="getAlgorithmTitleFromId(model.algorithm_id)"
        />
      </div>
      <div class="info-row">
        <span :class="{'sorted-item': isSortBy('LOSS')}">
          {{ LastBatchLoss }}
        </span>
        <span class="info-title">
          &nbsp;/&nbsp;
        </span>
        <span :class="{'sorted-item': isSortBy('M1')}">
          {{ ResultOfMetric1 }}
        </span>
        <span class="info-title">
          &nbsp;/&nbsp;
        </span>
        <span :class="{'sorted-item': isSortBy('M2')}">
          {{ ResultOfMetric2 }}
        </span>
      </div>
    </div>
    <div id="model-buttons">
      <rnc-button-close
        v-if="!isDeployedModel"
        @click="$emit('rm-model', model)"
      />
      <div
        v-else
        id="deploy-icon"
      >
        Deployed
      </div>
    </div>
    <div id="child-model">
      <model-item
        v-for="(item, key) in getChildModelList"
        :model="item"
        :key="key"
        :hierarchy="hierarchy+1"
      />
    </div>
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapActions, mapState } from 'vuex'
import { SORTBY } from './../../../const.js'
import RncButtonClose from './../../Atoms/rnc-button-close/rnc-button-close.vue'
import RncKeyValue from './../../Atoms/rnc-key-value/rnc-key-value.vue'

// ★時間がなく以下にメソッドなど残っていますが、親からデータやイベント発生はもらうように使ってください。

export default {
  name: 'RncModelListItem',
  components: {
    'rnc-button-close': RncButtonClose,
    'rnc-key-value': RncKeyValue
  },
  props: {
    model: {
      type: Object,
      require: true,
      default: function () {
        return {
          'algorithm_id': undefined,
          'id': undefined,
          'state': undefined
        }
      }
    },
    isDeployedModel: {
      type: Boolean,
      default: false
    },
    // TODO muraishi: not using currently
    // isAddButton: {
    //   type: Boolean,
    //   default: false
    // },
    isOpenChildModelList: {
      type: Boolean,
      default: false
    },
    hierarchy: {
      type: Number,
      default: 0
    },
    LastBatchLoss: {
      type: [Number, String],
      default: 0
    },
    ResultOfMetric1: {
      type: [Number, String],
      default: 0
    },
    ResultOfMetric2: {
      type: [Number, String],
      default: 0
    },
    SelectedModelId: {
      type: [Number, String],
      default: 0
    }
  },
  computed: {
    ...mapState([
      'sort_order'
    ]),
    ...mapGetters([
      'getAlgorithmTitleFromId',
      'getColorClass',
      'getDeployedModel'
    ]),
    getChildModelList: function () {
      if (this.isAddButton || this.hierarchy > 0) {
        return []
      } else {
        return this.model.model_list
      }
    },
    /*
    TODO muraishi: transfer to RncTrainPanelModelList
    getLastBatchLoss () {
      if (this.model.getBestLoss()) {
        return this.model.getBestLoss().toFixed(2)
      } else {
        return '-'
      }
    },
    */

    // TODO muraishi: solove it by props
    // isDeployedModel () {
    //   return this.model === this.getDeployedModel
    // }
  },
  created: function () {

  },
  methods: {
    ...mapMutations([
      'showModal',
      'showConfirm',
    ]),
    ...mapActions(['removeModel']),
    isSortBy: function (key) {
      return this.sort_order === SORTBY[key]
    }
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

.isSelected#model-item {
  border: solid 1px $component-header-sub-color;
}
#model-item:hover {
  background-color: $model-item-hover-color;
}

#model-item {
  position: relative;
  width: 100%;
  height: $model-item-height;
  min-height: $model-item-height-min;
  margin-bottom: $margin-micro;
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
    // TODO muraishi: 下消して、オリジナルに戻しました。
    // height: calc(100% - 10px);
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
      .rcn-key-value{
        font-size: 90%
      }
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
    position: absolute;
    bottom: 0;
    right: 0;
    display: flex;
    justify-content: space-around;
    align-items: flex-end;
    flex-direction: column;
    padding: 5px;
    #deploy-icon {
      font-size: 70%;
      font-weight: bold;
      color: $component-header-sub-color;
    }
  }
  #child-model {
    display: flex;
    flex-wrap: wrap;
    margin-top: $margin-micro;
    width: 100%;
    font-size: 85%;
    #model-item {
      width: 90%;
    }
  }
}
</style>
