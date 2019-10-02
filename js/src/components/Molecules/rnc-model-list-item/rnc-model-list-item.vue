<template>
  <div
    v-if="model && model.id"
    id="model-item"
    :class="{ isSelected: model.id === selectedModelId}"
    @click="$emit('clicked-model-item', model)"
  >
    <div
      id="model-color"
      :class="colorClass"
    />

    <div id="model-id">
      <div class="info-row">
        <rnc-key-value
          :key-text="'ID:'"
          :value-text="model.id"
          data-cy="model-list-model-id"
        />
        <span class="info-title">
          &nbsp;&nbsp;
        </span>
        <rnc-key-value
          :key-text="'Alg:'"
          :value-text="algorithmTitle"
        />
      </div>
      <div class="info-row">
        <span
          :class="{'sorted-item': isSortBy('LOSS')}"
          data-cy="model-list-loss"
        >
          {{ lastBatchLoss }}
        </span>
        <span class="info-title">
          &nbsp;/&nbsp;
        </span>
        <span
          :class="{'sorted-item': isSortBy('M1')}"
          data-cy="model-list-metric1"
        >
          {{ resultOfMetric1 }}
        </span>
        <span class="info-title">
          &nbsp;/&nbsp;
        </span>
        <span
          :class="{'sorted-item': isSortBy('M2')}"
          data-cy="model-list-metric2"
        >
          {{ resultOfMetric2 }}
        </span>
      </div>
    </div>
    <div id="model-buttons">
      <rnc-button-close
        v-if="!isDeployedModel && model.state != state_started"
        data-cy="model-list-delete-button"
        @click="$emit('rm-model', model)"
      />
      <div
        v-else-if="isDeployedModel"
        id="deploy-icon"
      >
        Deployed
      </div>
    </div>
  </div>
</template>

<script>
import { SORTBY, STATE } from './../../../const.js'
import RncButtonClose from './../../Atoms/rnc-button-close/rnc-button-close.vue'
import RncKeyValue from './../../Atoms/rnc-key-value/rnc-key-value.vue'

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
    lastBatchLoss: {
      type: [Number, String],
      default: 0
    },
    resultOfMetric1: {
      type: [Number, String],
      default: 0
    },
    resultOfMetric2: {
      type: [Number, String],
      default: 0
    },
    selectedModelId: {
      type: [Number, String],
      default: 0
    },
    algorithmTitle: {
      type: String,
      default: undefined
    },
    colorClass: {
      type: String,
      default: undefined
    },
    sortOrder: {
      type: Number,
      default: undefined
    }
  },
  computed: {
    state_started: function () {
      return STATE.STARTED
    }
  },
  created: function () {

  },
  methods: {
    isSortBy: function (key) {
      return this.sortOrder === SORTBY[key]
    }
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

.isSelected#model-item {
  border: solid 1px $blue;
}
#model-item:hover {
  background-color: $item-hover-color;
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
      color: $blue;
    }
  }
}
</style>
