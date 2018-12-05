<template>
  <div id="model-item" v-if="model" v-bind:class="{ isSelected: model === getSelectedModel}">
    <div id="model-color" v-bind:class='getColorClass(model)'>
    </div>
    <div id="model-add-button" v-if="isAddButton" @click="showModal({add_both: true})">
      ADD
    </div>
    <div id="model-id"  @click='setSelectedModel(model)' v-else>
      <div class="info-row">
        <span class="info-title">ID:</span>
        <span>{{ model.id }}</span>
        <span class="info-title">&nbsp;&nbsp;Alg:</span>
        <span>{{ getAlgorithmTitleFromId(model.algorithm_id) }}</span>
      </div>
      <div class="info-row">
        <span>{{ getLastBatchLoss }}</span>
        <span class="info-title">/</span>
        <span>{{ model.getResultOfMetric1().value }}</span>
        <span class="info-title">/</span>
        <span>{{ model.getResultOfMetric2().value }}</span>
      </div>
    </div>
    <div id="model-buttons">
      <i class="fa fa-cog" aria-hidden="true"></i>
      <i class="fa fa-times" aria-hidden="true" @click='removeModel(model.id)' v-if="!isDeployedModel"></i>
      <div id=deploy-icon v-else>
        Deployed
      </div>
    </div>
    <div id="child-model">
      <model-item v-for="item in getChildModelList" :model="item" :hierarchy="hierarchy+1"/>
    </div>
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapActions } from 'vuex'

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
    ...mapMutations(['showModal', 'setSelectedModel']),
    ...mapActions(['removeModel']),
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
      color: gray;
      padding-right: 5px;
    }
    #trush {
      align: right;
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
