<template>
  <div id="model-item" v-bind:class="{ isSelected: model===getSelectedModel}">
    <div id="model-color" v-bind:class='getAlgorithmClassFromId(model.algorithm_id)'>
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
        <span class="info-title">/</span>
        <span>{{ model.getResultOfMetric3().value }}</span>
      </div>
    </div>
    <div id="model-buttons">
      <i class="fa fa-cog" aria-hidden="true"></i>
      <i class="fa fa-times" aria-hidden="true" @click='removeModel(model.id)'></i>
    </div>
    <model-item v-for="item in getChildModelList" :model="item" :hierarchy="hierarchy+1"/>
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
      'getAlgorithmClassFromId',
      'getSelectedModel'
    ]),
    getChildModelList: function () {
      if (this.isAddButton || this.hierarchy > 0) {
        return []
      } else {
        return this.model.model_list
      }
    },
    getLastBatchLoss () {
      if (this.model.last_batch_loss) {
        const loss = this.model.last_batch_loss
        return loss.toFixed(2)
      } else {
        return '-'
      }
    },
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
  background-color: white;
  cursor: pointer;
  #model-color {
    width: 3%;
    height: 100%;
  }
  #model-id {
    width: 87%;
    height: 100%;
    margin-left: 5px;
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
    width: 10%;
    height: 100%;
    display: flex;
    justify-content: space-around;
    align-items: center;
    flex-direction: column;
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
  }
}
</style>
