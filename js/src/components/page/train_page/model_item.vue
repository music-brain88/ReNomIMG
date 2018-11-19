<template>
  <div id="model-item">
    <div id="model-add-button" v-if="isAddButton" @click="showModal({add_both: true})">
      ADD
    </div>
    <!-- <div id="model-id"  @click='setSelectedModel(model)' v-else> -->
    <div id="model-id"  @click='removeModel(model.id)' v-else>
      ID: {{ model.id }}
      ALGO: {{ getAlgorithmTitleFromId(model.algorithm_id) }}
      LOSS: {{ getLastBatchLoss }}
      STATE: {{ model.state }}
      RUN_STATE: {{ model.running_state }}
      mAP: {{ getMetric1 }}
      IOU: {{ getMetric2 }}
      LOSS: {{ getMetric3 }}
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
        return loss.toFixed(3)
      } else {
        return '-'
      }
    },

    getMetric1 () {
      if (this.model.best_epoch_valid_result) {
        return this.model.best_epoch_valid_result.mAP
      } else {
        return '-'
      }
    },
    getMetric2 () {
      if (this.model.best_epoch_valid_result) {
        return this.model.best_epoch_valid_result.IOU
      } else {
        return '-'
      }
    },
    getMetric3 () {
      if (this.model.best_epoch_valid_result) {
        if (this.model.best_epoch_valid_result.loss) {
          return this.model.best_epoch_valid_result.loss.toFixed(3)
        }
      }
      return '-'
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
#model-item {
  width: calc(100% - #{$model-item-margin}*2);
  height: $model-item-height;
  min-height: $model-item-height-min;
  margin: $model-item-margin;
  background-color: red;

  #model-add-button {

  }
}

</style>
