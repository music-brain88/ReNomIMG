<template>
  <div id="model-item" @click="testRun">
    <div id="model-add-button" v-if="isAddButton" @click="showModal({add_both: true})">
      ADD
    </div>
    <div id="model-id" v-else>
      ID: {{ model.id }}
      ALGO: {{ getAlgorithmTitleFromId(model.algorithm_id) }}
      Loss: {{ model.last_batch_loss.toFixed(3) }}
      Batch: {{ model.nth_batch }}
      State: {{ model.state }}
    </div>
  </div>
</template>

<script>
import { mapGetters, mapMutations } from 'vuex'

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
      default: false
    }
  },
  computed: {
    ...mapGetters([
      'getCurrentTask',
      'getModelResultTitle',
      'getAlgorithmTitleFromId',
    ]),
  },
  created: function () {

  },
  methods: {
    ...mapMutations(['showModal']),
    testRun: function () {
      if (!this.isAddButton) {
        this.$store.dispatch('runTrainThread', this.model.id)
      }
    }
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
