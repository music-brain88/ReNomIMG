<template>
  <div id="model-item">
    <div id="model-add-button" v-if="isAddButton" @click="showModal({add_both: true})">
      ADD
    </div>
    <div id="model-id" v-else>
      ID: {{ model.id }}
      ALGO: {{ getAlgorithmTitleFromId(model.algorithm_id) }}
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
    model: Object,
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
    ...mapMutations(['showModal'])
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
