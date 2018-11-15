<template>
  <div id="modal-add-model">

    <select v-model="selectedAlgorithm"
      v-on:change="setDefaultValue(getAlgorithmParamList(selectedAlgorithm))">
      <option v-for="item in getAlgorithmList">{{ item }}</option>
    </select>

    <div v-for="item in getAlgorithmParamList(selectedAlgorithm)">
      <div>{{ item.title }} 
        <input :type="item.type"
          :placeholder="item.default"
          v-model="parameters[item.key]"
          :disabled="item.disabled"
          :min="item.min"
          :max="item.max">
      </div>
    </div>
    <button @click="onCreateModel">create</button>
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapState, mapActions } from 'vuex'

export default {
  name: 'ModalAddModel',
  components: {
  },
  computed: {
    ...mapState(['show_modal']),
    ...mapGetters([
      'getCurrentTask',
      'getAlgorithmList',
      'getAlgorithmParamList',
      'getAlgorithmIdFromTitle'
    ]),
  },
  data: function () {
    return {
      selectedAlgorithm: '-',
      parameters: {},
    }
  },
  created: function () {

  },
  methods: {
    ...mapActions(['createModel']),
    ...mapMutations(['showModal']),
    setDefaultValue: function (params) {
      // Reset if selected algorithm is changed.
      this.parameters =
        Object.keys(params).reduce((obj, x) =>
          Object.assign(obj, {[params[x].key]: params[x].default}), {})
    },
    onCreateModel: function () {
      this.showModal({'all': false})

      // Perform action 'createModel' with specified params.
      this.createModel({
        hyper_params: this.parameters,
        algorithm_id: this.getAlgorithmIdFromTitle(this.selectedAlgorithm),
        dataset_id: 1,
        parents: [],
        task_id: this.getCurrentTask
      })
    }
  }
}
</script>

<style lang='scss'>
#modal-add-model {
  width: 100%;
  height: 100%;
}
</style>
