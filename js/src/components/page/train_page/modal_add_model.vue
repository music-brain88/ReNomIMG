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
          v-model="parameters[item.title]"
          :disabled="item.disabled">
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapState } from 'vuex'

export default {
  name: 'ModalAddModel',
  components: {
  },
  computed: {
    ...mapState(['show_modal']),
    ...mapGetters(['getAlgorithmList', 'getAlgorithmParamList']),
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
    ...mapMutations(['showModal']),
    setDefaultValue: function (params) {
      // Reset if selected algorithm is changed.
      this.parameters =
        Object.keys(params).reduce((obj, x) =>
          Object.assign(obj, {[params[x].key]: params[x].default}), {})
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
