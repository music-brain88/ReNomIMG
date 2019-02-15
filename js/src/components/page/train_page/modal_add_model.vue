<template>
  <div id="modal-add-model">
    <div id="generals">
      <div id="dataset-select">
        <div class="title">Dataset</div>
        <div class="subtitle">Dataset Name
          <select v-model="selectedDatasetId">
            <option
              disabled
              value=""
              selected>Select Dataset</option>
            <option
              v-for="(item, key) in getFilteredDatasetList"
              :key="key"
              :value="item.id"> {{ item.name }} </option>
          </select>
        </div>
      </div>
      <div class="title">Algorithm</div>
      <div class="subtitle">{{ getCurrentTaskTitle }} Algorithm
        <select
          v-model="selectedAlgorithm"
          @change="setDefaultValue(getAlgorithmParamList(selectedAlgorithm))">
          <option
            disabled
            value=""
            selected>Select Algorithm</option>
          <option
            v-for="(item, index) in getAlgorithmList"
            :key="index">{{ item }}</option>
        </select>
      </div>
    </div>
    <div id="params">
      <div
        id="hyper-params"
        class="title">Hyper parameters</div>
      <div
        v-for="(item, key) in getAlgorithmParamList(selectedAlgorithm)"
        :key="key">
        <div class="hyper-param">{{ item.title }}
          <input
            v-if="item.type !== 'select'"
            :type="item.type"
            :placeholder="item.default"
            v-model="parameters[item.key]"
            :disabled="item.disabled"
            :min="item.min"
            :max="item.max">
          <select
            v-else
            v-model="parameters[item.key]"
            :selected="item.default">
            <option
              v-for="(opt, key) of item.options"
              :key="key">{{ opt }}</option>
          </select>
        </div>
      </div>
    </div>
    <div id="button-area">
      <input
        :disabled="isRunnable"
        type="button"
        value="Create"
        @click="onCreateModel">
    </div>
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapState, mapActions } from 'vuex'

export default {
  name: 'ModalAddModel',
  data: function () {
    return {
      selectedAlgorithm: '',
      selectedDatasetId: '',
      parameters: {},
    }
  },
  computed: {
    ...mapState(['show_modal']),
    ...mapGetters([
      'getCurrentTask',
      'getCurrentTaskTitle',
      'getAlgorithmList',
      'getAlgorithmParamList',
      'getAlgorithmIdFromTitle',
      'getFilteredDatasetList'
    ]),
    isRunnable () {
      if (this.selectedDatasetId !== '' && this.selectedAlgorithm !== '') {
        const alg = this.selectedAlgorithm
        const params = this.getAlgorithmParamList(alg)
        for (const p in params) {
          const mn = params[p].min
          const mx = params[p].max
          const k = params[p].key
          let val = this.parameters[k]
          if (mn !== undefined && mx !== undefined && val !== undefined) {
            val = Number(val)
            if (mn > val || val > mx) {
              return true
            }
          }
        }
        return false
      } else {
        return true
      }
    }
  },
  methods: {
    ...mapActions(['createModel']),
    ...mapMutations(['showModal']),
    setDefaultValue: function (params) {
      // Reset if selected algorithm is changed.
      this.parameters =
        Object.keys(params).reduce((obj, x) =>
          Object.assign(obj, { [params[x].key]: params[x].default }), {})
    },
    onCreateModel: function () {
      this.showModal({ 'all': false })
      // Perform action 'createModel' with specified params.
      this.createModel({
        hyper_params: this.parameters,
        algorithm_id: this.getAlgorithmIdFromTitle(this.selectedAlgorithm),
        dataset_id: this.selectedDatasetId,
        task_id: this.getCurrentTask
      })
    },
  }
}
</script>

<style lang='scss'>
#modal-add-model {
  width: 100%;
  height: 100%;
  display: flex;
  flex-wrap: wrap;
  padding: 10px;
  font-size: 90%;

  .title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: gray;
  }

  #generals {
    width: 50%;
    height: calc(100% - 10px);
    .subtitle {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin: 4%;
    }
    select {
      background-color: white;
      width: 55%;
    }
  }

  #params {
    width: 50%;
    height: calc(100% - 10px);
    overflow: auto auto;
    .hyper-param {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin: 4%;
      input {
        width: 55%;
      }
    }
  }
  #button-area {
    display: flex;
    flex-direction: row-reverse;
    width: 100%;
  }
}
</style>
