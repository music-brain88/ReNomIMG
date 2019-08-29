<template>
  <div id="modal-add-model">
    <div id="generals">
      <div id="dataset-select">
        <div class="title">
          Dataset
        </div>
        <div class="subtitle">
          <div class="label">
            Dataset Name
          </div>
          <rnc-select
            :option-info="getFilteredDatasetList"
            :get-id="true"
            v-model="selected_dataset_id"
            class="input-value"
          >
            <template slot="default-item">
              Select Dataset
            </template>
          </rnc-select>
        </div>
        <div class="title">
          Algorithm
        </div>
        <div class="subtitle">
          <div class="label">
            {{ getCurrentTaskTitle }} Algorithm
          </div>
          <rnc-select
            :option-info="getAlgorithmList"
            v-model="selected_algorithm"
            class="input-value"
          >
            <template slot="default-item">
              Select Algorithm
            </template>
          </rnc-select>
        </div>
      </div>
    </div>
    <div id="params">
      <div
        id="hyper-params"
        class="title"
      >
        Hyper parameters
      </div>
      <div class="form-set-field">
        <div
          v-for="(item, itemkey) in getAlgorithmParamList(selected_algorithm)"
          :key="itemkey"
          class="form-set"
        >
          <div class="hyper-param">
            <div class="label">
              {{ item.title }}
            </div>
            <div
              v-if="item.type !== 'select'"
              class="input-value"
            >
              <rnc-input
                v-model="parameters[item.key]"
                :disabled="item.disabled"
                :input-type="item.type==='checkbox'? 'checkbox' : 'text'"
                :class="{'checkbox': item.type==='checkbox'}"
                :input-min-value="item.min"
                :input-max-value="item.max"
                :only-int="true"
                @rnc-input="onUpdateParams($event, itemkey)"
              />
            </div>
            <div v-else>
              <rnc-select
                :option-info="item.options"
                v-model="parameters[item.key]"
              />
            </div>
          </div>
          <div
            v-if="vali_params[itemkey]"
            class="vali-mes"
          >
            {{ vali_params[itemkey] }}
          </div>
        </div>
      </div>
    </div>
    <div id="button-area">
      <rnc-button
        :disabled="isRunnable"
        button-label="Create"
        @click-button="onCreateModel"
      />
    </div>
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapActions } from 'vuex'
import RncButton from '../../Atoms/rnc-button/rnc-button.vue'
import RncSelect from '../../Atoms/rnc-select/rnc-select.vue'
import RncInput from '../../Atoms/rnc-input/rnc-input.vue'

export default {
  name: 'RncModalModel',
  components: {
    'rnc-button': RncButton,
    'rnc-select': RncSelect,
    'rnc-input': RncInput
  },
  data: function () {
    return {
      selected_dataset_id: '',
      selected_algorithm: '',
      vali_params: {},
      parameters: {},
      running_disabled: '',
      paramKeys: {}
    }
  },
  computed: {
    ...mapGetters([
      'getCurrentTask',
      'getCurrentTaskTitle',
      'getAlgorithmList',
      'getAlgorithmParamList',
      'getAlgorithmIdFromTitle',
      'getFilteredDatasetList'
    ]),
    isRunnable () {
      if (this.selected_dataset_id !== '' && this.selected_algorithm !== '') {
        if (this.running_disabled) {
          return true
        } else {
          return false
        }
      } else {
        return true
      }
    }
  },
  watch: {
    selected_algorithm: function () {
      this.setDefaultValue(this.getAlgorithmParamList(this.selected_algorithm))
    },
    selected_dataset_id: function () {
      // TODO: console.log('selected_dataset_id', this.selected_dataset_id)
    }
  },
  methods: {
    ...mapActions(['createModel']),
    ...mapMutations(['showModal']),

    onUpdateParams: function (params, itemkey) {
      this.$set(this.vali_params, itemkey, params['errorMessage'])
      for (const k in this.vali_params) {
        this.running_disabled = false
        if (this.vali_params[k]) {
          this.running_disabled = true
          return this.running_disabled
        }
      }
      this.parameters[itemkey] = params['value']
      this.isRunnable
    },

    setDefaultValue: function (params) {
      this.parameters = Object.keys(params).reduce((obj, x) =>
        Object.assign(obj, { [params[x].key]: params[x].default }), {})
    },
    onCreateModel: function () {
      this.showModal({ 'all': false })
      this.createModel({
        hyper_params: this.parameters,
        algorithm_id: this.getAlgorithmIdFromTitle(this.selected_algorithm),
        dataset_id: this.selected_dataset_id,
        task_id: this.getCurrentTask
      })
    }
  }
}
</script>

<style lang="scss" scoped>
@import './../../../../static/css/unified.scss';

#modal-add-model {
  width: 100%;
  height: 100%;
  display: flex;
  flex-wrap: wrap;
  padding: 10px;
  font-size: $fs-regular;

  .title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: gray;
  }
  .label {
    color: $black;
  }

  #generals {
    width: 50%;
    height: calc(100% - 10px);
    .subtitle {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin: $margin-middle;
      .input-value{
        width: 55%;
      }
    }
  }

  #params {
    width: 50%;
    height: calc(100% - 10px);

    .form-set-field {
      overflow: auto;
      height: 90%;
      margin-top: 4px;
      .form-set {
        margin: $margin-middle;
        .hyper-param {
          display: flex;
          align-items: center;
          justify-content: space-between;
          .input-value{
            width: 55%;
          }
        }
        .vali-mes {
          margin-top: $margin-micro;
          color: $red;
          font-size: $fs-small;
          text-align: right;
        }
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
