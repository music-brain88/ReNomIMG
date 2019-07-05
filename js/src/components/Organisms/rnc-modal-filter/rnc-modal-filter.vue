<template>
  <div id="modal-add-filter">
    <div id="model-filter-title">
      Add Model Filter
    </div>
    <div id="model-filter-content">
      <div id="add-filter">
        <div id="item">
          <rnc-select
            :option-info="getOptionInfo(getFilterItemsOfCurrentTask)"
            v-model="item_input"
          >
            <template slot="default-item">
              FilterItem
            </template>
          </rnc-select>
        </div>

        <div id="condition">
          <rnc-select
            v-if="itemObject.type == 'select'"
            :key="1"

            :disabled="true"
            :option-info="['==']"
            v-model="condition"
          >
            <template slot="default-item">
              ==
            </template>
          </rnc-select>
          <rnc-select
            v-if="itemObject.type !== 'select'"
            :key="2"

            :option-info="['>=', '==', '<=']"
            v-model="condition"
          >
            <template slot="default-item">
              ==
            </template>
          </rnc-select>
        </div>

        <div id="value">
          <rnc-input
            v-if="itemObject.type !== 'select'"
            :key="value_input_key"

            :input-type="'text'"
            :input-min-length="input_min_length"
            :input-max-length="input_max_length"
            :input-min-value="itemObject.min"
            :input-max-value="itemObject.max"
            :only-number="true"
            :disabled="(item_input==='')"
            v-model="value_input"
            @rnc-input="error_message = $event.errorMessage"
          />
          <rnc-select
            v-if="itemObject.type == 'select'"
            :key="value_select_key"

            :option-info="getOptionInfo(itemObject.options)"
            v-model="value_input"
          >
            <template slot="default-item" />
          </rnc-select>
          <div
            v-if=" error_message != '' "
            class="vali-mes"
          >
            {{ error_message }}
          </div>
        </div>

        <div id="add">
          <rnc-button
            id="rnc-button"
            :button-label="'Add'"
            :disabled="addIsDisabled"
            @click="createFilter"
          />
        </div>
      </div>

      <div id="filter-list">
        <div
          v-for="(filterItem, key) in getFilterList"
          id="filter-item"
          :key="key"
        >
          <div
            v-if="filterItem.item.type === 'select'"
            class="select-item"
          >
            <div id="item">
              {{ filterItem.item.title }}
            </div>
            <div id="condition">
              ==
            </div>
            <div id="threshold">
              {{ filterItem.threshold.title }}
            </div>
            <rnc-button-close
              id="model-buttons"
              :display="true"
              @click="rmFilter(filterItem)"
            />
          </div>
          <div
            v-if="filterItem.item.type === 'condition'"
            class="condition-item"
          >
            <div id="item">
              {{ filterItem.item.title }}
            </div>
            <div id="condition">
              {{ filterItem.condition }}
            </div>
            <div id="threshold">
              {{ filterItem.threshold }}
            </div>
            <rnc-button-close
              id="model-buttons"
              :display="true"
              @click="rmFilter(filterItem)"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import Filter from './../../../store/classes/filter.js'
import { mapGetters, mapMutations } from 'vuex'
import RncButton from '../../Atoms/rnc-button/rnc-button.vue'
import RncSelect from '../../Atoms/rnc-select/rnc-select.vue'
import RncInput from '../../Atoms/rnc-input/rnc-input.vue'
import RncButtonClose from '../../Atoms/rnc-button-close/rnc-button-close.vue'
import { FILTER_INPUT_MAX_LENGTH, FILTER_INPUT_MIN_LENGTH } from './../../../const.js'

export default {
  name: 'RncModalFilter',
  components: {
    'rnc-button': RncButton,
    'rnc-select': RncSelect,
    'rnc-input': RncInput,
    'rnc-button-close': RncButtonClose
  },
  data: function () {
    return {
      value_input_key: 0,
      value_select_key: 0,
      item_input: '',
      value_input: '',
      condition: '==',
      threshold: '',
      itemObject: {
        item: 'select',
        type: 'condition',
        options: [],
        min: 0,
        max: 1,
      },
      error_message: '',
      input_min_length: FILTER_INPUT_MIN_LENGTH,
      input_max_length: FILTER_INPUT_MAX_LENGTH
    }
  },
  computed: {
    ...mapGetters([
      'getFilterItemsOfCurrentTask',
      'getFilterList',
    ]),
    addIsDisabled: function () {
      if (this.threshold === '' || this.threshold === undefined) {
        return true
      } else if (this.itemObject.type === 'condition' && isNaN(this.threshold)) {
        return true
      } else if (this.item_input === '' || this.value_input === '') {
        return true
      } else if (this.error_message !== '') {
        return true
      }
      return false
    },

  },
  watch: {
    item_input: function () {
      if (this.item_input === '') {
        return
      }
      this.itemObject = this.getSelectedObject(this.item_input, this.getFilterItemsOfCurrentTask)
      this.resetForm()
    },
    value_input: function () {
      if (this.value_input === '') {
        return
      }
      if (this.itemObject.type === 'select') {
        // this.value_input is a item name which was selected
        this.threshold = this.getSelectedObject(this.value_input, this.itemObject.options)
      } else {
        // this.value_input is a specific value
        this.threshold = this.value_input
      }
    }
  },
  methods: {
    ...mapMutations([
      'addFilter',
      'rmFilter'
    ]),
    getOptionInfo: function (option_obj) {
      if (!option_obj) return
      const ret = Object.values(option_obj).map(
        function (v) { return v.title }
      )

      return ret
    },
    createFilter: function () {
      if (this.addIsDisabled) return
      const filter = new Filter(this.itemObject, this.condition, this.threshold)
      this.addFilter(filter)
      this.resetForm()

      // item_input is the main option of indicators.
      this.item_input = ''
      this.itemObject = {
        title: 'select',
        type: 'condition',
        options: [],
        min: 0,
        max: 1,
      }
    },
    forceRerender: function (component_key) {
      // refrash input and select child components
      const val = component_key += 1
      return val
    },
    resetForm: function () {
      // reset all except this.item_input
      this.condition = '=='
      this.value_input = ''
      this.threshold = ''
      this.value_input_key = this.forceRerender(this.value_input_key)
      this.value_select_key = this.forceRerender(this.value_select_key)
      this.error_message = ''
    },
    getSelectedObject: function (e, aim_obj) {
      const selected_title = e
      const selected_obj = Object.values(aim_obj).filter(function (value) {
        return (value.title === selected_title)
      })
      return selected_obj[0]
    }
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

#modal-add-filter {
  width: 100%;
  height: 100%;
  padding: 8px;
  #model-filter-title {
    height: 7%;
    color: $component-font-color-title;
  }
  #model-filter-content {
    width: 70%;
    height: 93%;
    margin: 10px auto 0px;
    #add-filter {
      display: flex;
      align-items: flex-start;
      justify-content: center;
      width: 100%;
      height: 10%;
      #item {
        width: 40%;
        // height: 50%;
        *{
          width: 100%;
          // height: 100%;
        }
      }
      #condition {
        width: 10%;
        // height: 50%;
        margin-left: 5px;
        *{
          width: 100%;
          // height: 100%;
        }
      }
      #value {
        width: 40%;
        // height: 50%;
        margin-left: 5px;
        *{
          width: 100%;
          // height: 100%;
        }
        .vali-mes {
          margin-top: $margin-micro;
          color: $err_red;
          font-size: $fs-small;
          text-align: right;
        }
      }
      select {
        background-color: white;
      }
      #add {
        margin-left: 5px;
        #rnc-button {
          height: 21px;
        }
      }
    }
    #filter-list {
      font-size: $fs-regular;
      margin-top: 10px;
      width: 60%;
      height: calc(88% - 20px);
      overflow: auto;
      #filter-item {
        display: flex;
        width: 100%;
        height: 8%;
        .select-item {
          display: flex;
          width: 100%;
          height: 100%;
        }
       .condition-item {
          display: flex;
          width: 100%;
          height: 100%;
        }
        #item {
          width: 40%;
          margin-left: 10px;
          display: flex;
          align-items: center;
          justify-content: flex-start;
          // justify-content: center;
        }
        #condition {
          width: 10%;
          margin-left: 5px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        #threshold {
          width: 40%;
          margin-left: 5px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        #model-buttons {
          width: 10%;
          margin-left: 5px;
          display: flex;
          align-items: center;
          justify-content: center;
          // font-size: 1.1rem;
        }
      }
    }
  }
}
</style>
