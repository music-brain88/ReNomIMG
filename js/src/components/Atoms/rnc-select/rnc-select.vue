<template>
  <!-- デフォルト値を初期設定にする場合は、その項目が必須項目の場合のため、form-warnも指定 -->
  <select
    id="rnc-select"
    v-model="internalValue"
    :class="{'form-warn': (invalidInput || value === ''), 'default-style': value === ''}"
    :disabled="disabled"
  >
    <option
      disabled
      value=""
      selected
    >
      <slot name="default-item" />
    </option>
    <option
      v-for="(item, key) in fixedOptionInfo"
      :key="key"
      :value="getId ? item.id : item.name"
      class="select-item"
    >
      {{ item.name }}
    </option>
  </select>
</template>

<script>
export default {
  name: 'RncSelect',
  props: {
    value: {
      type: [String, Number],
      default: ''
    },
    optionInfo: {
      type: [Array, Object],
      default: function () { return undefined }
    },
    invalidInput: {
      type: Boolean,
      default: false
    },
    disabled: {
      type: Boolean,
      default: false
    },
    getId: {
      type: Boolean,
      default: function () { return undefined }
    }
  },
  data: function () {
    return {
      fixedOptionInfo: []
    }
  },
  computed: {
    internalValue: {
      get () {
        return this.value
      },
      set (newVal) {
        if (this.value !== newVal) {
          this.$emit('input', newVal)
        }
      }
    }
  },
  watch: {
    optionInfo: function () {
      this.fixedOptionInfo = []
      this.convertOptionInfo()
    }
  },
  mounted () {
    this.convertOptionInfo()
  },
  methods: {
    convertOptionInfo: function () {
      if (this.optionInfo && this.optionInfo.length) {
        if (!this.optionInfo[0].name) {
          this.optionInfo.forEach(function (value, i) {
            const inOptionInfo = {
              'id': i,
              'name': value
            }
            this.fixedOptionInfo.push(inOptionInfo)
          }.bind(this))
        } else {
          this.fixedOptionInfo = this.optionInfo
        }
      } else if (this.optionInfo && Object.keys(this.optionInfo).length) {
        Object.keys(this.optionInfo).forEach((i) => {
          const inOptionInfo = {
            'id': i,
            'name': this.optionInfo[i]
          }
          this.fixedOptionInfo.push(inOptionInfo)
        })
      }
    }
  }
}
</script>

<style lang="scss" scoped>
@import './../../../../static/css/unified.scss';
#rnc-select {
  width: 100%;
  cursor: pointer;

  &:disabled {
    color: $gray;
    background-color: $ex-light-gray;
    border-color: $light-gray;
    cursor: not-allowed;
  }
}
.default-style {
  color: $light-gray;
  background-color: $ex-light-gray;
}
</style>
