<template>
  <select
    id="rnc-select"
    :class="{'form-warn':invalidInput}"
    :disabled="disabled"
    v-model="internalValue"
  >
    <option
      disabled
      value=""
      selected
      class="default-select-item"
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
        if (this.value !== newVal) this.$emit('input', newVal)
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
    emitChangeSelect (e) {
      this.$emit('change-rnc-select', e.target.value)
    },
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
      }
    }
  }
}
</script>

<style lang="scss" scoped>
@import './../../../../static/css/unified.scss';
#rnc-select {
  width: 100%;
}
</style>
