<template>
  <div class="rnc-input">
    <div class="input-value">
      <input
        v-if="!isTextarea"
        :type="inputType"
        :disabled="disabled"
        :class="{'form-warn': (invalidInput || length_error || value_error || text_type_error), 'checkbox': inputType === 'checkbox'}"
        :place-holder="placeHolder"
        v-model="internalValue"
        class="input-in"
      >
      <textarea
        v-else
        :disabled="disabled"
        :class="{'form-warn': (invalidInput || length_error || value_error || text_type_error)}"
        :place-holder="placeHolder"
        :rows="rows"
        v-model="internalValue"
        class="input-in"
      />
    </div>
  </div>
</template>

<script>
export default {
  props: {
    value: {
      type: [String, Number, Boolean],
      default: undefined
    },
    inputType: {
      type: String,
      default: function () { return undefined },
      validator: function (val) {
        return ['checkbox', 'text'].includes(val)
      }
    },
    invalidInput: {
      type: Boolean,
      default: false
    },
    disabled: {
      type: Boolean,
      default: false
    },
    onlyInt: {
      type: Boolean,
      default: false
    },
    onlyNumber: {
      type: Boolean,
      default: false
    },

    placeHolder: {
      type: String,
      default: ''
    },

    // for String type input
    inputMaxLength: {
      type: Number,
      default: undefined
    },
    inputMinLength: {
      type: Number,
      default: undefined
    },

    // for Number type input
    inputMaxValue: {
      type: Number,
      default: undefined
    },
    inputMinValue: {
      type: Number,
      default: undefined
    },

    // for textarea
    isTextarea: {
      type: Boolean,
      default: false
    },
    rows: {
      type: Number,
      default: 10
    }
  },
  data: function () {
    return {
      text_type_error: false,
      length_error: false,
      value_error: false,
      error_message: {
        type_error: '',
        value_error: '',
        length_error: '',
      }
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
    },
  },
  watch: {
    value: function () {
      if (this.inputType !== 'checkbox') {
        // C. for any text-input=================================================================
        this.error_message.type_error = ''
        this.error_message.value_error = ''
        this.error_message.length_error = ''
        this.text_type_error = false
        this.length_error = false
        this.value_error = false
        if (this.onlyNumber || this.onlyInt) {
          // B. for positive real numbers only============================================
          if (this.onlyNumber) {
            if (!String(this.value).match(/^([1-9]\d*|0)(\.\d+)?$/)) {
              this.text_type_error = true
              this.error_message.type_error = 'Please enter number.'
              this.emitRncInput()
              return
            }
          }
          if (this.onlyInt) {
            // A. for positive integers only=========================================
            if (String(this.value).match(/[^\d]/)) {
              this.text_type_error = true
              this.error_message.type_error = 'Please enter integers.'
              this.emitRncInput()
              return
            }
          }
          let max_value_error = false
          let min_value_error = false
          if (this.inputMaxValue !== undefined) {
            max_value_error = (this.value > this.inputMaxValue)
          }
          if (this.inputMinValue !== undefined) {
            min_value_error = (this.value < this.inputMinValue)
          }
          if (max_value_error || min_value_error) {
            this.value_error = true
            this.error_message.value_error = 'The value should be between ' + this.inputMinValue + ' and ' + this.inputMaxValue + '.'
            this.emitRncInput()
            return
          }
          // B. for positive real numbers only============================================
        }

        let max_length_error = false
        let min_length_error = false

        if (this.inputMaxLength !== undefined) {
          max_length_error = (this.value.length > this.inputMaxLength)
        }
        if (this.inputMinLength !== undefined) {
          min_length_error = (this.value.length < this.inputMinLength)
        }

        if (max_length_error && this.inputMinLength === undefined) {
          this.length_error = true
          this.error_message.length_error = 'The input length should be ' + this.inputMaxLength + ' characters or less.'
          this.emitRncInput()
          return
        } else if (max_length_error || min_length_error) {
          this.length_error = true
          this.error_message.length_error = 'The input length should be between ' + this.inputMinLength + ' and ' + this.inputMaxLength + '.'
          this.emitRncInput()
          return
        }

        this.emitRncInput()
        return
        // C. for any text-input=================================================================
      }
    }
  },
  methods: {
    emitRncInput: function () {
      let ret_message = ''
      Object.values(this.error_message).forEach((message) => {
        ret_message += message
      })

      this.$emit('rnc-input', {
        'value': this.value,
        'textTypeError': this.text_type_error,
        'valueError': this.value_error,
        'lengthError': this.length_error,
        'errorMessage': ret_message
      })
    }
  }
}
</script>

<style lang="scss" scoped>
@import './../../../../static/css/unified.scss';

// .vali-params {
//   color: $err_red;
//   font-size: $fs-small;
//   padding-top: 3px
// }
</style>
