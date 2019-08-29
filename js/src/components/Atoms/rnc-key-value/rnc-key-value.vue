<template>
  <div class="rcn-key-value">
    <p
      :class="{'key-value-margin': (keyText && valueText)}"
      class="key-text"
    >
      {{ keyText }}
    </p>
    <p class="value-text">
      {{ valueText }}
    </p>
  </div>
</template>

<script>
export default {
  name: 'RncKeyValue',
  props: {
    keyText: {
      type: [String, Number, Boolean],
      default: undefined
    },
    valueText: {
      type: [String, Number, Boolean],
      default: undefined
    }
  },
  // computed: {
  //   key_text: function () {
  //     let key_t = this.keyText;
  //     if (typeof(key_t) === Boolean || typeof(key_t) === Number) key_t.toString();
  //     return key_t
  //   },
  //   value_text: function () {
  //     let value_t = this.valueText;
  //     if (typeof(value_t) === Boolean || typeof(value_t) === Number) value_t.toString();
  //     return value_t
  //   }
  // }
}
</script>

<style lang="scss" scoped>
@import './../../../../static/css/unified.scss';

.rcn-key-value {
  display: flex;
  font-family: $component-font-family;
  .key-text {
    text-align: center;
    color: $gray;
  }
  /* keyとvalue両方指定された場合のみマージンを入れます */
  .key-value-margin {
    margin-right: 10px;
  }
  .value-text {
    text-align: center;
    color: $black;
  }
}
</style>
