<template>
  <input
    id="rnc-button"
    :value="buttonLabel"
    :disabled="disabled"
    :class="{
      'cancel': cancel,
      'button-size': buttonSizeChange
    }"
    type="button"
    v-on="$listeners"
  >
</template>

<script>
export default {
  name: 'RncButton',
  props: {
    disabled: {
      type: Boolean,
      default: false
    },
    buttonLabel: {
      type: String,
      default: ''
    },
    cancel: {
      type: Boolean,
      default: false
    },
    buttonSizeChange: {
      type: Boolean,
      default: false
    }
  }
}
</script>

<style lang="scss" scoped>
@import './../../../../static/css/unified.scss';

</style>
