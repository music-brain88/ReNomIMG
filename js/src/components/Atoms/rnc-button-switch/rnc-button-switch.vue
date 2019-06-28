<template>
  <label id="rnc-button-switch">
    <input
      v-model="checked_data"
      :disabled="disabled"
      class="checkbox"
      type="checkbox"
      v-on="$listeners"
    >
    {{ label }}
  </label>
</template>

<script>
export default {
  name: 'RncButtonSwitch',
  props: {
    disabled: {
      type: Boolean,
      default: false
    },
    checked: {
      type: Boolean,
      default: false
    },
    label: {
      type: String,
      default: 'undefined',
    }
  },
  data: function () {
    return {
      checked_data: false
    }
  },
  watch: {
    checked: function () {
      this.checked_data = this.checked
    }
  },
  mounted: function () {
    this.checked_data = this.checked
  }
}
</script>

<style lang="scss" scoped>
@import './../../../../static/css/unified.scss';
#rnc-button-switch {
  cursor: pointer;
  display: flex;
  align-items: center;
  font-family: $component-header-font-family;
  font-size: 90%;

  input {
    cursor: pointer;
    display: none;
    -webkit-appearance: none;
  }
  input[type="checkbox"] {
    content: "";
    display: block;
    height: 12px;
    width: 12px;
    border: 1px solid white;
    border-radius: 6px;
  }
  input[type="checkbox"]:checked {
    content: "";
    display: block;
    border: 1px solid white;
    background-color: white;
  }
  input[type="checkbox"]:disabled {
    content: "";
    display: block;
    border: 1px solid gray;
    background-color: gray;
  }
  input[type="checkbox"]:focus {
      outline:none;
  }
}
</style>
