<template>
  <div
    id="rnc-button"
    :class="{
      'cancel': cancel,
      'button-size': buttonSizeChange,
      'disabled': disabled
    }"
    type="button"
    @click="buttonClick"
  >
    <i
      v-if="plus"
      class="fa fa-plus plus-margin"
      aria-hidden="true"
    />
    {{ buttonLabel }}
  </div>
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
    },
    plus: {
      type: Boolean,
      default: false
    }
  },
  methods: {
    buttonClick: function () {
      if (!this.disabled) {
        this.$emit('click-button')
      }
    }
  }
}
</script>

<style lang="scss" scoped>
@import './../../../../static/css/unified.scss';

#rnc-button {
  display: flex;
  align-items: center;
  justify-content: center;

  height: 28px;
  width: 100px;
  background-color: $blue;
  color: white;
  border: none;
  padding: 0px;
  cursor: pointer;
  font-size: $fs-small;

  // Style Default
  &:hover {
    background-color: $little-dark-blue;
  }
  &:active {
    background-color: $light-blue;
  }
  &.button-size {
    width: 100%;
    height: 100%;
    padding: 0px;
  }

  // Style Cancel
  &.cancel {
    background-color: $gray;
    &:hover {
      background-color: $black;
    }
    &:active {
      background-color: $light-gray;
    }
  }
  .plus-margin {
    margin-right: $margin-micro;
    margin-top: 1px;
  }
  &.disabled {
    border: 1px solid $light-gray;
    background-color: $ex-light-gray;
    color: $gray;
    cursor:not-allowed;
    &:hover {
      background-color: $ex-light-gray;
    }
    &:active {
      background-color: $ex-light-gray;
    }
  }
}
</style>
