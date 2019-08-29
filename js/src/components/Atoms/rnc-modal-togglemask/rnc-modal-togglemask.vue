<template>
  <transition
    v-if="showModal"
    name="modal"
  >
    <div
      id="modal-wrapper"
      :style="styles"
    >
      <div
        id="modal-mask"
        @click="toggle"
      />

      <div id="modal-container">
        <slot name="modal-contents">
          default: Hello World!
        </slot>
      </div>
    </div>
  </transition>
</template>

<script>
export default {
  name: 'RncModalTogglemask',
  props: {
    showModal: {
      type: Boolean,
      default: false
    },
    widthWeight: {
      type: Number,
      default: 7.5,
    },
    heightWeight: {
      type: Number,
      default: 9.5,
    }
  },
  data: function () {
    return {
      close_modal: false
    }
  },
  computed: {
    styles () {
      return {
        '--width-weight': this.widthWeight,
        '--height-weight': this.heightWeight
      }
    }
  },
  methods: {
    toggle: function () {
      const value = !this.showModal
      this.$emit('show-modal', value)
    },
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

#modal-wrapper {
  position: fixed;
  top: 0;
  left: 0;
  z-index: 10;
  #modal-mask {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: $modal-mask-color;
  }
  #modal-container {
    width: calc(var(--width-weight) * #{$component-block-width});
    height: calc(var(--height-weight) * #{$component-block-height});
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
    margin: auto;
    background-color: white;
    padding: $modal-content-padding;
  }
}
</style>
