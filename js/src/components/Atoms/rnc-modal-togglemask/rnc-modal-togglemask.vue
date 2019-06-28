<template>
  <transition
    v-if="showModal"
    class= "rnc-modal-togglemask"
    name="modal"
  >
    <div id="modal-wrapper">
      <div
        id="modal-mask"
        @click="toggle"
      />

      <div id="modal-container">
        <slot name="modal-contents">
          default: Hello World!
        </slot>
        <!-- <add-both v-if="show_modal.add_both"/>
        <add-dataset v-if="show_modal.add_dataset"/>
        <add-filter v-if="show_modal.add_filter"/>
        <show-image v-if="show_modal.show_image"/>
        <show-prediction-image v-if="show_modal.show_prediction_image"/> -->
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
    }
  },
  data: function () {
    return {
      close_modal: false
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
    position: fixed;
    left: calc((100vw - 60vw) / 2);
    top: calc((100vh - 60vh) / 2);
    width: 60vw;
    height: 60vh;
    background-color: white;
    padding: $modal-content-padding;
  }
}
</style>
