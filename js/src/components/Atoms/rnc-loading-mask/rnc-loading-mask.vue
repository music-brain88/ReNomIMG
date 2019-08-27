<template>
  <transition name="fade">
    <div
      v-if="showLoadingMask"
      id="rnc-loading-mask"
    >
      <div id="loader-container">
        <div class="loader" />
        <div class="loader" />
        <div class="loader" />
        <div class="loader" />
        <div class="loader" />
        <div class="message">
          {{ text }}
        </div>
      </div>
    </div>
  </transition>
</template>

<script>
export default {
  name: 'RncLoadingMask',
  props: {
    showLoadingMask: {
      type: Boolean,
      default: false
    },
    text: {
      type: String,
      default: 'Synchronizing to Server...'
    }
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';
#rnc-loading-mask {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100vh;
  z-index: 1000;
  background-color: $modal-loading-mask-color;
  #loader-container{
    display: flex;
    flex-wrap: wrap;
    width: 11%;
    height: 10%;
    position: relative;
    top: 45%;
    left: 45%;
    padding-left: 2px;
    padding-right: 2px;
    .loader {
      position: relative;
      top: 20%;
      width: calc(20% - 8px);
      height: 80%;
      margin-left: 4px;
      margin-right: 4px;
      background-color: white;
    }
    .loader:nth-child(1) {
      animation: 0.75s 0.0s linear infinite both loading;
    }
    .loader:nth-child(2) {
      animation: 0.75s 0.15s linear infinite both loading;
    }
    .loader:nth-child(3) {
      animation: 0.75s 0.3s linear infinite both loading;
    }
    .loader:nth-child(4) {
      animation: 0.75s 0.45s linear infinite both loading;
    }
    .loader:nth-child(5) {
      animation: 0.75s 0.6s linear infinite both loading;
    }
    .message {
      position: absolute;
      width: 100%;
      height: 20%;
      top: 110%;
      color: white;
      display: flex;
      justify-content: center;
      font-size: 0.7rem;
    }
    @keyframes loading {
      0% {
        top: 20%;
        height: 80%;
      }
      50% {
        top: 0%;
        height: 100%;
      }
      100% {
        top: 20%;
        height: 80%;
      }
    }
  }
}
</style>
