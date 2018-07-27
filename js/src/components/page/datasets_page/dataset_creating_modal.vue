<template>
  <div id="dataset-creating-modal">
    <div class="modal-background"></div>
    <div class="modal-content">
      <i id="spinner" class="fa fa-spinner fa-spin"></i>
      Creating dataset, please wait...
    </div>
  </div>
</template>

<script>
export default {
  name: 'DatasetCreatingModal'
}
</script>

<style lang="scss" scoped>
#dataset-creating-modal {
  // $header-height: 35px;
  //
  // $modal-color: #000000;
  // $modal-opacity: 0.5;

  position: fixed;
  width: 100vw;
  height: 100vh;
  top: $header-height;
  left: 0;

  .modal-background {
    width: 100%;
    height: 100%;
    background-color: $modal-color;
    opacity: $modal-opacity;
  }

  .modal-content {
    position: absolute;
    top: 50%;
    left: 50%;
    -webkit-transform: translate(-50%) translateX(-50%);
    transform: translateY(-50%) translateX(-50%);
    color: #fff
  }
}
</style>
