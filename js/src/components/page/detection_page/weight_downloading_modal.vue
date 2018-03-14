<template>
  <div id="weight-downloading-modal">
    <div class="modal-background"></div>
    <div class="modal-content">
      <i id='spinner' class="fa fa-spinner fa-spin"></i>
      Downloading weight parameters {{$store.state.weight_downloading_progress}} / 100 %, please wait...
    </div>
  </div>
</template>

<script>
export default {
  name: "WeightDownloadingModal",
}
</script>

<style lang="scss" scoped>
#weight-downloading-modal {
  $header-height: 35px;

  $modal-color: #000000;
  $modal-opacity: 0.7;

  position: fixed;
  width: 100vw;
  height: calc(100vh - #{$header-height});
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
    -webkit-transform: translateY(-50%) translateX(-50%);
    transform: translateY(-50%) translateX(-50%);
    color: #fff;
  }
}
</style>

