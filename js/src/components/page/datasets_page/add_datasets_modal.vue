<template>
  <div id="add-model-modal">
    <div class="modal-background" @click="hideAddModelModal"></div>
      <div class="modal-content">
        <div class="modal-body">
          <add-detasets></add-detasets>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import AddDetasets from '../detection_page/add_datasets.vue'
export default {
  components: {
    'add-detasets': AddDetasets
  },
  methods: {
    hideAddModelModal: function () {
      this.$store.commit('setAddModelModalShowFlag', {'add_model_modal_show_flag': false})
    }
  }
}
</script>

<style lang="scss" scoped>
@import '@/../node_modules/bootstrap/scss/bootstrap.scss';
#add-model-modal {
  $modal-color: #000000;
  $modal-opacity: 0.7;

  $modal-content-bg-color: #fefefe;
  $modal-content-padding: 32px;

  $content-margin: 8px;

  position: fixed;
  left: 0;
  top: $application-header-hight;
  width: 100vw;
  height: calc(100vh - #{$application-header-hight});
  z-index: 3;


  a{
    opacity: 1;
    transition: opacity 0s;
  }

  #nav-home-tab{
    display: block;
    width:205px;
    height:35px;
    background-color: $tab-bg-color;
    color: $font-color;
    border-radius: 0px;
    text-align: center;
    &.active{
      background-color: #FFFFFF;
      color: #000000;
    }
  }
  #nav-contact-tab{
    display: block;
    width:205px;
    height:35px;
    background-color: $tab-bg-color;
    color: $font-color;
    border-radius: 0px;
    text-align: center;
    &.active{
      background-color: #FFFFFF;
      color: #000000;
    }
  }

  nav {
    background: #ffffff;
    border: none;
    padding: 0;
  }
  @media (min-width: 576px){
    .modal-dialog{
      max-width: 70%;
    }
  }

  .modal-background {
    width: 100%;
    height: 100%;
    background-color: $modal-color;
    opacity: $modal-opacity;
  }
  .modal-content {
    display: flex;
    flex-direction: column;

    position: absolute;
    top: 50%;
    left: 50%;
    -webkit-transform: translateY(-50%) translateX(-50%);
    transform: translateY(-50%) translateX(-50%);
    padding-left: $content-horizontal-padding;
    padding-right: $content-horizontal-padding;
    padding-top: $content-top-padding;
    padding-bottom: $content-bottom-padding;
    width: calc(#{$modal-content-width} * #{$max-width}); //896
    height:$modal-content-height;
    background-color: $content-bg-color;
    opacity: 1;
    border-radius: 0;
  }
  .modal-body{
    padding: 0;
    background-color: $content-bg-color;
  }
}
</style>
