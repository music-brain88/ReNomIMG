<template>
  <div class="modal fade" id="exampleModalCenter" tabindex="-1" role="dialog" aria-labelledby="exampleModalCenterTitle" aria-hidden="true">
    <div class="modal-dialog modal-dialog-centered modal-8" role="document">
      <div class="modal-content">
        <div class="modal-body">
          <nav>
            <div class="nav nav-tabs" id="nav-tab" role="tablist">
              <a class="nav-item nav-link active" id="nav-home-tab" data-toggle="tab" href="#nav-home" role="tab" aria-controls="nav-home" aria-selected="true">Setting of New Training Model</a>
              <a class="nav-item nav-link" id="nav-contact-tab" data-toggle="tab" href="#nav-contact" role="tab" aria-controls="nav-contact" aria-selected="false">Setting of Dataset</a>
            </div>
          </nav>

          <div class="tab-content" id="nav-tabContent">
            <div class="tab-pane fade show active" id="nav-home" role="tabpanel" aria-labelledby="nav-home-tab">
              <add-model-modal></add-model-modal>
            </div>
            <div class="tab-pane fade" id="nav-contact" role="tabpanel" aria-labelledby="nav-contact-tab">
              <add-detasets></add-detasets>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import AddModelModal from './add_model_modal.vue'
import AddDetasets from './add_detasets.vue'
require('jquery')
require('bootstrap')

export default {
  components: {
    'add-model-modal': AddModelModal,
    'add-detasets': AddDetasets
  }
}
</script>

<style lang="scss" scoped>
@import '@/../node_modules/bootstrap/scss/bootstrap.scss';

nav {
  background: #ffffff;
  border: none;

}
@media (min-width: 576px){
  .modal-dialog{
    max-width: 70%;
  }
}
</style>
