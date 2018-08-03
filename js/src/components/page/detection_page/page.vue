<template>
  <div>
    <div class="row">
      <div class="col-md-10 col-sm-12">
        <dashboard></dashboard>
        <model-detail></model-detail>
        <model-sample></model-sample>
        <tag-list></tag-list>
      </div>
      <div class="col-md-2 col-sm-12">
        <model-list></model-list>
      </div>
    </div>
    <image-modal v-if='show_modal_image_sample'/>

    <!-- <add-model-modal v-if="$store.state.add_model_modal_show_flag"></add-model-modal> -->
    <setting-page v-if="$store.state.add_model_modal_show_flag"></setting-page>
    <weight-downloading-modal v-if="$store.state.weight_downloading_modal"></weight-downloading-modal>
  </div>
</template>

<script>
import { mapState } from 'vuex'

import ModelList from './model_list.vue'
import DashBoard from './dashboard.vue'
import ModelDetail from './model_detail.vue'
import ModelSample from './model_sample.vue'
import TagList from './tag_list.vue'
// import AddModelModal from './add_model_modal.vue'
import WeightDownloadingModal from './weight_downloading_modal.vue'
import ImageModal from './image_modal.vue'
import DataSettingPage from './setting_page.vue'

export default {
  name: 'DetectionPage',
  components: {
    'model-list': ModelList,
    'dashboard': DashBoard,
    'model-detail': ModelDetail,
    'model-sample': ModelSample,
    'tag-list': TagList,
    'setting-page': DataSettingPage,
    'weight-downloading-modal': WeightDownloadingModal,
    'image-modal': ImageModal
  },
  computed: {
    ...mapState([
      'show_modal_image_sample'
    ])
  },
  created: function () {
    this.$store.dispatch('initData', {'project_id': 1})
    this.$store.dispatch('loadDatasetDef')
  }
}
</script>

<style lang="scss" scoped>
// #detection-page {
//   $header-height: 44px;
//   $page-margin-top: 22px;
//   $page-margin-horizontal: 72px;
//
//   $side-bar-width: 216px;
//   $side-bar-margin: 24px;
//
//   display: flex;
//   display: -webkit-flex;
//   flex-direction: column;
//   -webkit-flex-direction: column;
//
//   width: calc(100% - #{$page-margin-horizontal}*2);
//   margin: 0 $page-margin-horizontal;
//
//   .detection-content {
//     display: flex;
//     display: -webkit-flex;
//     height: 784px;
//
//     .detection-dashboard-and-detail {
//       width: calc(1 - #{$side-bar-width});
//       height: 100%;
//     }
//     .detection-page-sidebar {
//       width: $side-bar-width;
//       height: 100%;
//       margin-left: $side-bar-margin;
//     }
//   }
//
//   .detection-sample-and-tags {
//     display: flex;
//     display: -webkit-flex;
//     margin-bottom: 44px;
//
//     .model-sample {
//       width: calc(100% - #{$side-bar-margin} - #{$side-bar-width});
//     }
//
//   }
//
// }

</style>
