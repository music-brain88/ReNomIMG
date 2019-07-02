<template>
  <!-- <div class="rnc-train-panel-model-scatter"> -->
  <rnc-title-frame
    :width-weight="widthWeight"
    :height-weight="heightWeight"
  >
    <template slot="header-slot">
      Model Distribution
    </template>

    <template slot="content-slot">
      <rnc-grid-x-y
        :filtered-and-grouped-model-list-array="FilteredAndGroupedModelListArray"
        kind="model-scatter"
      />
    </template>
  </rnc-title-frame>
  <!-- </div> -->
</template>

<script>
import { mapGetters } from 'vuex'
import RncTitleFrame from './../../Molecules/rnc-title-frame/rnc-title-frame.vue'
import RncGridXY from './../../Molecules/rnc-grid-x-y/rnc-grid-x-y.vue'

export default {
  name: 'RncTrainPanelModelScatter',
  components: {
    'rnc-title-frame': RncTitleFrame,
    'rnc-grid-x-y': RncGridXY
  },
  props: {
    widthWeight: {
      type: Number,
      default: 12,
    },
    heightWeight: {
      type: Number,
      default: 5,
    }
  },
  data: function () {
    return {
      // AxisNameX: "Epoch [-]",
      // AxisNameY: "Loss [-]",
      FilteredAndGroupedModelListArray: []
    }
  },
  computed: {
    ...mapGetters([
      'getFilteredAndGroupedModelList'
    ])
  },
  watch: {
    getFilteredAndGroupedModelList: function () {
      this.FilteredAndGroupedModelListArray = this.getFilteredAndGroupedModelList
    }
  },
  mounted: function () {
    this.FilteredAndGroupedModelListArray = this.getFilteredAndGroupedModelList
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

</style>
