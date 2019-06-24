<template>
  <!-- <div class="rnc-train-panel-tag-list"> -->
  <rnc-title-frame
    :width-weight="widthWeight"
    :height-weight="heightWeight"
  >
    <template slot="header-slot">
      Tag List
    </template>

    <template slot="content-slot">
      <div class="rnc-list">
        <rnc-list
          :item-array="getTagList"
          kind="tag"
        />
      </div>
    </template>
  </rnc-title-frame>
  <!-- </div> -->
</template>

<script>
import { mapGetters } from 'vuex'
import RncTitleFrame from './../../Molecules/rnc-title-frame/rnc-title-frame.vue'
import RncList from './../../Molecules/rnc-list/rnc-list.vue'

export default {
  name: 'RncTrainPanelTagList',
  components: {
    'rnc-title-frame': RncTitleFrame,
    'rnc-list': RncList
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
  computed: {
    ...mapGetters([
      'getSelectedModel', 'getDatasetFromId'
    ]),
    getTagList: function () {
      const model = this.getSelectedModel
      if (!model) {
        return []
      }
      const dataset = this.getDatasetFromId(model.dataset_id)
      if (dataset) {
        return dataset.class_map
      } else {
        return []
      }
    }
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

.rnc-list {
  padding: $padding-small;
  height: 100%;
}

</style>
