<template>
  <component-frame
    :width-weight="2"
    :height-weight="7">
    <template slot="header-slot">
      Tag List
    </template>
    <div
      id="tag-list"
      class="scrollbar-container">
      <tag-item
        v-for="(item, index) in getTagList"
        :tag-name="item"
        :tag-id="index"/>
    </div>
  </component-frame>
</template>

<script>
import { mapGetters } from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'
import TagItem from '@/components/page/train_page/tag_item.vue'

export default {
  name: 'ComponentTagList',
  components: {
    'component-frame': ComponentFrame,
    'tag-item': TagItem
  },
  computed: {
    ...mapGetters(['getTagList', 'getSelectedModel', 'getDatasetFromId']),
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
  },
  created: function () {

  },
  methods: {

  }
}
</script>

<style lang='scss'>
#tag-list {
  width: 100%;
  height: 100%;
  overflow: visible scroll;
  padding: 5px;
}
</style>
