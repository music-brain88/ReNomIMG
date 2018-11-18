<template>
    <div id="filter-container" class="scrollbar-container">
      <filter-item v-for="item in getFilterList"/>
      <filter-item :is-add-button="true"/>
    </div>
</template>

<script>
import {mapGetters} from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'
import FilterItem from '@/components/page/train_page/filter_item.vue'

export default {
  name: 'ComponentModelFilter',
  components: {
    'component-frame': ComponentFrame,
    'filter-item': FilterItem
  },
  computed: {
    ...mapGetters(['getFilterList']),
  },
  created: function () {

  },
  methods: {

  }
}
</script>

<style lang='scss'>
  #filter-container {
    width: 100%;
    height: 100%;
    display: flex;
    padding: $model-filter-margin;
    align-content: flex-start;
    flex-wrap: wrap;
    overflow: visible scroll;
  }
</style>
