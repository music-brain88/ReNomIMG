<template>
  <component-frame :width-weight="2" :height-weight="10"
    :style="{position: 'sticky', top: 10 + 'px'}" class="model-list">
    <template slot="header-slot">
      <div id="model-list-title">
        Model List
      </div>
      <div id="add-button" @click="showModal({add_both: true})">
        <i class="fa fa-plus" aria-hidden="true"></i>
      </div>
    </template>
    <div id="model-list" class="scrollbar-container">
      <model-item v-for="(model, index) in getFilteredModelList" :model="model"/>
    </div>
  </component-frame>
</template>

<script>
import { mapGetters, mapMutations} from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'
import ModelItem from '@/components/page/train_page/model_item.vue'

export default {
  name: 'ComponentModelList',
  components: {
    'component-frame': ComponentFrame,
    'model-item': ModelItem
  },
  computed: {
    ...mapGetters(['getFilteredModelList', 'getSortTitle']),
  },
  created: function () {

  },
  methods: {
    ...mapMutations(['setSortOrder', 'showModal']),
    addModel: function () {

    }
  }
}
</script>

<style lang='scss'>
#add-button {
  height: 100%;
  text-align: center;
  width: 30%;
  background-color: cyan;
}
#model-list {
  display: flex; 
  width: 100%;
  height: 100%;
  align-content: flex-start;
  flex-wrap: wrap;
  padding: $model-list-margin;
  overflow: visible scroll;
}

.model-list {
  #component-header {
    display: flex; 
    justify-content:space-between;
    #model-list-title {
      display: inline-block;
      height: 100%;
    }
    select {
      margin-left: auto;
    }
  }
}
.model-list#component-frame {
  #frame-content {
    background-color: transparent;
  }
}
</style>
