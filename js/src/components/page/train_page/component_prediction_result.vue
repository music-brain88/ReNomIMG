<template>
  <component-frame :width-weight="8" :height-weight="7">
    <template slot="header-slot">
      Prediction Result
    </template>
    <div id="pager">
      <div class="pager-arrow">
        <i class="fa fa-caret-left" aria-hidden="true"></i>
      </div>
      <div class="pager-number">
        1
      </div>
      <div class="pager-arrow">
        <i class="fa fa-caret-right" aria-hidden="true"></i>
      </div>
    </div>
    <div id="img-container">
      <transition-group name="fade">
        <img v-for="item in getValidImages" :src="item" :key="item"/>
      </transition-group>
    </div>
  </component-frame>
</template>

<script>
import { mapGetters, mapState } from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ComponentPredictionResult',
  components: {
    'component-frame': ComponentFrame
  },
  computed: {
    ...mapState(['datasets']),
    ...mapGetters(['getSelectedModel']),
    getValidImages: function () {
      const model = this.getSelectedModel
      if (model) {
        const dataset = this.datasets.find(d => d.id === model.dataset_id)
        const valid_data = dataset.valid_data
        return valid_data.img.slice(0, 4)
      }
      return []
    }
  },
  created: function () {

  },
  methods: {
  }
}
</script>

<style lang='scss'>
#img-container{
  width: 100%;
  height: 95%;
  display: flex;
  img {
    width: 10%;
    height: 20%;
    flex-flow:row-reverse wrap;
  }
}
#pager {
  width: 100%;
  height: 5%;
  display: flex;
  align-content: center;
  .pager-arrow {
    height: 100%;
    i {
    }
  }
  .pager-number {
    height: 100%;
    border: solid 1px gray;
  }
}
</style>
