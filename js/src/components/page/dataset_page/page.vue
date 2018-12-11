<template>
  <div id="dataset-page">
    <div id="components">
      <component-frame :width-weight=12 :height-weight=9>
        <template slot="header-slot">
          Dataset
        </template>
      <div id="container">
        <div id="left-colmn">
          <div class="dataset-item" v-for="item in datasets" @click="current_dataset=item">
            <span>{{ item.id }}</span>
            <span>{{ item.name }}</span>
            <span>{{ item.ratio }}</span>
          </div>
          <div class="dataset-item" v-if="!datasets">
            <span></span>
            <span>No dataset</span>
            <span></span>
          </div>
        </div>
        <div id="right-colmn">
        </div>
      </div>
      </component-frame>
    </div>
  </div>
</template>

<script>
import { Dataset } from '@/store/classes/dataset.js'
import { mapGetters, mapMutations, mapActions, mapState } from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'DatasetPage',
  components: {
    'component-frame': ComponentFrame,
  },
  data: function () {
    return {
      current_dataset: Dataset
    }
  },
  computed: {
    ...mapGetters(['getFilteredDatasetList']),
    datasets: function () {
      const datasets = this.getFilteredDatasetList
      if (datasets) {
        if (datasets.length > 0) {
          return datasets
        } else {
          return false
        }
      } else {
        return false
      }
    }
  },
  methods: {

  }
}
</script>

<style lang='scss'>
#dataset-page {
  display: flex;
  align-content: flex-start;
  flex-wrap: wrap;
  height: calc(#{$app-window-height} - #{$footer-height} - 100px);
  #components {
    display: flex;
    align-content: flex-start;
    flex-wrap: wrap;
    width: calc(12*#{$component-block-width});
    #container {
      width: 100%;
      height: 100%;
      display: flex;
    }
    #left-colmn {
      width: 40%;
      height: 100%;
      .dataset-item {
        display: flex;
        width: 100%;
        height: 25px;
        border-bottom: solid 1px #eee;
        span {
          text-align: center;
          width: 33%;
          height: 100%;
        }
      }
    }
    #right-colmn {
      width: 60%;
      height: 100%;
    }
  }  
}
</style>
