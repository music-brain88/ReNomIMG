<template>
  <div id="model-sample">
    <div class="row">
      <div class="col-md-12 col-sm-12 clear-padding">
        <div class="title">
          <div class="title-text">
            Prediction Sample
          </div>
          <div>
            <span @click='prevPage' v-bind:class='{inactive: hasPrevPage}'><img class="left_arrow" :src="left_arrow"></span>
            <span @click='nextPage' v-bind:class='{inactive: hasNextPage}'><img class="right_arrow" :src="right_arrow"></span>
          </div>
        </div>
        <div class="content">
           
          <sample-image v-for="(item, index) in getValidationResult" 
            :key="item.path"
            :image_idx="index + topImageIndex"
            :image_path="item.path"
            :image_width="item.width"
            :image_height="item.height"
            :bboxes="item.predicted_bboxes">
          </sample-image>
        </div>
      </div>
    </div>
  </div>
</template>

<script>

import { mapState, mapGetters } from 'vuex'
import SampleImage from './model_sample_parts/sample_image.vue'

export default {
  name: 'ModelSample',
  components: {
    'sample-image': SampleImage
  },
  data: function () {
    return {
      left_arrow: require('../../../../static/img/yajileft.png'),
      right_arrow: require('../../../../static/img/yajiright.png')
    }
  },
  computed: {
    ...mapState(['datasets']),
    ...mapGetters(['currentDataset']),

    topImageIndex: function () {
      const dataset = this.currentDataset
      if (!dataset) {
        return []
      }
      const r = dataset.pages[this.currentPage]
      return r[0]
    },

    getValidationResult: function () {
      const dataset = this.currentDataset
      if (!dataset) {
        return []
      }
      const [pagefrom, pageto] = dataset.pages[this.currentPage]
      let result = this.$store.getters.getLastValidationResults
      if (result) {
        if (dataset.pages.length <= this.currentPage) {
          this.$store.commit('setValidationPage', {
            'page': 0
          })
        }
        return result.slice(pagefrom, pageto)
      }
    },

    currentPage: function () {
      return this.$store.state.validation_page
    },
    hasPrevPage: function () {
      if (this.currentPage > 0) { return true } else { return false }
    },
    hasNextPage: function () {
      const dataset = this.currentDataset
      if (!dataset) {
        return false
      }
      if (dataset.pages.length - 1 > this.currentPage) { return true } else { return false }
    }
  },
  methods: {
    nextPage: function () {
      if (!this.hasNextPage) { return }
      this.$store.commit('setValidationPage', {
        'page': this.currentPage + 1
      })
    },
    prevPage: function () {
      if (!this.hasPrevPage) { return }
      this.$store.commit('setValidationPage', {
        'page': this.currentPage - 1
      })
    }
  }
}
</script>

<style lang="scss" scoped>
#model-sample {
  margin: 0;
  margin-top: $component-margin-top;
  // margin-left: $content-parts-margin;

  .title {
    display: flex;
    height: $content-top-header-hight;
    line-height: $content-top-header-hight;
    font-family: $content-top-header-font-family;
    font-size: $content-top-header-font-size;
    justify-content: space-between;
    background-color: $header-color;
    color:$font-color;
    .title-text{
      margin-left: $content-top-heder-horizonral-margin;
    }
    div {
      align-self: center;
      padding-right: 10px;
      span {
        padding-left: 5px;
        font-size: 1.4rem;
      }
      span:hover:not(.inactive) {
        color: #004cc9;
        cursor: not-allowed;
      }
    }
    .inactive {
      color: #929293;
      cursor: pointer;
    }
  }

  .content {
    margin-top: $content-top-margin;
    width: 100%;
    height:$content-prediction-height;
    border: 1px solid $content-border-color;
    padding: $content-top-padding $content-horizontal-padding $content-bottom-padding;

    display: flex;
    display: -webkit-flex;

    flex-flow: row wrap;

    background-color: #fff;
  }

}
</style>
