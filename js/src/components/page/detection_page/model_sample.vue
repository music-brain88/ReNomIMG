<template>
  <div id="model-sample">
    <div class="title">
      Prediction Sample
      <div>
        <span @click='prevPage' v-bind:class='{inactive: !hasPrevPage}'><i class="fa fa-arrow-left" aria-hidden="true"></i></span>
        <span @click='nextPage' v-bind:class='{inactive: !hasNextPage}'><i class="fa fa-arrow-right" aria-hidden="true"></i></span>
      </div>
    </div>
    <div class="content">
      <sample-image
        v-for="(item, index) in getValidationResult"
        :key="index"
        :image_path="item.path"
        :bboxes="item.predicted_bboxes">
      </sample-image>
    </div>
  </div>
</template>

<script>

import SampleImage from './model_sample_parts/sample_image.vue'

export default {
  name: 'ModelSample',
  components: {
    'sample-image': SampleImage
  },
  data: function () {
    return {
      maxImageNum: 12,
      maxPageLength: 0
    }
  },
  computed: {
    getValidationResult: function () {
      let result = this.$store.getters.getLastValidationResults
      if (result) {
        let current = this.currentPage * this.maxImageNum
        let last = Math.min((this.currentPage + 1) * this.maxImageNum, result.length)

        this.maxPageLength = Math.floor(result.length / this.maxImageNum)
        if (this.maxPageLength <= this.currentPage) {
          this.$store.commit('setValidationPage', {
            'page': 0
          })
        }
        return result.slice(current, last)
      }
    },
    currentPage: function () {
      return this.$store.state.validation_page
    },
    hasPrevPage: function () {
      if (this.currentPage > 0) { return true } else { return false }
    },
    hasNextPage: function () {
      if (this.maxPageLength - 1 > this.currentPage) { return true } else { return false }
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
  $component-margin-top: 32px;

  $border-width: 2px;
  $border-color: #006699;

  $title-height: 44px;
  $title-font-size: 15pt;
  $font-weight-medium: 500;

  margin: 0;
  margin-top: $component-margin-top;
  border-top: $border-width solid $border-color;

  .title {
    display: flex;
    line-height: $title-height;
    font-size: $title-font-size;
    font-weight: $font-weight-medium;
    justify-content: space-between;
    div {
      align-self: center;
      padding-right: 10px;
      span {
        padding-left: 5px;
        font-size: 1.4rem;
      }
      span:hover:not(.inactive) {
        color: #004cc9;
      }
    }
    .inactive {
      color: #929293;
    }
  }

  .content {
    width: 100%;
    min-height: calc(170px * 3);

    display: flex;
    display: -webkit-flex;

    flex-flow: row wrap;

    background-color: #fff;
  }
}
</style>

