<template>
  <div id="image-modal">
    <div class="modal-background" @click="hideModal">
      <div id="imagediv" ref="imagediv">
        <img id='image' :src="getImage(idx_active_image_sample).path" ref="image" @load='onResize'/>
        <div class='bbox'
          v-for="(item, index) in getImage(idx_active_image_sample).predicted_bboxes" :key="index" ref='box' :style="{border:'5px solid '+getColor(item[0])}">
          <div id='tag-name' v-bind:style="{backgroundColor: getColor(item[0])}">{{ getTagName(item[0]) }}:{{round(getMaxScore(item), 1000).toFixed(2)}}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import * as utils from '@/utils'
import { mapState, mapMutations, mapGetters } from 'vuex'

export default {
  name: 'ImageModal',
  data: function () {
    return {
      image: undefined
    }
  },
  computed: {
    ...mapState([
      'idx_active_image_sample'
    ]),
    ...mapGetters([
      'getTagName'
    ])
  },
  mounted: function () {
    window.addEventListener('keyup', this.onKeyup)
    window.addEventListener('resize', this.onResize)
    this.$nextTick(function () {
      this.onResize()
    })
  },

  beforeDestroy: function () {
    window.removeEventListener('keyup', this.onKeyup)
    window.removeEventListener('resize', this.onResize)
  },

  methods: {
    ...mapMutations([
      'setShowModalImageSample'
    ]),
    getImage: function (idx) {
      let result = this.$store.getters.getLastValidationResults
      return result[idx]
    },

    getColor: function (index) {
      let color_list = [
        '#E7009A',
        '#0A20C4',
        '#3E9AAF',
        '#FFCC33',
        '#EF8200',
        '#9F14C1',
        '#582396',
        '#8BAA1A',
        '#13894B',
        '#E94C33'
      ]
      return color_list[index % 4]
    },
    hideModal: function () {
      this.setShowModalImageSample({modal: false})
    },
    keyLeft: function () {
      if (this.idx_active_image_sample > 0) {
        this.setShowModalImageSample(
          {modal: true, img_idx: this.idx_active_image_sample - 1})
      }
    },
    keyRight: function () {
      if (this.idx_active_image_sample >= 0) {
        this.setShowModalImageSample(
          {modal: true, img_idx: this.idx_active_image_sample + 1})
      }
    },
    onKeyup: function (e) {
      if (e.keyCode === 37) {
        this.keyLeft()
      } else if (e.keyCode === 39) {
        this.keyRight()
      }
    },

    onResize: function () {
      const imgdiv = this.$refs['imagediv']
      const img = this.$refs['image']

      const [orgHeight, orgWidth] = [img.naturalHeight, img.naturalWidth]
      const rc = imgdiv.parentNode.getBoundingClientRect()
      rc.height = rc.height - 153 // "153" is footer height defined as $footer-height.

      const IMG_RATIO = 0.8
      let imgHeight, imgWidth

      imgHeight = rc.height * IMG_RATIO
      let ratio = (imgHeight / orgHeight)
      imgWidth = orgWidth * ratio

      if (imgWidth > (rc.width * IMG_RATIO)) {
        imgWidth = rc.width * IMG_RATIO
        let ratio = (imgWidth / orgWidth)
        imgHeight = orgHeight * ratio
      }

      const vmargin = (rc.height - imgHeight) / 2
      const hmargin = (rc.width - imgWidth) / 2

      imgdiv.style.top = (rc.top + vmargin) + 'px'
      imgdiv.style.left = (rc.left + hmargin) + 'px'
      imgdiv.style.height = imgHeight + 'px'
      imgdiv.style.width = imgWidth + 'px'

      const bboxes = this.getImage(this.idx_active_image_sample).predicted_bboxes
      for (let i = 0; i < this.$refs['box'].length; i++) {
        let bbox = bboxes[i]
        let div = this.$refs['box'][i]
        div.style.top = bbox[2] + '%'
        div.style.left = bbox[1] + '%'
        div.style.width = bbox[3] + '%'
        div.style.height = bbox[4] + '%'
      }
    },
    getMaxScore: function (item) {
      let max = 0
      for (let i = 1; i < item.length; i++) {
        console.log('log', item[i])
        max = max < item[i] ? item[i] : max
      }
      return max
    },
    round: function (v, round_off) {
      return utils.round(v, round_off)
    }

  }
}
</script>

<style lang="scss" scoped>
#image-modal {
  $modal-color: rgba(0,0,0,0.7);
  $modal-content-bg-color: #fefefe;
  $image-border-width: 16px;

  position: fixed;
  width: 100%;
  height: calc(100vh - #{$application-header-hight});
  top: $application-header-hight;
  left: 0;
  z-index: 3;
  .modal-background {
    position: fixed;
    width: 100%;
    height: 100%;
    background-color: $modal-color;

    #imagediv {
      position: fixed;
      border: solid $image-border-width $modal-content-bg-color;
    }

    #image {
      width: 100%;
      height: 100%;
    }

    .bbox {
      position: absolute;
      box-sizing:border-box;
      #tag-name {
        color: white;
        width: -webkit-fit-content;
        width: -moz-fit-content;
        margin-top: -2px;
        margin-left: -2px;
        padding-left: 4px;
        padding-right: 4px;
        font-size: 1.4rem;
      }
    }
  }
}
</style>
