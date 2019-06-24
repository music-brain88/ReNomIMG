<template>
  <img
    class= "rnc-image"
    v-if="showImage"
    :src="img"
    :style="modifiedSize"
  >
</template>

<script>

import {isNaNhandler} from '../../../utils/errorHandling.js'

export default {
  name: 'RncImage',
  props: {
    img: {
      type: String,
      default: ''
    },
    imgWidth: {
      type: Number,
      default: 0
    },
    imgHeight: {
      type: Number,
      default: 0
    },
    maxWidth: {
      type: Number,
      default: 0
    },
    maxHeight: {
      type: Number,
      default: 0
    },
    showImage: {
      type: Boolean,
      default: true
    }
  },
  computed: {
    modifiedSize: function () {
      isNaNhandler(this.maxWidth, `'maxWidth' must be Number`);
      isNaNhandler(this.maxHeight,`'maxHeight' must be Number`);
      isNaNhandler(this.imgWidth, `'imgWidth' must be Number`);
      isNaNhandler(this.imgHeight,`'imgHeight' must be Number`);
      
      let w, h

      if (this.maxWidth === 0) {
        if (this.maxHeight === 0) {
          w = this.imgWidth
          h = this.imgHeight
        } else {
          const r = this.maxHeight / this.imgHeight
          w = this.imgWidth * r
          h = this.imgHeight * r
        }
      } else if (this.maxHeight === 0) {
        if (this.maxWidth === 0) {
          // Never reach here
        } else {
          const r = this.maxWidth / this.imgWidth
          w = this.imgWidth * r
          h = this.imgHeight * r
        }
      } else {
        const wr = this.maxWidth / this.imgWidth
        const hr = this.maxHeight / this.imgHeight
        const r = (wr < hr) ? wr : hr
        w = this.imgWidth * r
        h = this.imgHeight * r
      }
      return {
        width: 'calc(' + w + 'px' + ' - 0.4vmin)',
        height: 'calc(' + h + 'px' + ' - 0.4vmin)'
      }
    }
  }
}
</script>

<style lang="scss" scoped>
</style>
