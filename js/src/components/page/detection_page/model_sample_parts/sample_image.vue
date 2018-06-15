<template>
  <div id='sample-image' @click.stop.prevent='onClick'>
    <img id='image' :src='image_path'></img>
    <div id='box'
      v-for="(item, index) in bboxes" :key="index" :style="{top: item[2]+'%', left: item[1]+'%', width: item[3]+'%', height: item[4]+'%', border:'2px solid '+getColor(item[0])}">
      <div id='tag-name' v-bind:style="{backgroundColor: getColor(item[0])}">{{ getTagName(item[0]) }}</div>
    </div>
  </div>
</template>

<script>
  import { mapMutations } from 'vuex'

  export default {
    name: 'SampleImage',

    props: {
      image_path: undefined,
      bboxes: undefined,
      image_idx: undefined
    },

    methods: {
      ...mapMutations([
        'setShowModalImageSample'
      ]),
      getColor: function (index) {
        let color_list = ['#f19f36', '#53b05f', '#536cff', '#f86c8e']
        return color_list[index % 4]
      },
      getTagName: function (index) {
        let label_dict = this.$store.state.class_names
        return label_dict[index]
      },
      onClick () {
        this.setShowModalImageSample({modal: true, img_idx: this.image_idx})
      }
    }
  }
</script>

<style lang="scss" scoped>

  #sample-image {
    $image-height: 160px;
    $bg-color: #ffffff;
    $border-color: #cccccc;

    position: relative;
    height: $image-height;
    margin: 4px;
    box-sizing: content-box;
    text-align: center;

    #image {
      height: $image-height;
      background-size: auto 100%;
    }
    #box {
      position: absolute;
      box-sizing:border-box;
      x-index: 10;
      opacity: 0.9;
      #tag-name {
        color: white;
        width: -webkit-fit-content;
        width: -moz-fit-content;
        padding-right: 2px;
        font-size: 0.8rem;
        opacity: 0.9;
      }
    }
  }

</style>
