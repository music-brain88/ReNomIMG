<template>
  <div id='sample-image' @click="showImageModal">
    <img id='image' :src="image"></img>
    <div id='box'
      v-for="(item, index) in bboxes" :style="{top: item[2]+'%', left: item[1]+'%', width: item[3]+'%', height: item[4]+'%', border:'2px solid '+getColor(item[0])}">
      <div id='tag-name' v-bind:style="{backgroundColor: getColor(item[0])}">{{ getTagName(item[0]) }}</div>
    </div>
  </div>
</template>

<script>
  export default {
    name: 'SampleImage',
    props: {
      image_path: undefined,
      bboxes: undefined,
      index: undefined
    },
    data: function () {
      return {
        image: undefined
      }
    },
    created: function () {
      this.loadImg()
    },
    updated: function () {
      this.loadImg()
    },
    methods: {
      loadImg: function () {
        var self = this
        let path = this.$props.image_path
        if (path) {
          self.image = path
        }
      },
      getColor: function (index) {
        let color_list = ['#f19f36', '#53b05f', '#536cff', '#f86c8e']
        return color_list[index % 4]
      },
      getTagName: function (index) {
        if (!this.$store.getters.getSelectedModel) {
          return
        }
        let dataset_def_id = this.$store.getters.getSelectedModel.dataset_def_id
        let dataset_def = this.$store.state.dataset_defs
        let label_dict
        for (let i in Object.keys(this.$store.state.dataset_defs).length) {
          if (dataset_def[i].id === dataset_def_id) {
            label_dict = this.$store.state.dataset_defs[i].class_map
            break
          }
        }
        return label_dict[index]
      },
      showImageModal: function () {
        this.$store.commit('setImageIndexOnModal', {'index': this.index})
        this.$store.commit('setImageModalShowFlag', {'flag': true})
      }
    }
  }
</script>

<style lang='scss' scoped>

  #sample-image {
    position: relative;
    height: 175px;
    // border: 4px solid #ffffff;
    box-shadow:2px 2px 2px 1px #b6b6b6;
    margin-left: 6px;
    margin-right: 6px;
    margin-top: 10px;
    margin-bottom: 10px;

    #image {
      height: 180px;
      background-size: auto 100%;
    }
    #box {
      position: absolute;
      box-sizing:border-box;
      x-index: 10;
      #tag-name {
        color: white;
        width: -webkit-fit-content;
        width: -moz-fit-content;
        padding-right: 2px;
        font-size: 0.8rem;
      }
    }
  }

</style>
