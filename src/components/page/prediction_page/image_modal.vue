<template>
  <div id="image-modal">
    <div class="modal-background" @click="hideModal"></div>
    <div class="modal-content" @keyup.37="keyLeft" @keyup.39="keyRight">
      <img id='image' :src="image"></img>
      <div id='box'
        v-for="(item, index) in bboxes" :style="{top: item[2]+'%', left: item[1]+'%', width: item[3]+'%', height: item[4]+'%', border:'2px solid '+getColor(item[0])}">
        <div id='tag-name' v-bind:style="{backgroundColor: getColor(item[0])}">{{ getTagName(item[0]) }}</div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "ImageModal",
  data: function() {
    return {
      image: undefined,
    }
  },
  computed: {
    getPredictResults: function() {
      return this.$store.getters.getPredictResults;
    },
    imageIndex: function() {
      return this.$store.state.image_index_on_modal;
    },
    imgPath: function() {
      return this.getPredictResults[this.imageIndex].path;
    },
    bboxes: function() {
      return this.getPredictResults[this.imageIndex].predicted_bboxes;
    },
    currentPage: function() {
      return this.$store.state.predict_page;
    },
    pageMax: function() {
      return this.$store.getters.getPageMax;
    }
  },
  created: function() {
    window.onkeyup = this.keyupFunc;
    this.loadImg();
  },
  updated: function() {
    this.loadImg();
  },
  methods: {
    loadImg: function() {
      var self = this
      let path = this.imgPath
      if(path){
        self.image = path
      }
    },
    getColor: function (index) {
      let color_list = ["#f19f36", "#53b05f", "#536cff", "#f86c8e"]
      return color_list[index%4]
    },
    getTagName: function (index) {
      let label_dict = this.$store.getters.getLabelDict;
      return label_dict[index]
    },
    hideModal: function() {
      this.$store.commit("setImageModalShowFlag", {
        "flag": false,
      });
    },
    keyLeft: function() {
      if(this.imageIndex > 0) {
        this.$store.commit("setImageIndexOnModal", {
          "index": this.imageIndex - 1,
        });
      }else if(this.currentPage > 0) {
        this.$store.commit("setPredictPage", {
          "page": this.currentPage - 1,
        });
        this.$store.commit("setImageIndexOnModal", {
          "index": this.getPredictResults.length - 1,
        });
      }
    },
    keyRight: function() {
      if(this.imageIndex < this.getPredictResults.length - 1) {
        this.$store.commit("setImageIndexOnModal", {
          "index": this.imageIndex + 1,
        });
      }else if(this.currentPage < this.pageMax) {
        this.$store.commit("setPredictPage", {
          "page": this.currentPage + 1,
        });
        this.$store.commit("setImageIndexOnModal", {
          "index": 0,
        });
      }
    },
    keyupFunc: function(e) {
      if(e.keyCode == 37) {
        this.keyLeft();
      }else if(e.keyCode == 39){
        this.keyRight();
      }
    }
  }
}
</script>

<style lang="scss" scoped>
#image-modal {
  $header-height: 35px;

  $modal-color: #000000;
  $modal-opacity: 0.7;

  $modal-content-width: 50%;
  $modal-content-height: 60%;
  $modal-content-bg-color: #fefefe;
  $modal-content-padding: 32px;

  $modal-title-font-size: 32px;
  $modal-sub-title-font-size: 24px;

  $content-margin: 8px;
  $content-label-width: 120px;
  $content-font-size: 16px;

  $image-border-width: 16px;

  position: fixed;
  width: 100%;
  height: calc(100vh - #{$header-height});
  top: $header-height;
  left: 0;

  .modal-background {
    width: 100%;
    height: 100%;
    background-color: $modal-color;
    opacity: $modal-opacity;
  }

  .modal-content {
    position: absolute;
    top: 50%;
    left: 50%;
    -webkit-transform: translateY(-50%) translateX(-50%);
    transform: translateY(-50%) translateX(-50%);

    height: $modal-content-height;
    padding: 0;
    text-align: center;
    background-color: #ffffff;
    border: $image-border-width solid #ffffff;
    opacity: 1;

    #image {
      height: 100%;
      background-size: auto 100%;
    }
    #box {
      position: absolute;
      box-sizing:border-box;
      x-index: 10;
      #tag-name {
        color: white;
        top: -4px;
        margin-top: -4px;
        width: -webkit-fit-content;
        width: -moz-fit-content;
        padding-right: 2px;
        font-size: 0.8rem;
      }
    }
  }
}
</style>

