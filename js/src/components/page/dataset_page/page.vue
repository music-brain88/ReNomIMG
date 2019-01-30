<template>
  <div id="dataset-page">
    <div id="components">
      <component-frame :width-weight=12 :height-weight=9>
        <template slot="header-slot">
          Dataset
        </template>
        <div id="container">
          <div id="left-colmn">
            <div class="dataset-item" style="color: #666;">
              <span>Dataset ID</span>
              <span>Name</span>
              <span>Ratio</span>
            </div>
            <div class="dataset-item" :class="{selected: current_dataset===item}"
              v-for="item in datasets" @click="current_dataset=item">
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
            <div class="title">
              Dataset Breakdown
            </div>
            <div id="dataset-description">
              <div class="col">
                <span class="item">ID : {{ id }}</span>
                <span class="item">Name : {{ name }}</span>
                <span class="item">Ratio : {{ ratio }}</span>
              </div>
              <div class="col">
                <div v-if="description" id="description-field">
                  {{ description }}
                </div>
                <div v-else>
                  No description
                </div>
              </div>
            </div>
            <div id="dataset-num-bar" class="bar" :class="{'bar-anime': current_dataset}"
              @mouseenter="isHover=true" @mouseleave="isHover=false">
              <section class="bar color-train" :style="train_style">
                <span>Train</span>
              </section>
              <section class="bar color-valid" :style="valid_style">
                <span>Valid</span>
              </section>
            </div>
            <div id="dataset-class-bars">
              <div id="item" v-for="item in class_items">
                <span>{{item[0]}}</span>
                <div class="bar" :class="{'bar-anime': current_dataset}" :style="{width: item[1] + item[2] + '%'}">
                  <section class="color-train" :style="{width: item[1]/(item[1] + item[2])*100 + '%'}"/>
                  <section class="color-valid" :style="{width: item[2]/(item[1] + item[2])*100 + '%'}"/>
                </div>
              </div>
            </div>
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
      current_dataset: undefined,
      isHover: false
    }
  },
  watch: {
    getCurrentTask: function () {
      this.reset()
    }
  },
  computed: {
    ...mapGetters([
      'getFilteredDatasetList',
      'getCurrentTask'
    ]),
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
    },
    id: function () {
      const d = this.current_dataset
      if (!d) return
      return d.id
    },
    name: function () {
      const d = this.current_dataset
      if (!d) return
      return d.name
    },
    ratio: function () {
      const d = this.current_dataset
      if (!d) return
      return d.ratio
    },
    description: function () {
      const d = this.current_dataset
      if (!d) return
      return d.description
    },
    train_num: function () {
      const dataset = this.current_dataset
      if (!dataset) return
      const t = dataset.class_info.train_img_num
      return t
    },
    valid_num: function () {
      const dataset = this.current_dataset
      if (!dataset) return
      const t = dataset.class_info.valid_img_num
      return t
    },
    class_items: function () {
      const dataset = this.current_dataset
      if (!dataset) return
      const info = dataset.class_info
      if (!info) return
      const t = info.train_ratio
      const v = info.valid_ratio
      const c = info.class_ratio
      const n = info.class_map
      return t.map((i, index) => [
        n[index],
        i * c[index] * 100,
        v[index] * c[index] * 100
      ])
    },
    train_style: function () {
      const t = this.train_num
      const v = this.valid_num
      if (!t || !v) return
      return {
        width: t / (t + v) * 100 + '%'
      }
    },
    valid_style: function () {
      const t = this.train_num
      const v = this.valid_num
      if (!t || !v) return
      return {
        width: v / (t + v) * 100 + '%'
      }
    },
  },
  methods: {
    reset: function () {
      this.current_dataset = undefined
    }
  }
}
</script>

<style lang='scss'>
#dataset-page {
  display: flex;
  align-content: flex-start;
  flex-wrap: wrap;
  height: calc(#{$app-window-height} - #{$footer-height} - 70px);
  font-size: $component-font-size-small;

  .title {
    width: 100%;
    height: 5%;
    color: gray;
    font-size: $component-font-size;
  }

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
      padding: 10px;
      color: #999;
      overflow: auto;
      .dataset-item {
        display: flex;
        align-content: center;
        width: 100%;
        height: 35px;
        border-bottom: solid 1px #eee;
        cursor: pointer;
        &:hover {
          background-color: #ddd;
        }
        &:active {
          background-color: #eee;
        }
        &.selected {
          color: $component-header-sub-color;
        }
        span {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 33%;
          height: 100%;
        }
      }
    }
    #right-colmn {
      width: 60%;
      height: 100%;
      padding: 10px;
      #dataset-description {
        width: 100%;
        height: 10%;
        display: flex;
        .col {
          display: flex;
          flex-wrap: wrap;
          width: calc(30% - 10px);
          height: 100%;
          margin-left: 10px;
          .item {
            width: 100%;
            padding-left: 10px;
          }
        }
        .col:nth-child(2) {
          width: 70%;
          div {
            width: calc(100% - 20px);
            height: 100%;
            word-wrap: break-word;
            padding-right: 20px;
            background-color: #eee;
            border: solid 1px #ddd;
            overflow: auto;
          }
        }
      }
      #dataset-num-bar {
        width: 100%;
        height: 13%;
        padding: 30px;
        display: flex;
        .bar {
          display: flex;
          align-items: center;
          justify-content: center;
          height: calc(100%);
          width: 100%;
          color: white;
        }
      }
      #dataset-class-bars{
        width: calc(100% - 40px);
        height: calc(72% - 20px);
        overflow: hidden auto;
        margin-bottom: 20px;
        margin-left: 20px;
        margin-right: 20px;
        #item {
          display: flex;
          align-items: center;
          width: 100%;
          height: 20px;
          .bar {
            height: 80%;
            display: flex;
          }
          section {
            height: 100%;
          }
          span {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            width: 15%;
            height: 100%;
            margin-right: 10px;
          }
        }
      }
      .color-train {
        background-color: #0762AD;
      }
      .color-valid {
        background-color: #EF8200;
      }
      .bar-anime {
        animation: growX 0.8s;
        animation-fill-mode: both;
        animation-iteration-count: 1;
      }
      @keyframes growX {
        0% {
          transform: translateX(-50%) scaleX(0);
        }
        100% {
          transform: translateX(0) scaleX(1);
        }
      }
    }
  }  
}
</style>
