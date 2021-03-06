<template>
  <div id="dataset-page">
    <div id="components">
      <rnc-title-frame
        :width-weight="widthWeight"
        :height-weight="heightWeight"
      >
        <template slot="header-slot">
          Dataset
          <div id="setting-button">
            <rnc-button
              :button-label="'> Setting of Dataset'"
              :button-size-change="true"
              @click-button="showModal({add_both: true})"
            />
          </div>
        </template>
        <template slot="content-slot">
          <div id="container">
            <div id="left-colmn">
              <div
                class="dataset-item-title"
                style="color: #666;"
              >
                <span>
                  Name
                </span>
                <span>
                  Ratio
                </span>
                <span>
                  Train
                </span>
                <span>
                  Validation
                </span>
              </div>
              <div class="dataset-items">
                <div
                  v-for="(item, key) in datasets"
                  ref="key"
                  :class="{selected: selected_dataset===item}"
                  :key="key"
                  class="dataset-item"
                  @click="clickedDatasetsItem(item)"
                >
                  <span>
                    {{ item.name }}
                  </span>
                  <span>
                    {{ item.ratio }}
                  </span>
                  <span>
                    {{ item.class_info.train_img_num }}
                  </span>
                  <span>
                    {{ item.class_info.valid_img_num }}
                  </span>
                </div>
                <div
                  v-if="!datasets"
                  class="dataset-item"
                >
                  <span />
                  <span>
                    No dataset
                  </span>
                  <span />
                </div>
              </div>

            </div>
            <div id="right-colmn">
              <div class="title">
                Dataset Breakdown
              </div>
              <div id="dataset-description">
                <div class="col">
                  <span class="item">
                    <rnc-key-value
                      :value-text="id"
                      key-text="ID :"
                    />
                  </span>
                  <span class="item">
                    <rnc-key-value
                      :value-text="name"
                      key-text="Name :"
                    />
                  </span>
                  <span class="item">
                    <rnc-key-value
                      :value-text="ratio"
                      key-text="Ratio :"
                    />
                  </span>
                </div>
                <div class="col">
                  <rnc-input
                    v-model="description"
                    :is-textarea="true"
                    :disabled="true"
                    :place-holder="'No description'"
                    :rows="5"
                  />
                </div>
              </div>

              <div
                v-if="selected_dataset && bar_move"
                id="dataset-num-bar"
                :class="{'grow-x-anime': bar_move}"
              >
                <div class="bar">
                  <rnc-bar-dataset
                    :train-num="train_num"
                    :valid-num="valid_num"
                  />
                </div>
              </div>

              <div class="items">
                <div
                  v-for="(item, key) in class_items"
                  id="item"
                  :key="key"
                >
                  <div
                    v-if="bar_move"
                    id="dataset-class-bars"
                    :class="{'grow-x-anime': bar_move}"
                  >
                    <div class="bar">
                      <rnc-bar-dataset
                        :train-num="item[1]"
                        :valid-num="item[2]"
                        :class-name="item[0]"
                        :class-ratio="item[3]"
                      />
                    </div>
                  </div>
                </div>
              </div>

            </div>
          </div>
        </template>
      </rnc-title-frame>
    </div>
  </div>
</template>

<script>
import { mapGetters, mapActions, mapMutations } from 'vuex'
import RncBarDataset from './../../Atoms/rnc-bar-dataset/rnc-bar-dataset.vue'
import RncKeyValue from './../../Atoms/rnc-key-value/rnc-key-value.vue'
import RncInput from './../../Atoms/rnc-input/rnc-input.vue'
import RncButton from './../../Atoms/rnc-button/rnc-button.vue'
import RncTitleFrame from './../../Molecules/rnc-title-frame/rnc-title-frame.vue'

export default {
  name: 'RncDatasetPanelDataset',
  components: {
    'rnc-bar-dataset': RncBarDataset,
    'rnc-key-value': RncKeyValue,
    'rnc-input': RncInput,
    'rnc-button': RncButton,
    'rnc-title-frame': RncTitleFrame
  },
  props: {
    widthWeight: {
      type: Number,
      default: 12,
    },
    heightWeight: {
      type: Number,
      default: 12,
    }
  },
  data: function () {
    return {
      clicked_dataset_id: undefined,
      bar_move: false,
      selected_id: undefined
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
    selected_dataset: function () {
      if (!this.datasets) return
      const index = this.datasets.findIndex(n => n.id === this.clicked_dataset_id)
      return this.datasets[index]
    },
    id: function () {
      const d = this.selected_dataset

      if (!d) return ''
      return d.id
    },
    name: function () {
      const d = this.selected_dataset

      if (!d) return ''
      return d.name
    },
    ratio: function () {
      const d = this.selected_dataset

      if (!d) return ''
      return d.ratio
    },
    description: function () {
      const d = this.selected_dataset

      if (!d) return ''
      return d.description
    },
    train_num: function () {
      const d = this.selected_dataset

      if (!d) return
      return d.class_info.train_img_num
    },
    valid_num: function () {
      const d = this.selected_dataset

      if (!d) return
      return d.class_info.valid_img_num
    },
    class_items: function () {
      const d = this.selected_dataset

      if (!d) return
      const info = d.class_info

      if (!info) return
      const t = info.train_ratio
      const v = info.valid_ratio
      const c = info.class_ratio
      const n = info.class_map

      if (!t) return

      const ret = t.map((i, index) => [
        n[index],
        i * c[index] * 100,
        v[index] * c[index] * 100,
        c[index]
      ])
      return ret
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
  watch: {
    getCurrentTask: function () {
      this.reset()
    },
    selected_dataset: function () {
      // 少し時間を開けて、モーションが正常稼働しない不具合を解消
      // ↓2度選択される現象が発生しているため、selected_id判定追加
      if (this.selected_id !== this.selected_dataset.id) {
        this.bar_move = false
        this.selected_id = this.selected_dataset.id
        setTimeout(this.barMoveTrue, 100)
      }
    }
  },
  mounted: function () {
    if (this.datasets) {
      this.clickedDatasetsItem(this.datasets[0])
    }
  },
  methods: {
    ...mapActions([
      'loadDatasetsOfCurrentTaskDetail'
    ]),
    ...mapMutations([
      'showModal'
    ]),

    reset: function () {
      this.selected_dataset = undefined
    },
    barMoveTrue: function () {
      this.bar_move = true
    },
    clickedDatasetsItem: function (dataset) {
      this.loadDatasetsOfCurrentTaskDetail(dataset.id)
      this.clicked_dataset_id = dataset.id
    }
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

#dataset-page {
  display: flex;
  align-content: flex-start;
  flex-wrap: wrap;
  font-size: $fs-small;

  .title {
    width: 100%;
    height: 5%;
    color: gray;
    font-size: $fs-regular;
  }

  #components {
    display: flex;
    align-content: flex-start;
    flex-wrap: wrap;
    width: calc(12*#{$component-block-width});

    #setting-button {
      height: 100%;
      width: 15%;
    }

    #container {
      width: 100%;
      height: 100%;
      display: flex;
    }
    #left-colmn {
      width: 30%;
      height: 100%;
      padding: 20px;
      color: #999;
      overflow: auto;
      .dataset-item-title {
        display: flex;
        align-content: center;
        width: 100%;
        height: 35px;
        border-bottom: $border-width-regular solid $light-gray;
        span {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 33%;
          height: 100%;
        }
      }
      .dataset-items {
        overflow: auto;
        height: calc(100% - 35px);
        width: 100%;
        .dataset-item {
          display: flex;
          align-content: center;
          width: 100%;
          height: 35px;
          border-bottom: $border-width-regular solid $light-gray;
          cursor: pointer;
          &:hover {
            background-color: $item-hover-color;
          }
          &:active {
            background-color: $ex-light-gray;
          }
          &.selected {
            color: $blue;
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
    }
    #right-colmn {
      width: 70%;
      height: 100%;
      padding: 20px;
      color: $black;
      .items {
        overflow: auto;
        height: 65%;
        width: 100%;
        .item {
          width: 100%;
          padding-left: 10px;
        }
      }
      #dataset-description {
        width: 100%;
        height: 17%;
        display: flex;
        line-height: $text-height-micro;
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
        }
      }
      #dataset-num-bar {
        width: 100%;
        height: 13%;
        padding: 21px;
        display: flex;
      }
      #dataset-class-bars{
        width: calc(100% - 40px);
        height: calc(72% - 20px);
        overflow: hidden auto;
        margin-bottom: 10px;
        margin-left: 20px;
        margin-right: 20px;
        #item {
          display: flex;
          align-items: center;
          width: 100%;
          height: 20px;
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
      .bar {
        height: 20px;
        width: 94%;
      }
    }
  }
}
</style>
