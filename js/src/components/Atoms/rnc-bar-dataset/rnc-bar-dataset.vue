<template>
  <div class="rnc-bar-dataset">
    <div
      v-if="className"
      class="class-name"
    >
      {{ className }}
    </div>
    <div
      :style="barStyle"
      class="bar"
    >
      <section
        :style="'width:' + trainRatio + '%;'"
        :class="{'invisible': trainNum===0}"
        class="color-train"
        data-cy="bar-dataset-data-train"
      >
        <span
          v-if="!(className)"
          class="bar-content"
        >
          Train  {{ trainNum }}
        </span>
      </section>
      <section
        :style="'width:' + validRatio + '%;'"
        :class="{'invisible': validNum===0}"
        class="color-valid"
        data-cy="bar-dataset-data-valid"
      >
        <span
          v-if="!(className)"
          class="bar-content"
        >
          Valid  {{ validNum }}
        </span>
      </section>
    </div>
  </div>
</template>

<script>
export default {
  name: 'RncBarDataset',
  props: {
    trainNum: {
      type: Number,
      default: undefined
    },
    validNum: {
      type: Number,
      default: undefined
    },
    // : not using currenty
    // animated: {
    //   type: Boolean,
    //   default: undefined
    // },

    // if break down bar
    className: {
      type: String,
      default: undefined
    },
    classRatio: {
      type: Number,
      default: undefined
    }
  },
  computed: {
    trainRatio: function () {
      let train_ratio
      if (this.trainNum === undefined) {
        train_ratio = 0.5
      } else {
        train_ratio = this.trainNum / (this.trainNum + this.validNum)
      }

      return this.calc_width(train_ratio)
    },

    validRatio () {
      let valid_ratio
      if (this.validNum === undefined) {
        valid_ratio = 0.5
      } else {
        valid_ratio = this.validNum / (this.trainNum + this.validNum)
      }
      return this.calc_width(valid_ratio)
    },

    barStyle () {
      const ratio = this.classRatio
      const ret = {
        'height': 100 + '%',
        'width': 100 + '%'
      }
      if (ratio) {
        ret['height'] = 80 + '%'
        ret['width'] = ratio * 90 + '%'
      }
      return ret
    }
  },
  methods: {
    calc_width (ratio) {
      const percent = ratio * 100
      return percent
    }
  }
}
</script>

<style lang='scss' scoped>
@import './../../../../static/css/unified.scss';

.rnc-bar-dataset{
  display: flex;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  font-size: $fs-small;
  .bar {
    display: flex;
    align-items: center;
    justify-content: center;
    section {
      height: 100%;
      // TODO muraishi : consider min width
      min-width: 10%;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-wrap: wrap;
      color: white;
      .bar-content {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        width: 100%;
      }
    }
  }
  .class-name {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    width: 10%; // for class-bar, width is 90%
    height: 80%;
    margin-right: 10px;
  }
  .invisible {
    display: none !important;
    visibility: hidden;
  }

}
</style>
