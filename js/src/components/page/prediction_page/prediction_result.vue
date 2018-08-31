<template>
  <div id="prediction-result">
    <div class="row">
      <div class="col-md-12">
        <div class="title">
          <div class="title-text">
            Prediction result
          </div>
          <div>
            <span><img class="left_arrow" :src="left_arrow"></span>
            <span><img class="right_arrow" :src="right_arrow"></span>
          </div>
          <!-- <pager></pager>-->
        </div>
        <div v-if="getPredictResults.length > 0" class="content">
          <sample-image
            v-for="(item, index) in getPredictResults"
            :key="index"
            :image_path="item.path"
            :bboxes="item.predicted_bboxes"
            :index="index">
          </sample-image>
        </div>
        <div v-else class="content none-image">
          None Image
          {{getPredictResults}}
        </div>
      </div>
    </div>




    <div id='loading' v-if='$store.state.predict_running_flag'>
      <div class="loading">
        <div class="animation">
          <div class="bar bar1"></div>
          <div class="bar bar2"></div>
          <div class="bar bar3"></div>
          <div class="bar bar4"></div>
          <div class="bar bar5"></div>
        </div>
      </div>
      <div class="prediction-progress">
        Predicting {{$store.state.predict_last_batch}} / {{$store.state.predict_total_batch}}, please wait...
      </div>
    </div>

    <div class="result-foot">
      <pager></pager>
    </div>
  </div>
</template>

<script>
import SampleImage from './prediction_result_parts/sample_image.vue'
import Pager from './prediction_result_parts/pager.vue'

export default {
  name: 'PredictionResult',
  components: {
    'sample-image': SampleImage,
    'pager': Pager
  },
  data: function () {
    return {
      left_arrow: require('../../../../static/img/yajileft.png'),
      right_arrow: require('../../../../static/img/yajiright.png')
    }
  },
  computed: {
    getPredictResults: function () {
      return this.$store.getters.getPredictResults
    }
  }
}
</script>

<style lang="scss" scoped>
#prediction-result {

  width: 100%;
  margin: 0;
  margin-top: $component-margin-top;

  .title {
    height: $content-top-header-hight;

    font-family: $content-top-header-font-family;
    font-size: $content-top-header-font-size;
    background-color: $header-color;
    color:$font-color;
    .title-text{
      line-height: $content-top-header-hight;
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
      }
    }
    
  }

  .content {
    margin-top: $content-top-margin;
    width: 100%;
    height:calc(#{$content-prediction-height} + #{$content-top-margin} + #{$content-top-header-hight} );
    //min-height: calc(170px * 3);
    border: 1px solid $content-border-color;
    padding: $content-top-padding $content-horizontal-padding $content-bottom-padding;

    display: flex;
    display: -webkit-flex;

    flex-flow: row wrap;

    background-color: #fff;
  }
  .none-image{
    text-align: center;
    margin-left: auto;
  }

  .page_nav {
    display: flex;
    margin-left: auto;
    margin-top: 16px;
  }
}

/* loading animation */
.loading {
  position: fixed;
  top: 0;
  left: 0;
  height: 100vh;
  width: 100vw;
  z-index: 998;
  background-color: #000000;
  opacity: 0.5;

  .animation {
    position: absolute;
    top: 50%;
    left: 50%;
    -webkit-transform: translateY(-50%) translateX(-50%);
    transform: translateY(-50%) translateX(-50%);
  }
  .animation div {
    height:30px;
    width:10px;
    background-color: #ffffff;
    display:inline-block;
    margin-right:10px;
    -webkit-animation: animation-bar 1s infinite;
    animation: animation-bar 1s infinite;
  }

  .animation .bar1 {
    -webkit-animation-delay: 0.5s;
    animation-delay: 0.5s;
  }
  .animation .bar2 {
    -webkit-animation-delay: 0.6s;
    animation-delay: 0.6s;
  }
  .animation .bar3 {
    -webkit-animation-delay: 0.7s;
    animation-delay: 0.7s;
  }
  .animation .bar4 {
    -webkit-animation-delay: 0.8s;
    animation-delay: 0.8s;
  }
  .animation .bar5 {
    -webkit-animation-delay: 0.9s;
    animation-delay: 0.9s;
  }

  @keyframes animation-bar{
    30%
    {
        transform: scaleY(2);
    }
  }
}
.prediction-progress {
  position: absolute;
  top: 60%;
  left: 50%;
  -webkit-transform: translateX(-50%);
  transform: translateX(-50%);
  z-index: 999;
  color: #fff;
  opacity: 1;
}


</style>
