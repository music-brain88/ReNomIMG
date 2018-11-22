<template>
  <div id="app-header">
    <div id="menu-title">
      <div id="header-menu">
        <i class="fa fa-bars" aria-hidden="true" @click="showSlideMenu(!getShowSlideMenu)"></i>
      </div>
      <div id="title">
        <span id="product-title" class="header-title">ReNomIMG</span>
        <span class="header-title"> > </span>
        <span id="task-title" class="header-title"> {{ getCurrentTaskTitle }} </span>
        <span class="header-title"> > </span>
        <span id="page-title" class="header-title"> {{ getCurrentPageTitle }} </span>
      </div>
    </div>
    <div id="task-buttons">
      <div id="classification-button"
          @click="setCurrentTask(TASK.CLASSIFICATION)">
        Classification
      </div>  
      <div id="classification-button"
          @click="setCurrentTask(TASK.DETECTION)">
        Detection
      </div>  
      <div id="classification-button"
          @click="setCurrentTask(TASK.SEGMENTATION)">
        Segmentation
      </div>  
    </div>  
  </div>  
</template>

<script>
import { mapState, mapGetters, mapMutations } from 'vuex'
import { TASK_ID } from '@/const.js'

export default {
  name: 'AppHeader',
  computed: {
    ...mapGetters([
      'getShowSlideMenu',
      'getCurrentTaskTitle',
      'getCurrentPageTitle']),
    TASK: function () {
      return TASK_ID
    }
  },
  created: function () {

  },
  methods: {
    ...mapMutations(['showSlideMenu', 'setCurrentTask']),
  }
}
</script>

<style lang='scss'>
#app-header {
  width: 100%;
  height: $header-height;
  min-height: $header-min-height;
  background-color: $header-background-color;
  color: $header-font-color;
  display: flex;
  justify-content: space-between;

  #task-buttons {
    width: 25%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: space-around;
    div {
      margin-right: $header-title-margin-left;
      cursor: pointer;
    }
    
  }

  #menu-title {
    height: 100%;
    width: 50%;
    display: flex;
    #header-menu {
      width: 20px;
      height: 100%;
      margin-left: $header-title-margin-left;
      display: inline-flex;
      font-size: 150%;
      align-items: center;
    }

    #title {
      height: 100%;
      margin: 0 0 0 $header-title-margin-left;
  
      .header-title {
        height: $header-height;
        min-height: $header-min-height;
        display: inline-flex;
        align-items: center;
      }
  
      #product-title {
        font-family: $header-product-name-font-family;
        font-size: $header-product-name-font-size;
      }
    
      #task-title {
        font-family: $header-title-font-family;
        font-size: $header-title-font-size;
      }
  
      #page-title {
        font-family: $header-title-font-family;
        font-size: $header-title-font-size;
      }
    }
  }
}
</style>
