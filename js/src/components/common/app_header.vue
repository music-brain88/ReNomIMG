<template>
  <div id="app-header">
    <div id="title">
        <span id="product-title" class="header-title">ReNomIMG</span>
        <span class="header-title"> > </span>
        <span id="task-title" class="header-title"> {{ task_title }} </span>
        <span class="header-title"> > </span>
        <span id="page-title" class="header-title"> {{ page_title }} </span>
    </div>
  </div>  
</template>

<script>
import { mapState } from 'vuex'
export default {
  name: 'AppHeader',
  computed: {
    ...mapState([
      'task_title',
      'page_title'
    ])
  },
  created: {

  },
  methods: {

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
</style>
