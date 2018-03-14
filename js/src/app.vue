<template>
<div id="app">
  <app-header></app-header>
  <navigation-bar></navigation-bar>

  <div class="container">
    <router-view></router-view>
  </div>
</div>
</template>

<script>
import AppHeader from '@/components/common/app_header.vue'
import NavigationBar from '@/components/common/navigation_bar.vue'

export default {
  name: "App",
  components:{
    'app-header': AppHeader,
    'navigation-bar': NavigationBar
  },
  created: function() {
    this.$store.commit("setPageName", {
      "page_name": this.$router.currentRoute.name,
    });
  },
}
</script>

<style lang="scss">
@import url('https://rsms.me/inter/inter-ui.css');
* {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  -o-box-sizing: border-box;
  -ms-box-sizing: border-box;
  box-sizing: border-box;

  font-family: 'Inter UI', sans-serif;
}

#app {
  $max-width: 1280px;
  $header-height: 44px;

  position: relative;
  width: 100%;
  height: calc(100% - #{$header-height});
  margin: 0 auto;
  padding: 0;
  padding-top: $header-height;

  .container {
    max-width: $max-width;
  }
}
</style>
