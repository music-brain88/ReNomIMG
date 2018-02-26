<template>
  <div id="app" class="container">
    <app-header></app-header>

    <div id="navigation" v-bind:class="{open: isShown}">
      <div id='mask' @click='hideNavigationBar'></div>
      <navigation-bar></navigation-bar>
    </div>

    <router-view></router-view>
  </div>
</template>

<script>
import AppHeader from './common/app_header.vue'
import NavigationBar from './common/navigation_bar.vue'

export default {
  name: "App",
  components:{
    'app-header': AppHeader,
    'navigation-bar': NavigationBar
  },
  data: function () {
    return {
      showNavigationBarFlag: true
    }
  },
  computed: {
    isShown: function () {
      return this.$store.getters.getNavigationBarShowFlag
    }
  },
  created: function() {
    this.$store.commit("setPageName", {
      "page_name": this.$router.currentRoute.name,
    });
  },
  methods: {
    hideNavigationBar: function () {
      this.$store.commit('setNavigationBarShowFlag', {
        flag: false,
      })
    }
  }
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
  .container {
    $max-width: 1280px;
    $header-height: 44px;

    position: relative;
    max-width: $max-width;
    height: calc(100% - #{$header-height});
    margin: 0 auto;
    padding: 0;
    padding-top: $header-height;

    #navigation {
      position: absolute;
      visibility: hidden;
      transition: 30ms;

      #navigation-bar {
        position: fixed;
        top: $header-height;
        left: 0;
        z-index: 100;
      }
      #mask {
        position: fixed;
        top: $header-height;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 99;
        background-color: rgba(0, 0, 0, 0.4);
      }
    }
    #navigation.open{
      visibility: visible;
    }
  }
</style>
