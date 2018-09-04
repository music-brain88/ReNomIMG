<template>
<div id="app">
  <div class="app-content">
    <app-header></app-header>
    <navigation-bar></navigation-bar>
    <alert-modal v-if="$store.state.alert_modal_flag"></alert-modal>

    <div class="container">
      <router-view></router-view> 
    </div>
  </div>

  <app-footer class="footer"></app-footer>
</div>
</template>

<script>
import AppHeader from '@/components/common/app_header.vue'
import NavigationBar from '@/components/common/navigation_bar.vue'
import AlertModal from '@/components/common/alert_modal.vue'
import AppFooter from '@/components/common/app_footer.vue'

export default {
  name: 'App',
  components: {
    'app-header': AppHeader,
    'navigation-bar': NavigationBar,
    'alert-modal': AlertModal,
    'app-footer': AppFooter
  },
  created: function () {
    this.$store.commit('setPageName', {
      'page_name': this.$router.currentRoute.name
    })
  }
}
</script>

<style lang="scss">
@font-face{
  font-family: $header-product-name-font-family;
  src: url("../static/fonts/OpenSans-Regular.ttf");
}

@font-face{
  font-family: $content-inner-header-font-family;
  src: url("../static/fonts/OpenSans-Light.ttf");
}


* {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  -o-box-sizing: border-box;
  -ms-box-sizing: border-box;
  box-sizing: border-box;

  font-family: $content-top-header-font-family;
}
#app {
  position: relative;
  width: 100%;
//  height: calc(100% - #{$application-header-hight});
  padding: 0;
  padding-top: $application-header-hight;

  .app-content{
    min-height: calc(100vh - ( #{$footer-height} + #{$application-header-hight} + #{$content-footer-margin-top}));
  }

  .container {
    max-width: $max-width;
  }
  .footer{
    height:$footer-height;
  }
   
}
</style>
