<template>
<div id="app">
  <app-header></app-header>
  <navigation-bar></navigation-bar>
  <alert-modal v-if="$store.state.alert_modal_flag"></alert-modal>

  <div class="container">
    <router-view></router-view>
  </div>
  <app-footer></app-footer>
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
@import url('https://rsms.me/inter/inter-ui.css');
@import '@/../node_modules/bootstrap/scss/bootstrap.scss';
* {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  -o-box-sizing: border-box;
  -ms-box-sizing: border-box;
  box-sizing: border-box;

  font-family: 'Inter UI', sans-serif;
}

#app {
  // $max-width: 1280px;
  // $header-height: 44px;

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
