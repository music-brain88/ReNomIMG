<template lang="html">
  <footer>
    <div class="footer-content">
      <div class="logo-and-version">
        <img
          class="logo-img"
          src="/static/img/renomlogo.png">
        <div class="version">{{ version }}</div>
      </div>
    </div>
    <div class="copyright-bar">
      <div class="copyright">©︎2019 GRID INC. ALL rights reserved.</div>
    </div>
  </footer>
</template>

<script>
export default {
  name: 'AppFooter',
  data: function () { return { 'version': '2.0.0' } }
}
</script>

<style lang="scss" scoped>
footer {
  position: relative;
  width: 100%;
  height: $footer-height;
  background: $footer-background;
  z-index: 1;
  .footer-content {
    max-width: $app-window-width;
    margin: 0 auto;
    .logo-and-version {
      display: flex;
      position: relative;
      .logo-img {
        height: 24px;
        margin: 16px;
        margin-left: 0;
      }
      .version {
        height: 20px;
        line-height: 20px;
        margin-top: 24px;
        color: white;
        font-size: 0.8rem;
      }
    }
  }
  .copyright-bar {
    position: absolute;
    bottom: 0;
    width: 100%;
    height: 20px;
    background: $footer-background-dark;
    .copyright {
      max-width: $app-window-width;
      height: 20px;
      margin: 0 auto;
      line-height: 20px;
      color: white;
      font-size: 0.6rem;
    }
  }
}
</style>

