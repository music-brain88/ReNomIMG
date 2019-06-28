<template>
  <footer>
    <div class="footer-content">
      <div class="logo-and-version">
        <img
          class="logo-img"
          src="./../../../../static/img/renomlogo.png"
        >
        <div class="version">
          {{ version }}
        </div>
      </div>
    </div>
    <div class="copyright-bar">
      <div class="copyright">
        {{ copyright }}
      </div>
    </div>
  </footer>
</template>

<script>
export default {
  name: 'AppFooter',
  props: {
    version: {
      type: String,
      default: '0.0.0 beta',
    },
    copyright: {
      type: String,
      default: '©︎GRID INC. ALL rights reserved.',
    }
  }
}
</script>

<style lang='scss' scoped>
@import './../../../../static/css/unified.scss';

footer {
  position: relative;
  width: 100%;
  height: $footer-height;
  background: $footer-background;
  z-index: 1;
  .footer-content {
    max-width: $max-width;
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
        font-size: $fs-small;
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
      max-width: $max-width;
      height: 20px;
      margin: 0 auto;
      line-height: 20px;
      color: white;
      font-size: $fs-micro;
    }
  }
}
</style>
