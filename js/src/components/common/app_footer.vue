<template lang="html">
<footer>
  <div class="row">
    <div class="col-md-12 footer-content">
      <div class="row">
        <div class="col-md-3">
          <div class="media-wrapper">
            <img :src="url">
            <div class="content-margin-top">
                ReNom_IMG {{version}}
            </div>
          </div>
        </div>
        <div class="col-md-9">
          <!--none -->
        </div>
      </div>
    </div>
  </div>
  <div class="row">
    <div class="col-md-12 copyright align-middle">
      <div class="copyright-text">
        <small>&copy; 2018 GRID INC. ALL rights reserved.</small>
      </div>
    </div>
  </div>
</footer>
</template>

<script>

export default {
  name: 'AppFooter',
  data () {
    return {
      version: 'ver 1.0.4',
      url: require('../../../static/img/renomlogo.png')
    }
  }
}
</script>

<style lang="scss" scoped>
@import '@/../node_modules/bootstrap/scss/bootstrap.scss';
footer {
  display: block;
  width: 100%;
  height:$footer-height;
  padding: 0;
  margin: 0;
  background: $footer-color;
  margin-top: $component-margin-top;
  z-index: 99999;
  .row {
    margin-left: 0px;
    margin-right: 0px;
  }
  .footer-content{
    padding-left: 3%;
    background: $footer-color;
    height:118px;
  }
  .copyright{
    background: $footer-copyright-color;
    margin-top: 0;
    height:35px;
    .copyright-text{
      line-height: 35px;
      padding-left: 3%;
    }
  }
}
.content-margin-top{
  margin-top: 15px;
}
.media-wrapper{
  margin-top: $content-margin;
}
.test{
  margin-top: 5%;
  font-family: $content-inner-header-font-family;
  font-size: $content-inner-header-font-size;
}
</style>
