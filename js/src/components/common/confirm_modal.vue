<template>
  <transition name="modal">
    <div class="modal-mask">
      <div class="modal-wrapper">
        <div class="modal-container">
          <slot name="contents">
            <span v-html="$store.state.modal_msg"/>
          </slot>
          <div class="modal-footer">
            <slot name="footer">
              <slot name="okbutton">
                <input
                  type="button"
                  class="modal-default-button"
                  value="OK"
                  @click="onClick">
              </slot>
              <slot name="cancelbutton">
                <input
                  type="button"
                  class="modal-default-button cancel"
                  value="Cancel"
                  @click="hide">
              </slot>
            </slot>
          </div>
        </div>
      </div>
    </div>
  </transition>
</template>

<script>

export default {
  name: 'ConfirmModal',
  methods: {
    hide: function () {
      this.$store.commit('hideConfirm')
    },
    onClick: function () {
      if (this.$store.state.confirm_modal_callback_function) {
        this.$store.state.confirm_modal_callback_function()
      }
      this.hide()
    }
  }
}
</script>

<style lang='scss'>
.modal-mask {
  position: fixed;
  z-index: 9998;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: table;
  transition: opacity 0.3s ease;
}

.modal-wrapper {
  display: table-cell;
  vertical-align: middle;
}

.modal-container {
  width: 500px;
  margin: 0px auto;
  padding: 20px 30px;
  background-color: #fff;
  border-radius: 0;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.33);
  transition: all 0.3s ease;
  font-family: Helvetica, Arial, sans-serif;
}

.modal-default-button {
  border-radius:0;
}

/*
 * The following styles are auto-applied to elements with
 * transition="modal" when their visibility is toggled
 * by Vue.js.
 *
 * You can easily play with the modal transition by editing
 * these styles.
 */

.modal-enter {
  opacity: 0;
}

.modal-leave-active {
  opacity: 0;
}

.modal-enter .modal-container,
.modal-leave-active .modal-container {
  -webkit-transform: scale(1.1);
  transform: scale(1.1);
}

.modal-footer {
  margin: 20px 10px 0px 10px;
  text-align: right;
}
</style>
