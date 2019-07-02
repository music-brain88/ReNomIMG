<template>
  <transition
    appear
    name="modal"
  >
    <div class="rnc-modal-fixmask">
      <div class="modal-wrapper">
        <div class="modal-container">
          <slot name="modal-contents">
            default content
          </slot>
          <div class="modal-footer">
            <slot name="footer">
              <slot name="okbutton">
                <input
                  type="button"
                  class="modal-default-button"
                  value="OK"
                  @click="$emit('modal-ok')"
                >
              </slot>
              <slot name="cancelbutton">
                <input
                  type="button"
                  class="modal-default-button cancel"
                  value="Cancel"
                  @click="$emit('modal-cancel')"
                >
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
  name: 'RncModalFixmask'
  // props: {
  //   showModal: {
  //     type: Boolean,
  //     default: false
  //   }
  // },
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

.rnc-modal-fixmask {
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

.modal-leave {
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
