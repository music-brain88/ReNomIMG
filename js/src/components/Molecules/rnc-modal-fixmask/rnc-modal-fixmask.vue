<template>
  <transition
    appear
    name="modal"
  >
    <div class="rnc-modal-fixmask">
      <div class="modal-wrapper">
        <div class="modal-container">
          <slot
            name="modal-contents"
            class="modal-contents"
          >
            default content
          </slot>

          <div class="modal-footer">
            <slot name="okbutton">
              <rnc-button
                :button-label="'OK'"
                @click-button="$emit('modal-ok')"
              />
            </slot>
            <span>&nbsp;</span>
            <slot name="cancelbutton">
              <rnc-button
                :cancel="true"
                :button-label="'Cancel'"
                @click-button="$emit('modal-cancel')"
              />
            </slot>
          </div>
        </div>
      </div>
    </div>
  </transition>
</template>

<script>
import RncButton from './../../Atoms/rnc-button/rnc-button.vue'

export default {
  name: 'RncModalFixmask',
  components: {
    'rnc-button': RncButton
  }
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
  background-color: $modal-mask-color;
  display: table;
  transition: opacity 0.3s ease;
}

.modal-wrapper {
  display: table-cell;
  vertical-align: middle;
}

.modal-container {
  position: relative;
  width: 500px;
  margin: 0px auto;
  padding: $padding-middle $padding-large;
  padding-bottom: 60px;
  background-color: $white;
  border-radius: 0;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
  transition: all 0.3s ease;
  font-family: Helvetica, Arial, sans-serif;

  .modal-footer {
    position: absolute;
    bottom: $padding-middle;
    right: $padding-large;
    display: flex;
  }
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
</style>
