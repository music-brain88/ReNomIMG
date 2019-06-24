# RncModalTogglemask

A Vue implementation of a ReNom Components.
It would render the modal with toggle mask.
When the mask is clicked during the modal is showing, the modal would close.



## Attributes

- [prop] showModal: The boolean type prop. When it's true, the modal appears.
- [v-on] show-modal: the "show-modal" event. It would be emitted by rnc-modal-togglemask.vue when mask is clicked. After listen this, you should change the value of showModal prop.

- [slot] modal-contents: Add specific contents in the modal.

## UNIT TEST of: RncModalTogglemask

Render:
- Component it renders

Props default:
- Expected existence of 『showModal』
- props, for『showModal』expecting default as: 『false』

- Setting one prop: expecting existence of 『showModal』
- Setting props: expecting『showModal』to be: 『true』

- Test of emitter by click on 『#modal-mask』. Expecting to emit『show-modal』 with the data: 『false』
