# RncModalFixmask

A Vue implementation of a ReNom Components.
The modal which doesn't hide until the buttons would be clicked.
Listen the events emitted by this component and use the mutations such as "hideAlert()", "hideConfirm()" in the parent to remove a modal.



## Attributes

- [event] modal-ok: The component would emit this event when the OK button is clicked.

- [event] modal-cancel: The component would emit this event when the Cancel button is clicked.

- [slot] modal-contents: Any type of template can be slotted here. Usually messages are inputted.

## UNIT TEST of: RncModalFixmask

Render:
- Component it renders

- Test of emitter by click on 『[value="OK"]』. Expecting to emit『modal-ok』 with the data: 『undefined』

- Test of emitter by click on 『[value="Cancel"]』. Expecting to emit『modal-cancel』 with the data: 『undefined』
