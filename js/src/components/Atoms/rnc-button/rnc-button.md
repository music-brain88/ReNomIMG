# RncButton

A Vue implementation of a ReNom Components button.  

※ If you specify "buttonSizeChange", you can specify the size on the parent side.  
(Refer to the left menu "size change")  


## Attributes

- disabled: When set to true, the button is disabled.
- buttonLabel: The button name.
- cancel: Please specify in case of "cancel" button.
- buttonSizeChange: When set to true, the button size is set to the size that specified on the parent element.

## UNIT TEST of: RncButton

Render:
- Component it renders

Props default:
- Expected existence of 『disabled』
- props, for『disabled』expecting default as: 『false』

- Expected existence of 『buttonLabel』
- props, for『buttonLabel』expecting default as: 『""』

- Expected existence of 『cancel』
- props, for『cancel』expecting default as: 『false』

- Setting one prop: expecting existence of 『disabled』
- Setting props: expecting『disabled』to be: 『true』

- Setting one prop: expecting existence of 『buttonLabel』
- Setting props: expecting『buttonLabel』to be: 『"TEST TEXT"』

- Setting one prop: expecting existence of 『cancel』
- Setting props: expecting『cancel』to be: 『true』

- Setting one prop: expecting existence of 『buttonSizeChange』
- Setting props: expecting『buttonSizeChange』to be: 『true』
