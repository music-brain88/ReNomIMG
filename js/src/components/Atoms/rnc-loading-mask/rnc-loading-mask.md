# RncModalTogglemask

A Vue implementation of a ReNom Components.
It would render when the page is in loading state.


## Attributes
- showLodingMask: The value type is boolean. When "true" is input RncLoadingMask appears.
- text: Please specify if you want to change the words to be displayed.

## UNIT TEST of: RncLoadingMask

Render:
- Component it renders

Props default:
- Expected existence of 『showLodingMask』
- props, for『showLodingMask』expecting default as: 『false』

- Expected existence of 『text』
- props, for『text』expecting default as: 『"Synchronizing to Server..."』

- Setting one prop: expecting existence of 『showLodingMask』
- Setting props: expecting『showLodingMask』to be: 『true』

- Setting one prop: expecting existence of 『text』
- Setting props: expecting『text』to be: 『"TEST TEXT"』
