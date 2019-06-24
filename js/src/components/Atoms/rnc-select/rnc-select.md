# RncSelect

A Vue implementation of a ReNom Components select box.



## Attributes
- value: bound item by v-model.
- disabled: When set to false, the drop-down list should be disabled.
- optionInfo: An array which contains select options. This is often created by the getter : "getFilteredDatasetList" or "getAlgorithmParamList" or etc..
- getId: When set to true, rnc-select returns "item.id".
- [event] change-select: The components emit event like this: this.$emit("change-select", e.target.value). The second argument represents the value of selected item. ( When get-id=true, the value is index of an item. When get-id=false, it is an item's name)


## UNIT TEST of: RncModalTogglemask

Render:
- Component it renders

Props default:
- Expected existence of 『showModal』
- props, for『showModal』expecting default as: 『false』

- Setting one prop: expecting existence of 『showModal』
- Setting props: expecting『showModal』to be: 『true』

- Test of emitter by click on 『#modal-mask』. Expecting to emit『show-modal』 with the data: 『false』
