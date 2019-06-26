# RncModelListItem

A Vue implementation of a ReNom Components.


## Attributes
- model: An object containing "algorithm_id", "id" and "state". To obtain it, usually you use getters that give model objects such as "getDeployedModel", "getFilteredAndGroupedModelList", etc..
- last-batch-loss: It is an item to be displayed.  
In ReNomIMG, acquired as follows.  

```Javascript
getLastBatchLoss () {
  if (this.model.getBestLoss()) {
    return this.model.getBestLoss().toFixed(2)
  } else {
    return '-'
  }
}
```

- result-of-metric1: It is an item to be displayed.  
"model.getResultOfMetric1().value" of model acquired by "getFilteredAndGroupedModelList".
- result-of-metric2: It is an item to be displayed.  
"model.getResultOfMetric1().value" of model acquired by "getFilteredAndGroupedModelList".
- selected-model-id: Acquired by "getSelectedModel" inReNomIMG.
- is-deployed-model: When this is true, it would render the style for a deployed model.
- [v-on]clicked-model-item: This event is emitted when model list item is clicked. It conveys the id of selected model.

  （The following is the item originally included. I have put in just in case.）  
  （以下は、元々入っていた項目です。念の為に入れています。）  
- isAddButton:  
- isOpenChildModelList:  
- hierarchy:  


## UNIT TEST of: RncModelListItem

Render:
- Component it renders

Props default:
- Expected existence of 『model』
- props, for『model』expecting default as: 『undefined』

- Expected existence of 『isAddButton』
- props, for『isAddButton』expecting default as: 『false』

- Expected existence of 『isOpenChildModelList』
- props, for『isOpenChildModelList』expecting default as: 『false』

- Expected existence of 『hierarchy』
- props, for『hierarchy』expecting default as: 『0』

- Expected existence of 『LastBatchLoss』
- props, for『LastBatchLoss』expecting default as: 『0』

- Expected existence of 『ResultOfMetric1』
- props, for『ResultOfMetric1』expecting default as: 『0』

- Expected existence of 『ResultOfMetric2』
- props, for『ResultOfMetric2』expecting default as: 『0』

- Expected existence of 『SelectedModelId』
- props, for『SelectedModelId』expecting default as: 『0』

- Setting one prop: expecting existence of 『model』
- Setting props: expecting『model』to be: 『{"algorithm_id":30,"id":1}』

- Setting one prop: expecting existence of 『isAddButton』
- Setting props: expecting『isAddButton』to be: 『true』

- Setting one prop: expecting existence of 『isOpenChildModelList』
- Setting props: expecting『isOpenChildModelList』to be: 『true』

- Setting one prop: expecting existence of 『hierarchy』
- Setting props: expecting『hierarchy』to be: 『6』

- Setting one prop: expecting existence of 『LastBatchLoss』
- Setting props: expecting『LastBatchLoss』to be: 『6』

- Setting one prop: expecting existence of 『ResultOfMetric1』
- Setting props: expecting『ResultOfMetric1』to be: 『6』

- Setting one prop: expecting existence of 『ResultOfMetric2』
- Setting props: expecting『ResultOfMetric2』to be: 『6』

- Setting one prop: expecting existence of 『SelectedModelId』
- Setting props: expecting『SelectedModelId』to be: 『666』
