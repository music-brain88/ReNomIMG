# RncProgressDetail

A Vue implementation of a ReNom Components.
This describes the progress and details of model trainings, and is contained in RncTrainPanelModelDetail Organism.
In addition, it contains RncButtonStop and RncBarProgress Atoms.



## Attributes
- isTitle : The tag with is-title="true" would be used only once in RncTrainPanelModelDetail. It describes the title of items in RncProgressDetail.
- colorClass : Use getter getColorClass to obtain the algorithm color of a model.
- model : A proceeding model instance of which details would be shown.
- [event] click-stop-button : Listen to a click event on the stop button.

## UNIT TEST of: RncProgressDetail

Render:
- Component it renders

Props default:
- Expected existence of 『model』
- props, for『model』expecting default as: 『undefined』

- Expected existence of 『isTitle』
- props, for『isTitle』expecting default as: 『false』

- Expected existence of 『colorClass』
- props, for『colorClass』expecting default as: 『"color-0"』

Set list of Props:
- Setting one prop: expecting existence of 『model』
- Setting props: expecting『model』to be: 『undefined』

- Setting one prop: expecting existence of 『isTitle』
- Setting props: expecting『isTitle』to be: 『true』

- Setting one prop: expecting existence of 『colorClass』
- Setting props: expecting『colorClass』to be: 『"color-5"』
