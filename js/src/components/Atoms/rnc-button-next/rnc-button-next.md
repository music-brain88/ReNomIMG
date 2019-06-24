# RncButtonNext

A Vue implementation of a ReNom Components.


## Attributes

- kind: Choose "right" or "left".

## UNIT TEST of: RncButtonNext

Render:
- Component it renders

Props default:
- Expected existence of 『kind』
- props, for『kind』expecting default as: 『"right"』

- Test of computed property. Test existence of 『kindClass』
Setting props 『{"kind":"RIGHT"}』, expecting 『"right"』

- Test of computed property. Test existence of 『kindClass』
Setting props 『{"kind":"LeFT"}』, expecting 『"left"』

- Test of error when props 『kind』 is set to　『"TEXT PLACE HOLDER"』
