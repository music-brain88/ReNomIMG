# RncTitleFrame

A Vue implementation of a ReNom Components.

※ Background-color is added on the Storybook side to check the style.   
※ "width-weight" and "height-weight" are responsive.   
※ When "width-weight" is set to 12, the width becomes full.   
※ If "height-weight" is set to 12, the entire height including the header will be full.   
　(45px per weight is secured as the minimum value of the vertical width)  
※ If you do not specify "height-weight", the width of "content-slot" will automatically fit.   
　(Please set the height and padding etc. on the content side inserted in)  
※ If "height-weight" and "content-slot" are not specified, only titles can be used.   
※ Please set padding of "content-slot" area by the side (Organism) of use.  



## Attributes
- width-weight: Up to 12 (full width) can be specified.
- height-weight:You can specify the height of the body.


## Slots
- If you describe a slot normally, it will be displayed in the Body.
- slot="header-slot": The slot for which 「slot="header-slot"」 is specified for the tag is displayed in the title.


## UNIT TEST of: RncTitleFrame

Render:
- Component it renders

Props default:
- Expected existence of 『widthWeight』
- props, for『widthWeight』expecting default as: 『1』

- Expected existence of 『heightWeight』
- props, for『heightWeight』expecting default as: 『1』

Set list of Props:
- Setting one prop: expecting existence of 『widthWeight』
- Setting props: expecting『widthWeight』to be: 『666』

- Setting one prop: expecting existence of 『heightWeight』
- Setting props: expecting『heightWeight』to be: 『666』

- Test of computed property. Test existence of 『styles』
Setting props 『{"widthWeight":666,"heightWeight":666}』, expecting 『{"--width-weight":666,"--height-weight":666}』
