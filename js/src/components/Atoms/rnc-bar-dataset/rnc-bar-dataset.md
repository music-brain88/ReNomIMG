# RncBarModel

A Vue implementation of a ReNom Bar which indicates the split ratio of train and validation data.


## Attributes
- trainNum: The amount of data splitted for training.
- validNum: The amount of data splitted for validation.
- className: The name of class represented by the bar.
- classRatio: The ratio of the class data to the whole dataset.

## UNIT TEST of: RncBarDataset

       Render:
        - Component it renders

       Props default:
        - Expected existence of 『trainNum』
        - props, for『trainNum』expecting default as: 『undefined』

        - Expected existence of 『validNum』
        - props, for『validNum』expecting default as: 『undefined』

        - Expected existence of 『animated』
        - props, for『animated』expecting default as: 『undefined』

        - Expected existence of 『className』
        - props, for『className』expecting default as: 『undefined』

        - Expected existence of 『classRatio』
        - props, for『classRatio』expecting default as: 『undefined』

       Set list of Props:
        - Setting one prop: expecting existence of 『trainNum』
        - Setting props: expecting『trainNum』to be: 『666』

        - Setting one prop: expecting existence of 『validNum』
        - Setting props: expecting『validNum』to be: 『666』

        - Setting one prop: expecting existence of 『animated』
        - Setting props: expecting『animated』to be: 『true』

        - Setting one prop: expecting existence of 『className』
        - Setting props: expecting『className』to be: 『"TEXT TEST"』

        - Setting one prop: expecting existence of 『classRatio』
        - Setting props: expecting『classRatio』to be: 『666』

        - Test of computed property. Test existence of 『trainRatio』
       Setting props 『{"trainNum":666,"validNum":666}』, expecting 『50』

        - Test of computed property. Test existence of 『barStyle』
       Setting props 『{"classRatio":666}』, expecting 『{"height":"80%","width":"59940%"}』
