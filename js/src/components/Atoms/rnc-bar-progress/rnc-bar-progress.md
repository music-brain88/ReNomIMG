# RncBarModel

A Vue implementation of a ReNom Bar which represents the progress of training and the status of loading.


## Attributes
- barClass: 'training' or 'validating'.　Status outcome from isStopping() and isWeightDownloading() also uses 'validating' class.
- colorClass: The colors corresponds to algorithms.
- totalBatch: Only when 'training'. Default is 0.
- currentBatch:  Only when 'training'.  Default is 0.

## UNIT TEST of: RncBarProgress

     Render:
      - Component it renders

     Props default:
      - Expected existence of 『barClass』
      - props, for『barClass』expecting default as: 『"validating"』

      - Expected existence of 『colorClass』
      - props, for『colorClass』expecting default as: 『"color-0"』

      - Expected existence of 『totalBatch』
      - props, for『totalBatch』expecting default as: 『0』

      - Expected existence of 『currentBatch』
      - props, for『currentBatch』expecting default as: 『0』

     Set list of Props:
      - Setting one prop: expecting existence of 『barClass』
      - Setting props: expecting『barClass』to be: 『"TEST TEXT"』

      - Setting one prop: expecting existence of 『colorClass』
      - Setting props: expecting『colorClass』to be: 『"color-5"』

      - Setting one prop: expecting existence of 『totalBatch』
      - Setting props: expecting『totalBatch』to be: 『666』

      - Setting one prop: expecting existence of 『currentBatch』
      - Setting props: expecting『currentBatch』to be: 『666』

      - Test of computed property. Test existence of 『getWidthOfBar』
     Setting props 『{"barClass":"validating","colorClass":"color-5","totalBatch":666,"currentBatch":666}』, expecting 『{"width":"20%"}』
