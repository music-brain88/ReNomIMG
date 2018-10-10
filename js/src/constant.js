export const ALGORITHM_ID = {
  'YOLO': 0,
  'YOLO2': 1,
  'SSD': 2
}

export const ALGORITHM_NAME = {
  0: 'YOLO',
  1: 'YOLO2',
  2: 'SSD'
}

export const ALGORITHM_COLOR = {
  0: '#423885', // YOLO
  'YOLO': '#423885',

  1: '#136eab', // YOLO2
  'YOLO2': '#136eab',

  2: '#009453', // SSD
  'SSD': '#009453'
}

export const STATE_ID = {
  'Created': 0,
  'Running': 1,
  'Finished': 2,
  'Deleted': 3,
  'Reserved': 4
}

export const STATE_NAME = {
  0: 'Created',
  1: 'Running',
  2: 'Finished',
  3: 'Deleted',
  4: 'Reserved'
}

export const STATE_COLOR = {
  0: '#898989',
  'Created': '#898989',

  1: '#0099ce',
  'Running': '#0099ce',

  4: '#cccccc',
  'Reserved': '#cccccc'
}
