import numpy as np
import json
import pyximport; pyximport.install()
from classification import *
import pickle

pred = [0, 4,2,5,6,1,5,2,6,7, 2, 7, 0]
true = [0, 4,2,1,6,5,7,2,0,7, 6, 3, 20]

print(precision_recall_score(pred, true))

