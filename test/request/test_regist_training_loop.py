# -*- coding: utf-8 -*-
import urllib.request
import json

HEADER = {'Content-Type': 'application/json'}

# test regist
URL = 'http://0.0.0.0:8080/api/obj_detector/test_insert_trainining_loop'
REQ = {
    "training_id": 1,
    "epoch": 20,
    "data": {
        "train_loss": 1.23,
        "test_loss": 1.55,
        "test_labels": [0, 0, 1, 1],
        "predict_labels": [0, 1, 1, 0]
    }
}

data = json.dumps(REQ).encode("utf-8")

req = urllib.request.Request(URL, data=data, headers=HEADER, method="POST")
f = urllib.request.urlopen(req)
print(f.read().decode("utf-8"))
f.close()
