# -*- coding: utf-8 -*-
import urllib.request
import json

HEADER = {'Content-Type': 'application/json'}

# test GET
URL = 'http://0.0.0.0:8080/api/obj_detector/test_get_trainining'
REQ = {
    "training_id": 1
}
data = json.dumps(REQ).encode("utf-8")

req = urllib.request.Request(URL, data=data, headers=HEADER, method="POST")
f = urllib.request.urlopen(req)
print(f.read().decode("utf-8"))
f.close()
