# coding: utf-8

import datetime
import os
import sys
import sqlite3
import json
from renom_img.server import DB_DIR
from renom_img.server.utility.DAO import Session
from renom_img.server.utility.DAO import engine
from renom_img.server.utility.table import *
TOTAL_ALOGORITHM_NUMBER = 3
TOTAL_TASK = 3

try:
    import _pickle as pickle
except:
    import cPickle as pickle


def pickle_dump(obj):
    return pickle.dumps(obj)


def pickle_load(pickled_obj):
    if sys.version_info.major == 2:
        if isinstance(pickled_obj, unicode):
            pickled_obj = pickled_obj.encode()
        return pickle.loads(pickled_obj)
    else:
        return pickle.loads(pickled_obj, encoding='ascii')



class SessionContext:

    def __init__(self):
        self.session = Session()

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc_value,traceback):
        self.session.flush()
        self.session.commit()
        self.session.close()



class Storage:

    def __init__(self):
        Base.metadata.create_all(bind=engine) 
        self.init_table_info()

    def is_Algorithm_exists(self):
        with SessionContext() as session:
            result = session.query(Algorithm).all()
            return result

    def is_Task_exists(self):
        with SessionContext() as session:
            result = session.query(Task).all()
            return result

    def init_table_info(self):
        algo = self.is_Algorithm_exists()
        task = self.is_Task_exists()
       
        if len(algo) < TOTAL_ALOGORITHM_NUMBER:
            with SessionContext() as session: 
                session.add(Algorithm(algorithm_id=None, name='Yolov1'))
                session.add(Algorithm(algorithm_id=None, name='Yolov2'))
                session.add(Algorithm(algorithm_id=None, name='Yolov3'))
                session.add(Algorithm(algorithm_id=None, name='SSD'))
       
        if len(task) < TOTAL_TASK:
            with SessionContext() as session:
                session.add(Task(task_id=None, name='Detection'))
                session.add(Task(task_id=None, name='Classification'))
                session.add(Task(task_id=None, name='Segmentation'))
    
    def register_model(self, model_id, task_id, 
                       dataset_id, algorithm_id, hypoer_parameters):
    
        print('a')

    def fetch_models(self):
        with SessionContext() as session:
            result = session.query(Model).all()
            dict_result = self.remove_instance_state_key(result)
            return result

    def fetch_model(self, model_id):
        with SessionContext() as session:
            result = session.query(Model.model_id).all()
            #dict_result = self.remove_instance_state_key(result)
            return result

    def fetch_datasets(self):
        with SessionContext() as session:
            result = session.query(Dataset).all()
            dict_result = self.remove_instance_state_key(result)
            return result
    
    def fetch_test_datasets(self):
        with SessionContext() as session:
            result = session.query(Test_Dataset).all()
            dict_result = self.remove_instance_state_key(result)
            return result
    
    def fetch_algorithms(self):
        with SessionContext() as session:
            result = session.query(Algorithm).all()
            dict_result = self.remove_instance_state_key(result)
            return dict_result
    
    def fetch_task(self): 
        with SessionContext() as session:
            result = session.query(Task).all()
            dict_result = self.remove_instance_state_key(result)
            return dict_result

    def remove_instance_state_key(self, result):
        dict_result = list()
        # for res in result:
        #     res_dict = res.__dict__
        #     res_dict.pop('_sa_instance_state')
        #
        #     print(res_dict)
        #     dict_result.append(res_dict)
        #
        # print("return")
        #
        for res in result:
            res_dict = {}
            for key,value in res.__dict__.items():
                if not key == '_sa_instance_state':
                    res_dict[key] = value

            dict_result.append(res_dict)

        return dict_result

global storage
storage = Storage()
# if not storage.is_poject_exists():
#     storage.register_project('objdetection', 'comment')
