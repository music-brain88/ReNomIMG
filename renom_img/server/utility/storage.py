# coding: utf-8

import datetime
import os
import sys
import sqlite3
import json
import inspect
import _pickle as pickle
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

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.flush()
        self.session.commit()
        self.session.close()


class Storage:

    def __init__(self):
        Base.metadata.create_all(bind=engine)
        self.init_table_info()

    def is_Task_exists(self):
        with SessionContext() as session:
            result = session.query(Task).all()
            return result

    def init_table_info(self):
        task = self.is_Task_exists()

        if len(task) < TOTAL_TASK:
            with SessionContext() as session:
                session.add(Task(id=None, name='Detection'))
                session.add(Task(id=None, name='Classification'))
                session.add(Task(id=None, name='Segmentation'))

    def register_model(self, task_id,
                       dataset_id, algorithm_id, hyper_parameters):
        with SessionContext() as session:
            new_model = Model(
                task_id=task_id, dataset_id=dataset_id,
                algorithm_id=algorithm_id, hyper_parameters=pickle_dump(hyper_parameters)
            )
            session.add(new_model)
            session.commit()
            return new_model.id

    def fetch_models_of_task(self, task_id):
        with SessionContext() as session:
            result = session.query(Model).filter(Model.task_id == task_id)
            dict_result = self.remove_instance_state_key(result)
            return dict_result

    def fetch_models(self):
        with SessionContext() as session:
            result = session.query(Model).all()
            dict_result = self.remove_instance_state_key(result)
            return dict_result

    def fetch_model(self, id):
        with SessionContext() as session:
            result = session.query(Model).filter(Model.id == id)
            dict_result = self.remove_instance_state_key(result)
            assert dict_result, "No model registered as the model id {}".format(id)
            return dict_result[0]

    def fetch_datasets(self):
        with SessionContext() as session:
            result = session.query(Dataset).all()
            dict_result = self.remove_instance_state_key(result)
            return dict_result

    def fetch_datasets_of_task(self, id):
        with SessionContext() as session:
            result = session.query(Dataset).filter(Dataset.task_id == id)
            dict_result = self.remove_instance_state_key(result)
            return dict_result

    def update_model(self,
                     id, state=None, running_state=None, total_epoch=None, nth_epoch=None,
                     total_batch=None, nth_batch=None, last_batch_loss=None,
                     train_loss_list=None, valid_loss_list=None, best_epoch_valid_result=None
                     ):
        with SessionContext() as session:
            model = session.query(Model).filter(Model.id == id).first()
            if model:
                if state is not None:
                    model.state = state
                if running_state is not None:
                    model.running_state = running_state
                if total_epoch is not None:
                    model.total_epoch = total_epoch
                if nth_epoch is not None:
                    model.nth_epoch = nth_epoch
                if total_batch is not None:
                    model.total_batch = total_batch
                if nth_batch is not None:
                    model.nth_batch = nth_batch
                if last_batch_loss is not None:
                    model.last_batch_loss = last_batch_loss
                if train_loss_list is not None:
                    model.train_loss_list = pickle_dump(train_loss_list)
                if valid_loss_list is not None:
                    model.valid_loss_list = pickle_dump(valid_loss_list)
                if best_epoch_valid_result is not None:
                    model.best_epoch_valid_result = pickle_dump(best_epoch_valid_result)
            session.commit()

    def fetch_dataset(self, id):
        with SessionContext() as session:
            result = session.query(Dataset).filter(Dataset.id == id)
            dict_result = self.remove_instance_state_key(result)
            assert dict_result
            return dict_result[0]

    def register_dataset(self, task_id, name, description, ratio,
                         train_data, valid_data, class_map, class_tag_list, test_dataset_id):
        with SessionContext() as session:
            new_dataset = Dataset(
                task_id=task_id, name=name, description=description, ratio=ratio,
                train_data=pickle_dump(train_data), valid_data=pickle_dump(valid_data),
                class_map=pickle_dump(class_map), class_tag_list=pickle_dump(class_tag_list),
                test_dataset_id=test_dataset_id
            )
            a = session.add(new_dataset)
            print(a)
            session.commit()
            return new_dataset.id

    def register_test_dataset(self, task_id, name, description, data):
        with SessionContext() as session:
            new_test_dataset = TestDataset(
                task_id=task_id, name=name, description=description, data=pickle_dump(data)
            )
            session.add(new_test_dataset)
            session.commit()
            return new_test_dataset.id

    def fetch_test_datasets(self):
        with SessionContext() as session:
            result = session.query(TestDataset).all()
            dict_result = self.remove_instance_state_key(result)
            return dict_result

    def fetch_test_dataset(self, id):
        with SessionContext() as session:
            result = session.query(TestDataset).filter(TestDataset.id == id)
            dict_result = self.remove_instance_state_key(result)
            assert dict_result
            return dict_result[0]

    def fetch_test_datasets_of_task(self, id):
        with SessionContext() as session:
            result = session.query(TestDataset).filter(TestDataset.task_id == id)
            dict_result = self.remove_instance_state_key(result)
            return dict_result

    def fetch_task(self):
        with SessionContext() as session:
            result = session.query(Task).all()
            dict_result = self.remove_instance_state_key(result)
            return dict_result

    def remove_instance_state_key(self, result):
        dict_result = list()
        for res in result:
            res_dict = {}
            for key, value in res.__dict__.items():
                if not key == '_sa_instance_state':
                    if isinstance(value, bytes):
                        res_dict[key] = pickle_load(value)
                    else:
                        res_dict[key] = value
            dict_result.append(res_dict)
        return dict_result


global storage
storage = Storage()
# if not storage.is_poject_exists():
#     storage.register_project('objdetection', 'comment')
