# coding: utf-8

import datetime
import os
import sys
import sqlite3
import json
from renom_img.server import DB_DIR
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


class Storage:
    def __init__(self):
        dbname = os.path.join(DB_DIR, 'test_storage.db')
        self.db = sqlite3.connect(dbname,
                                  check_same_thread=False,
                                  detect_types=sqlite3.PARSE_DECLTYPES,
                                  isolation_level=None)
        self.db.execute('PRAGMA journal_mode = WAL')
        self.db.execute('PRAGMA foreign_keys = ON')
        self._init_db()

    def cursor(self):
        return self.db.cursor()

    def _init_db(self):
        c = self.cursor()
        c.execute("""
                CREATE TABLE IF NOT EXISTS project
                (project_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 project_name TEXT NOT NULL,
                 project_comment TEXT,
                 deploy_model_id INTEGER REFERENCES model(model_id) ON DELETE CASCADE,
                 dataset_id INTEGER REFERENCES model(model_id) ON DELETE CASCADE,
                 created TIMESTAMP NOT NULL,
                 updated TIMESTAMP NOT NULL)
            """)

        c.execute("""
                CREATE TABLE IF NOT EXISTS dataset_def
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name VARCHAR(256),
                 ratio FLOAT,
                 train_imgs CLOB,
                 valid_imgs CLOB,
                 class_map BLOB,
                 created TIMESTAMP NOT NULL,
                 updated TIMESTAMP NOT NULL);
          """)

        c.execute("""
                CREATE TABLE IF NOT EXISTS model
                (model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 project_id INTEGER NOT NULL REFERENCES project(project_id) ON DELETE CASCADE,
                 hyper_parameters BLOB,
                 dataset_def_id INTEGER NOT NULL REFERENCES dataset_def(id) ON DELETE CASCADE,
                 algorithm INTEGER NOT NULL,
                 algorithm_params BLOB,
                 state INTEGER NOT NULL DEFAULT 0,
                 train_loss_list BLOB,
                 validation_loss_list BLOB,
                 best_epoch INTEGER,
                 best_epoch_iou NUMBER,
                 best_epoch_map NUMBER,
                 best_epoch_validation_result BLOB,
                 best_epoch_weight TEXT,
                 last_epoch INTEGER DEFAULT 0,
                 last_weight TEXT,
                 last_batch INTEGER DEFAULT 0,
                 total_batch INTEGER DEFAULT 0,
                 last_train_loss NUMBER DEFAULT 0,
                 running_state INTEGER DEFAULT 3,
                 created TIMESTAMP NOT NULL,
                 updated TIMESTAMP NOT NULL)
            """)

        c.execute("""
                CREATE TABLE IF NOT EXISTS epoch
                (epoch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 model_id INTEGER NOT NULL REFERENCES model(model_id) ON DELETE CASCADE,
                 nth_epoch INTEGER NOT NULL,
                 train_loss NUMBER,
                 validation_loss NUMBER,
                 epoch_iou NUMBER,
                 epoch_map NUMBER,
                 created TIMESTAMP NOT NULL,
                 updated TIMESTAMP NOT NULL,
                 UNIQUE(model_id, nth_epoch))
          """)

        c.execute("""
                CREATE INDEX IF NOT EXISTS
                IDX_EPOCH_MODEL_ID
                ON epoch(model_id, nth_epoch)
            """)

        c.execute("""
                CREATE TABLE IF NOT EXISTS dataset
                (dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 dataset_name TEXT NOT NULL,
                 train_size NUMBER NOT NULL,
                 type INTEGER NOT NULL,
                 invalid INTEGER NOT NULL DEFAULT 0,
                 train_num INTEGER,
                 valid_num INTEGER,
                 train_class_num INTEGER,
                 valid_class_num INTEGER,
                 img_dir TEXT,
                 label_dir TEXT,
                 train_img_list BLOB,
                 valid_img_list BLOB,
                 train_label_list BLOB,
                 valid_label_list BLOB,
                 class_names BLOB,
                 created TIMESTAMP NOT NULL,
                 updated TIMESTAMP NOT NULL)
            """)

        c.execute("""
                CREATE TABLE IF NOT EXISTS epoch
                (epoch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 model_id INTEGER NOT NULL REFERENCES model(model_id) ON DELETE CASCADE,
                 nth_epoch INTEGER NOT NULL,
                 train_loss NUMBER,
                 validation_loss NUMBER,
                 epoch_iou NUMBER,
                 epoch_map NUMBER,
                 created TIMESTAMP NOT NULL,
                 updated TIMESTAMP NOT NULL,
                 UNIQUE(model_id, nth_epoch))
          """)

    def is_poject_exists(self):
        with self.db:
            project_id = 1
            c = self.cursor()
            c.execute("""
                    SELECT COUNT(*) FROM project
                """)
            for data in c:
                return data[0]

    def register_project(self, name, comment):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            c.execute("""
                    INSERT INTO
                        project(project_name, project_comment, created, updated)
                    VALUES
                        (?, ?, ?, ?)
                """, (name, comment, now, now))
            return c.lastrowid

    def update_project_deploy(self, project_id, model_id):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            c.execute("""UPDATE project
                SET deploy_model_id=?, updated=?
                WHERE project_id=?""", (model_id, now, project_id))
        return c.lastrowid

    def register_model(self, project_id, dataset_def_id, hyper_parameters, algorithm, algorithm_params):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            dumped_hyper_parameters = pickle_dump(hyper_parameters)
            dumped_algorithm_params = pickle_dump(algorithm_params)
            train_loss_list = pickle_dump([])
            validation_loss_list = pickle_dump([])
            best_epoch_validation_result = pickle_dump({})

            c.execute("""
                    INSERT INTO
                        model(project_id, dataset_def_id, hyper_parameters, algorithm,
                            algorithm_params, train_loss_list,
                            validation_loss_list, best_epoch_validation_result,
                            created, updated)
                    VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (project_id, dataset_def_id, dumped_hyper_parameters, algorithm,
                      dumped_algorithm_params, train_loss_list,
                      validation_loss_list, best_epoch_validation_result,
                      now, now))
        return c.lastrowid

    def update_model_state(self, model_id, state):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            c.execute("""
                    UPDATE model
                    SET
                        state=?, updated=?
                    WHERE model_id=?
                """, (state, now, model_id))
        self.db.commit()
        return c.lastrowid

    def update_model_last_epoch(self, model_id, last_epoch):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            c.execute("""
                    UPDATE model
                    SET
                        last_epoch=?, updated=?
                    WHERE model_id=?
                """, (last_epoch, now, model_id))
        return c.lastrowid

    def update_model_running_info(self, model_id, last_batch, total_batch, last_train_loss, running_state):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            c.execute("""
                    UPDATE model
                    SET
                        last_batch=?, total_batch=?,
                        last_train_loss=?, running_state=?, updated=?
                    WHERE model_id=?
                """, (last_batch, total_batch, last_train_loss, running_state, now, model_id))
        return c.lastrowid

    def update_model_loss_list(self, model_id, train_loss_list, validation_loss_list):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            dumped_train_loss_list = pickle_dump(train_loss_list)
            dumped_validation_loss_list = pickle_dump(validation_loss_list)

            c.execute("""
                    UPDATE model
                    SET
                        train_loss_list=?, validation_loss_list=?,
                        updated=?
                    WHERE model_id=?
                """, (dumped_train_loss_list, dumped_validation_loss_list, now, model_id))
        return c.lastrowid

    def update_model_validation_result(self, model_id, best_validation_result):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            dumped_best_validation_result = pickle_dump(best_validation_result)
            c.execute("""
                      UPDATE model
                      SET
                          best_epoch_validation_result=?, updated=?
                      WHERE model_id=?
                  """, (dumped_best_validation_result,
                        now, model_id))

    def update_model_best_epoch(self, model_id, best_epoch, best_iou,
                                best_map, best_weight, best_validation_result):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            dumped_best_validation_result = pickle_dump(best_validation_result)
            c.execute("""
                      UPDATE model
                      SET
                          best_epoch=?, best_epoch_iou=?,
                          best_epoch_map=?, best_epoch_weight=?,
                          best_epoch_validation_result=?, updated=?
                      WHERE model_id=?
                  """, (best_epoch, best_iou, best_map,
                        best_weight, dumped_best_validation_result,
                        now, model_id))

    def register_epoch(self, model_id, nth_epoch):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            c.execute("""
                    INSERT INTO
                    epoch(model_id, nth_epoch, created, updated)
                    VALUES
                    (?, ?, ?, ?)
                """, (model_id, nth_epoch, now, now))
        return c.lastrowid

    def register_dataset_v0(self, train_data_count, valid_data_count, class_list):
        with self.db:
            c = self.cursor()
            class_names = {i: n for i, n in enumerate(class_list)}
            dumped_class_names = pickle_dump(class_names)
            c.execute("""
                    REPLACE INTO
                        dataset_v0(dataset_id, train_data_count, valid_data_count, class_names)
                    VALUES
                        (?, ?, ?, ?);
                """, (0, train_data_count, valid_data_count, dumped_class_names))

    def update_epoch(self, epoch_id, train_loss, validation_loss, epoch_iou, epoch_map):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            c.execute("""
                    UPDATE epoch
                    SET
                        train_loss=?, validation_loss=?, epoch_iou=?, epoch_map=?, updated=?
                    WHERE epoch_id=?
                  """, (train_loss, validation_loss, epoch_iou, epoch_map, now, epoch_id))

    def delete_project(self, project_id):
        with self.db:
            c = self.cursor()
            c.execute("""
                    DELETE FROM project WHERE project_id=?
                """, (project_id,))

    def fetch_projects(self, fields='project_id', order_by='updated DESC'):
        with self.db:
            c = self.cursor()
            sql = "SELECT " + fields + " FROM project ORDER BY " + order_by
            c.execute(sql)

            ret = {}
            for index, data in enumerate(c):
                item = {}
                for j, f in enumerate(fields.split(',')):
                    item[f] = data[j]
                ret.update({index: item})
            return ret

    def fetch_project(self, project_id, fields='project_id'):
        with self.db:
            c = self.cursor()
            sql = "SELECT " + fields + " FROM project WHERE project_id=?"
            c.execute(sql, (project_id,))

            ret = {}
            for index, data in enumerate(c):
                item = {}
                for j, f in enumerate(fields.split(',')):
                    item[f] = data[j]
                ret.update({index: item})
            return ret[0]

    def fetch_models(self, project_id, order_by='model_id DESC'):
        with self.db:
            c = self.cursor()
            fields = "model_id,project_id,dataset_def_id,hyper_parameters,algorithm,algorithm_params,state,train_loss_list,validation_loss_list,best_epoch,best_epoch_iou,best_epoch_map,best_epoch_validation_result,last_epoch,last_batch,total_batch,last_train_loss,running_state"

            sql = "SELECT " + fields + \
                " FROM model WHERE project_id=? AND state!=3 ORDER BY " + order_by
            c.execute(sql, (project_id,))

            blob_items = ['hyper_parameters', 'algorithm_params',
                          'train_loss_list', 'validation_loss_list',
                          'best_epoch_validation_result']
            ret = {}
            for index, data in enumerate(c):
                item = {}
                for j, f in enumerate(fields.split(',')):
                    if f in blob_items:
                        item[f] = pickle_load(data[j])
                    else:
                        item[f] = data[j]
                ret.update({index: item})
            return ret

    def fetch_running_models(self, project_id, order_by='model_id'):
        with self.db:
            c = self.cursor()
            fields = "model_id,project_id,hyper_parameters,dataset_def_id,algorithm,algorithm_params,state,train_loss_list,validation_loss_list,best_epoch,best_epoch_iou,best_epoch_map,best_epoch_validation_result,last_epoch,last_batch,total_batch,last_train_loss,running_state"

            sql = "SELECT " + fields + \
                " FROM model WHERE project_id=? AND state=1 ORDER BY " + order_by
            c.execute(sql, (project_id,))

            blob_items = ['hyper_parameters', 'algorithm_params',
                          'train_loss_list', 'validation_loss_list',
                          'best_epoch_validation_result']
            ret = {}
            for index, data in enumerate(c):
                item = {}
                for j, f in enumerate(fields.split(',')):
                    if f in blob_items:
                        item[f] = pickle_load(data[j])
                    else:
                        item[f] = data[j]
                ret.update({index: item})
            return ret

    def fetch_model(self, project_id, model_id, fields='model_id'):
        with self.db:
            c = self.cursor()
            sql = "SELECT " + fields + " FROM model WHERE project_id=? AND model_id=?"
            c.execute(sql, (project_id, model_id))

            blob_items = ['hyper_parameters', 'algorithm_params',
                          'train_loss_list', 'validation_loss_list',
                          'best_epoch_validation_result']
            ret = {}
            for index, data in enumerate(c):
                item = {}
                for j, f in enumerate(fields.split(',')):
                    if f in blob_items:
                        item[f] = pickle_load(data[j])
                    else:
                        item[f] = data[j]
                ret.update({index: item})
            if ret:
                return ret[0]
            else:
                return {}

    def fetch_deployed_model_id(self, project_id):
        with self.db:
            c = self.cursor()
            fields = "deploy_model_id"

            sql = "SELECT " + fields + \
                " FROM project WHERE project_id=?"
            c.execute(sql, (project_id,))

            blob_items = []
            ret = {}
            for index, data in enumerate(c):
                item = {}
                for j, f in enumerate(fields.split(',')):
                    if f in blob_items:
                        item[f] = pickle_load(data[j])
                    else:
                        item[f] = data[j]
                ret.update({index: item})
            return ret

    def fetch_dataset_v0(self):
        with self.db:
            c = self.cursor()
            c.execute("""
                SELECT train_data_count, valid_data_count, class_names
                FROM dataset_v0
                WHERE dataset_id=0
                """)
            ret = {}
            for data in c:
                ret.update({
                    "train_data_count": data[0],
                    "valid_data_count": data[1],
                    "class_names": pickle_load(data[2])
                })
        return ret

    def register_dataset(self, dataset_name, train_size, type):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            c.execute("""
                INSERT INTO
                dataset(dataset_name, train_size, type,
                        created, updated)
                VALUES
                    (?, ?, ?, ?, ?)
            """, (dataset_name, train_size, type, now, now))
            return c.lastrowid

    def update_dataset(self, dataset_id, train_num, valid_num, train_class_num,
                       valid_class_num, train_img_list, valid_img_list,
                       train_label_list, valid_label_list, class_names,
                       img_dir, label_dir):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            dumped_train_img_list = pickle_dump(train_img_list)
            dumped_valid_img_list = pickle_dump(valid_img_list)
            dumped_train_label_list = pickle_dump(train_label_list)
            dumped_valid_label_list = pickle_dump(valid_label_list)
            dumped_class_names = pickle_dump(class_names)
            c.execute("""
                UPDATE dataset
                SET train_num=?, valid_num=?, train_class_num=?,
                    valid_class_num=?, train_img_list=?,
                    valid_img_list=?, train_label_list=?,
                    valid_label_list=?, class_names=?,
                    img_dir=?, label_dir=?, updated=?
                WHERE dataset_id=?
            """, (train_num, valid_num, train_class_num,
                  valid_class_num, dumped_train_img_list,
                  dumped_valid_img_list, dumped_train_label_list,
                  dumped_valid_label_list, dumped_class_names,
                  img_dir, label_dir, now, dataset_id))

    def fetch_datasets(self):
        with self.db:
            c = self.cursor()
            c.execute("""
                SELECT dataset_id, dataset_name
                FROM dataset
                WHERE invalid=0
                """)

            ret = {}
            for index, data in enumerate(c):
                ret.update({index: {
                    "dataset_id": data[0],
                    "dataset_name": data[1]
                }})
            return ret

    def fetch_dataset(self, dataset_id):
        with self.db:
            c = self.cursor()
            c.execute("""
                SELECT dataset_name, train_size, type,
                       train_num, valid_num, train_class_num,
                       valid_class_num, train_img_list,
                       valid_img_list, train_label_list,
                       valid_label_list, class_names
                FROM dataset
                WHERE dataset_id=? AND invalid=0
                """, (dataset_id,))

            ret = {}
            for index, data in enumerate(c):
                ret.update({index: {
                    "dataset_id": dataset_id,
                    "dataset_name": data[0],
                    "train_size": data[1],
                    "type": data[2],
                    "train_num": data[3],
                    "valid_num": data[4],
                    "train_class_num": data[5],
                    "valid_class_num": data[6],
                    "train_img_list": pickle_load(data[7]),
                    "valid_img_list": pickle_load(data[8]),
                    "train_label_list": pickle_load(data[9]),
                    "valid_label_list": pickle_load(data[10]),
                    "class_names": pickle_load(data[11])
                }})
            return ret

    def fetch_dataset_train_size(self, dataset_id):
        with self.db:
            c = self.cursor()
            c.execute("""
                SELECT train_size
                FROM dataset
                WHERE dataset_id=? AND invalid=0
                """, (dataset_id,))

            ret = {}
            for index, data in enumerate(c):
                ret.update({
                    "train_size": data[0]
                })
            return ret

    def register_dataset_def(self, name, ratio, train_imgs, valid_imgs, class_map):

        train_imgs = json.dumps(train_imgs)
        valid_imgs = json.dumps(valid_imgs)
        class_map = json.dumps(class_map)

        now = datetime.datetime.now()
        with self.db:
            c = self.cursor()
            c.execute("""
                INSERT INTO dataset_def(name, ratio, train_imgs, valid_imgs, class_map,
                    created, updated)
                VALUES(?, ?, ?, ?, ?, ?, ?)
            """, (name, ratio, train_imgs, valid_imgs, class_map, now, now))
            return c.lastrowid

    def fetch_dataset_defs(self):
        with self.db:
            ret = []
            c = self.cursor()
            c.execute("""SELECT id, name, ratio, valid_imgs, class_map, created, updated FROM dataset_def""")
            for rec in c:
                ret.append([
                    rec[0], rec[1], rec[2],
                    json.loads(rec[3]),
                    json.loads(rec[4]),
                    rec[5].isoformat(), rec[6].isoformat()
                ])
            return ret

    def fetch_dataset_def(self, id):
        with self.db:
            c = self.cursor()
            c.execute(
                """SELECT id, name, ratio, train_imgs, valid_imgs, class_map, created, updated FROM dataset_def""")

            for rec in c:
                id, name, ratio, train_imgs, valid_imgs, class_map, created, updated = rec
                train_imgs = json.loads(train_imgs)
                valid_imgs = json.loads(valid_imgs)
                class_map = json.loads(class_map)
                return (id, name, ratio, train_imgs, valid_imgs, class_map, created, updated)
            return None

    def fetch_class_map(self):
        with self.db:
            c = self.cursor()
            c.execute("""
                SELECT class_map FROM dataset_def
                LIMIT 1""")
            for rec in c:
                class_map = json.loads(rec[0])
                return {"class_map": class_map}
            return None


global storage
storage = Storage()
if not storage.is_poject_exists():
    storage.register_project('objdetection', 'comment')
