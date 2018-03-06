# coding: utf-8

import datetime
import os
import sys
import sqlite3
try:
    import _pickle as pickle
except:
    import cPickle as pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRAGE_DIR = os.path.join(BASE_DIR, "../../.storage")


def pickle_dump(obj):
    return pickle.dumps(obj)

def pickle_load(pickled_obj):
    if sys.version_info.major==2:
        if isinstance(pickled_obj, unicode):
            pickled_obj = pickled_obj.encode()
        return pickle.loads(pickled_obj)
    else:
        return pickle.loads(pickled_obj, encoding='ascii')

class Storage:
    def __init__(self):
        if not os.path.isdir(STRAGE_DIR):
            os.makedirs(STRAGE_DIR)

        dbname = os.path.join(STRAGE_DIR, 'test_storage.db')
        self.db = sqlite3.connect(dbname,
                                  check_same_thread=False,
                                  detect_types=sqlite3.PARSE_DECLTYPES,
                                  isolation_level=None)

        self.db.execute('PRAGMA journal_mode = WAL')
        self.db.execute('PRAGMA synchronous = OFF')
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
                 created TIMESTAMP NOT NULL,
                 updated TIMESTAMP NOT NULL)
            """)

        c.execute("""
                CREATE TABLE IF NOT EXISTS model
                (model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 project_id INTEGER NOT NULL REFERENCES project(project_id) ON DELETE CASCADE,
                 hyper_parameters BLOB,
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
                CREATE TABLE IF NOT EXISTS dataset_v0
                (dataset_id INTEGER PRIMARY KEY,
                 train_data_count INTEGER,
                 valid_data_count INTEGER,
                 class_names BLOB)
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

    def register_model(self, project_id, hyper_parameters, algorithm, algorithm_params):
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
                        model(project_id, hyper_parameters, algorithm,
                            algorithm_params, train_loss_list,
                            validation_loss_list, best_epoch_validation_result,
                            created, updated)
                    VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (project_id, dumped_hyper_parameters, algorithm,
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

    def fetch_models(self, project_id, fields='model_id', order_by='model_id'):
        with self.db:
            c = self.cursor()
            sql = "SELECT " + fields + " FROM model WHERE project_id=? AND state<3 ORDER BY " + order_by
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
            return ret[0]

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


global storage
storage = Storage()
if not storage.is_poject_exists():
    storage.register_project('objdetection', 'comment')
    print("Project Created")
