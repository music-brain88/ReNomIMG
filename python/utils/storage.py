# coding: utf-8

import _pickle as cPickle
import datetime
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRAGE_DIR = os.path.join(BASE_DIR, "../../.storage")


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
                 created TIMESTAMP NOT NULL,
                 updated TIMESTAMP NOT NULL)
            """)

        c.execute("""
                CREATE TABLE IF NOT EXISTS model
                (model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 project_id INTEGER NOT NULL,
                 dataset_id INTEGER NOT NULL,
                 total_epoch INTEGER NOT NULL,
                 seed INTEGER NOT NULL,
                 algorithm INTEGER NOT NULL,
                 hyper_parameter BLOB,
                 state INTEGER NOT NULL,
                 best_epoch INTEGER,
                 max_memory_usage INTEGER,
                 max_memory_usage_forward INTEGER,
                 created TIMESTAMP NOT NULL,
                 updated TIMESTAMP NOT NULL)
            """)

        c.execute("""
                CREATE TABLE IF NOT EXISTS epoch
                (epoch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 project_id INTEGER NOT NULL,
                 model_id INTEGER NOT NULL,
                 nth_epoch INTEGER NOT NULL,
                 train_loss NUMBER,
                 validation_loss NUMBER,
                 weight TEXT,
                 iou_value NUMBER,
                 map_value NUMBER,
                 validation_results BLOB,
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
                 total_class INTEGER,
                 train_dir_path TEXT,
                 train_label_path TEXT,
                 validation_dir_path TEXT,
                 validation_label_path TEXT,
                 test_dir_path TEXT,
                 test_label_path TEXT,
                 created TIMESTAMP NOT NULL,
                 updated TIMESTAMP NOT NULL)
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

    def register_model(self, project_id, dataset_id, total_epoch, seed, algorithm, hyper_parameter):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            dumped_hyper_parameter = cPickle.dumps(hyper_parameter)
            c.execute("""
                    INSERT INTO
                        model(project_id, dataset_id, total_epoch, seed, algorithm, hyper_parameter, state, created, updated)
                    VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (project_id, dataset_id, total_epoch, seed, algorithm, dumped_hyper_parameter, 0, now, now))
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

    def update_model_best_epoch(self, model_id, best_epoch):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            c.execute("""
                      UPDATE model
                      SET
                          best_epoch=?, updated=?
                      WHERE model_id=?
                  """, (best_epoch, now, model_id))

    def register_dataset(self, total_class, train_dir_path, train_label_path,
                         validation_dir_path, validation_label_path, test_dir_path, test_label_path):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            c.execute("""
                    INSERT INTO
                        dataset(total_class, train_dir_path, train_label_path,
                                validation_dir_path, validation_label_path,
                                test_dir_path, test_label_path,
                                created, updated)
                    VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (total_class, train_dir_path, train_label_path,
                      validation_dir_path, validation_label_path,
                      test_dir_path, test_label_path, now, now))
        return c.lastrowid

    def register_epoch(self, project_id, model_id, nth_epoch, validation_results):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            dumped_result = cPickle.dumps(validation_results)
            c.execute("""
                    INSERT INTO
                    epoch(project_id, model_id, nth_epoch, validation_results, created, updated)
                    VALUES
                    (?, ?, ?, ?, ?, ?)
                """, (project_id, model_id, nth_epoch, dumped_result, now, now))
        return c.lastrowid

    def register_dataset_v0(self, train_data_count, valid_data_count, class_list):
        with self.db:
            c = self.cursor()
            class_names = {i: n for i, n in enumerate(class_list)}
            dumped_class_names = cPickle.dumps(class_names)
            c.execute("""
                    REPLACE INTO
                        dataset_v0(dataset_id, train_data_count, valid_data_count, class_names)
                    VALUES
                        (?, ?, ?, ?);
                """, (0, train_data_count, valid_data_count, dumped_class_names))

    def update_epoch(self, epoch_id, iou_value, map_value, validation_results):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            dumped_result = cPickle.dumps(validation_results)
            c.execute("""
                      UPDATE epoch
                      SET
                          iou_value=?, map_value=?,
                          validation_results=?, updated=?
                      WHERE epoch_id=?
                  """, (iou_value, map_value,
                        dumped_result, now, epoch_id))

    def update_epoch_loss_weight(self, epoch_id, train_loss, validation_loss, weight):
        with self.db:
            c = self.cursor()
            now = datetime.datetime.now()
            c.execute("""
                      UPDATE epoch
                      SET
                          train_loss=?, validation_loss=?, weight=?, updated=?
                      WHERE epoch_id=?
                  """, (train_loss, validation_loss, weight, now, epoch_id))

    def delete_project(self, project_id):
        with self.db:
            c = self.cursor()
            c.execute("""
                    DELETE FROM epoch WHERE project_id=?
                """, (project_id,))
            c.execute("""
                    DELETE FROM model WHERE project_id=?
                """, (project_id,))
            c.execute("""
                    DELETE FROM project WHERE project_id=?
                """, (project_id,))

    def fetch_projects(self):
        with self.db:
            c = self.cursor()
            c.execute("""
                SELECT project_id, project_name, project_comment
                FROM project
                ORDER BY updated DESC
                """)
            ret = {}
            for index, data in enumerate(c):
                ret.update({index: {
                    "project_id": data[0],
                    "project_name": data[1],
                    "project_comment": data[2]
                }})
            return ret

    def fetch_project(self, project_id):
        with self.db:
            c = self.cursor()
            c.execute("""
                SELECT project_name, project_comment
                FROM project
                WHERE project_id=?
                """, (project_id,))
            ret = {}
            for index, data in enumerate(c):
                ret.update({
                    "project_id": project_id,
                    "project_name": data[0],
                    "project_comment": data[1]
                })
            return ret

    def fetch_models(self, project_id):
        with self.db:
            c = self.cursor()
            c.execute("""
                SELECT model_id, dataset_id, total_epoch, seed, algorithm, hyper_parameter, state, best_epoch
                FROM model
                WHERE project_id=? AND state<3
                ORDER BY model_id DESC
                """, (project_id,))
            ret = {}
            for index, data in enumerate(c):
                ret.update({index: {
                    "model_id": data[0],
                    "dataset_id": data[1],
                    "total_epoch": data[2],
                    "seed": data[3],
                    "algorithm": data[4],
                    "hyper_parameter": cPickle.loads(data[5], encoding='ascii'),
                    "state": data[6],
                    "best_epoch": data[7],
                }})
            return ret

    def fetch_model(self, project_id, model_id):
        with self.db:
            c = self.cursor()
            c.execute("""
                SELECT dataset_id, total_epoch, seed, algorithm,
                       hyper_parameter, state, best_epoch
                FROM model
                WHERE project_id=? AND model_id=?
                """, (project_id, model_id))
            ret = {}
            for index, data in enumerate(c):
                ret.update({
                    "model_id": model_id,
                    "dataset_id": data[0],
                    "total_epoch": data[1],
                    "seed": data[2],
                    "algorithm": data[3],
                    "hyper_parameter": cPickle.loads(data[4], encoding='ascii'),
                    "state": data[5],
                    "best_epoch": data[6],
                })
            return ret

    def fetch_epochs(self, project_id, model_id):
        with self.db:
            c = self.cursor()
            c.execute("""
                SELECT epoch_id, nth_epoch, train_loss, validation_loss,
                       weight, iou_value, map_value, validation_results
                FROM epoch
                WHERE project_id=? AND model_id=?
                ORDER BY nth_epoch
                """, (project_id, model_id))
            ret = {}
            for data in c:
                ret.update({data[1]: {
                    "epoch_id": data[0],
                    "nth_epoch": data[1],
                    "train_loss": data[2],
                    "validation_loss": data[3],
                    "weight": data[4],
                    "iou_value": data[5],
                    "map_value": data[6],
                    "validation_results": cPickle.loads(data[7], encoding='ascii')
                }})
            return ret

    def fetch_epoch(self, project_id, model_id, nth_epoch):
        with self.db:
            c = self.cursor()
            c.execute("""
                SELECT epoch_id, nth_epoch, train_loss, validation_loss,
                       weight, iou_value, map_value, validation_results
                FROM epoch
                WHERE project_id=? AND model_id=? AND nth_epoch=?
                """, (project_id, model_id, nth_epoch))
            ret = {}
            for data in c:
                ret.update({
                    "epoch_id": data[0],
                    "nth_epoch": data[1],
                    "train_loss": data[2],
                    "validation_loss": data[3],
                    "weight": data[4],
                    "iou_value": data[5],
                    "map_value": data[6],
                    "validation_results": cPickle.loads(data[7], encoding='ascii')
                })
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
                    "class_names": cPickle.loads(data[2], encoding='ascii')
                })
        return ret


global storage
storage = Storage()
if not storage.is_poject_exists():
    storage.register_project('objdetection', 'comment')
    print("Project Created")
