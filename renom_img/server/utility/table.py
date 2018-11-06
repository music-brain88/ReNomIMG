# coding: utf-8
import re
from sqlalchemy.dialects.sqlite import DATETIME
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy import ForeignKey, BLOB, CLOB, TEXT, NUMERIC
from sqlalchemy.orm import relationship
from renom_img.server.utility.DAO import Base


class Algorithm(Base):

    __tablename__ = 'algorithm'

    algorithm_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)

    relationship("Model")

    def __repr__(self):
        return "<Algorithm(algorithm_id='%s', name='%s')>" % (
            self.algorithm_id,
            self.name
        )


class Task(Base):

    __tablename__ = 'task'

    task_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    deployed_model_id = Column(Integer)

    relationship("Model")

    def __repr__(self):
        return "<Task(task_id='%s', name='%s', deployed_model_id='%s')>" % (
            self.task_id,
            self.name,
            self.deployed_model_id
        )


class Test_Dataset(Base):
    __tablename__ = 'test_dataset'

    test_dataset_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    description = Column(TEXT)
    test_imgs = Column(CLOB)
    created = Column(DateTime)

    relationship("Dataset")

    def __repr__(self):
        return "<Test_Dataset(test_dataset_id='%s', name='%s')>" % (
            self.test_dataset_id,
            self.name
        )


class Dataset(Base):
    __tablename__ = 'dataset'

    dataset_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    ratio = Column(Float, nullable=False)
    description = Column(TEXT)
    train_imgs = Column(CLOB)
    valid_imgs = Column(CLOB)
    class_map = Column(BLOB)
    class_tag_list = Column(BLOB)
    created = Column(DateTime, nullable=False)
    test_dataset_id = Column(
        Integer,
        ForeignKey('test_dataset.test_dataset_id', ondelete='CASCADE')
    )

    def __repr__(self):
        line = """
                <Dataset(
                    test_dataset_id='%s',
                    name='%s',
                    ratio='%s',
                    description='%s',
                    train_imgs='%s',
                    valid_imgs='%s'
                    class_map='%s',
                    class_tag_list='%s'
                    created='%s'
                )>
                """
        return line % (
            self.dataset_id,
            self.name,
            self.ratio,
            self.description,
            self.train_imgs,
            self.valid_imgs,
            self.class_map,
            self.class_tag_list,
            self.created,
            self.test_dataset_id
        )


class Model(Base):

    __tablename__ = 'model'

    model_id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey('task.task_id'))
    dataset_id = Column(Integer, ForeignKey('dataset.dataset_id'))
    algorithm_id = Column(Integer,  ForeignKey('algorithm.algorithm_id'))
    hyper_parameters = Column(BLOB)
    state = Column(Integer)
    train_loss_list = Column(BLOB)
    validation_loss_list = Column(BLOB)
    best_epoch = Column(Integer)
    best_epoch_iou = Column(NUMERIC)
    best_epoch_map = Column(NUMERIC)
    best_epoch_validation_reuslt = Column(BLOB)
    best_epoch_weight = Column(TEXT)
    last_epoch = Column(Integer)
    last_weight = Column(TEXT)
    last_batch = Column(Integer)
    last_train_loss = Column(NUMERIC)
    total_batch = Column(Integer)
    running_state = Column(Integer)

    dt = DATETIME(storage_format="%(year)04d/%(month)02d/%(day)02d "
                             "%(hour)02d:%(min)02d:%(second)02d",
              regexp=r"(\d+)/(\d+)/(\d+) (\d+)-(\d+)-(\d+)"
                  )

    created = Column(DATETIME, nullable=False)
    updated = Column(DATETIME, nullable=False)

    def __repr__(self):
        return "<Model(model_id='%s', task_id='%s')>" % (
            self.model_id,
            self.task_id
        )
