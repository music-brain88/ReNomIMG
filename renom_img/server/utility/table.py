# coding: utf-8
import re
from datetime import datetime
from sqlalchemy.dialects.sqlite import DATETIME
from sqlalchemy.sql import func
from sqlalchemy import Column, Integer, String, Float, DateTime, text
from sqlalchemy import ForeignKey, BLOB, CLOB, TEXT, NUMERIC
from sqlalchemy.orm import relationship
from renom_img.server.utility.DAO import Base
from renom_img.server import DB_DIR_TRAINED_WEIGHT


class Task(Base):

    __tablename__ = 'task'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    deployed_model_id = Column(Integer)
    relationship("Model")

    def __repr__(self):
        return "<Task(id='%s', name='%s', deployed_model_id='%s')>" % (
            self.id,
            self.name,
            self.deployed_model_id
        )


class TestDataset(Base):
    __tablename__ = 'test_dataset'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    description = Column(TEXT)
    task_id = Column(Integer)
    data = Column(CLOB)
    created = Column(DateTime)
    class_info = Column(BLOB)

    relationship("Dataset")

    def __repr__(self):
        return "<TestDataset(id='%s', name='%s', task_id='%s')>" % (
            self.id,
            self.name,
            self.task_id,
        )


class Dataset(Base):
    __tablename__ = 'dataset'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    ratio = Column(Float, nullable=False)
    description = Column(TEXT)
    task_id = Column(Integer)
    train_data = Column(BLOB)
    valid_data = Column(BLOB)
    class_map = Column(BLOB)
    class_info = Column(BLOB)
    created = Column(DateTime(timezone=True), server_default=func.now())
    test_dataset_id = Column(
        Integer,
        ForeignKey('test_dataset.id', ondelete='CASCADE')
    )

    def __repr__(self):
        line = """
                <Dataset(
                    id='%s',
                    name='%s',
                    ratio='%s',
                    description='%s',
                    train_imgs='%s',
                    valid_imgs='%s'
                    class_map='%s',
                    class_info='%s'
                    created='%s'
                    test_dataset_id='%s',
                )>
                """
        return line % (
            self.id,
            self.name,
            self.ratio,
            self.description,
            self.train_data,
            self.valid_data,
            self.class_map,
            self.class_info,
            self.created,
            self.test_dataset_id
        )


class Model(Base):

    __tablename__ = 'model'

    # Must be given
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey('task.id'))
    dataset_id = Column(Integer, ForeignKey('dataset.id'))
    algorithm_id = Column(Integer)
    hyper_parameters = Column(BLOB)

    # Modified during training.
    state = Column(Integer, server_default=text('0'))
    running_state = Column(Integer, server_default=text('0'))

    train_loss_list = Column(BLOB, nullable=True)
    valid_loss_list = Column(BLOB, nullable=True)

    best_epoch_valid_result = Column(BLOB, nullable=True)
    best_epoch_weight = Column(TEXT)

    total_epoch = Column(Integer, server_default=text('0'))
    nth_epoch = Column(Integer, server_default=text('0'))

    total_batch = Column(Integer, server_default=text('0'))
    nth_batch = Column(Integer, server_default=text('0'))

    last_weight = Column(TEXT)
    last_batch_loss = Column(NUMERIC, server_default=text('0'))

    last_prediction_result = Column(BLOB, nullable=True)

    # Dates
    created = Column(DateTime(timezone=True), server_default=func.now())
    updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __init__(self, *arg, **kwargs):
        # Register path.
        super(Model, self).__init__(*arg, **kwargs)
        unixtime = datetime.now().strftime('%s')
        last_weight_name = str(DB_DIR_TRAINED_WEIGHT / "last_model_{}.h5".format(unixtime))
        self.last_weight = last_weight_name
        best_weight_name = str(DB_DIR_TRAINED_WEIGHT / "best_model_{}.h5".format(unixtime))
        self.best_epoch_weight = best_weight_name

    def __repr__(self):
        return "<Model(id='%s', task_id='%s')>" % (
            self.id,
            self.task_id
        )

    def update_params(self, dict):
        for name, value in dict.items():
            if name in self.__dict__:
                setattr(self, name, value)
