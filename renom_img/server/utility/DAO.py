import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound
from renom_img.server import DB_DIR, create_directories

create_directories()

DATABASE = os.path.join('sqlite:///', str(DB_DIR), 'renom_img_v2_0.db')
engine = create_engine(
    DATABASE,
    encoding="utf-8",
    echo=False
)

Session = sessionmaker(
    bind=engine
)
session = Session()
Base = declarative_base()
