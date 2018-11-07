import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound
from renom_img.server import DB_DIR

DATABASE = os.path.join('sqlite:///', DB_DIR, 'new_storage.db')
engine = create_engine(
    DATABASE,
    encoding="utf-8",
    echo=False  # Trueだと実行のたびにSQLが出力される
)

# Session = sessionmaker(
#     sessionmaker(
#         autocommit = True,
#         autoflush = True,
#         bind = engine
#     )
# )
Session = sessionmaker(
    bind=engine
)
session = Session()

Base = declarative_base()
