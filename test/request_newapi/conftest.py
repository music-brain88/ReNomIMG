# content of a/conftest.py
import os
import sys
import traceback
import pytest
import tempfile
import shutil

import bottle
from bottle import default_app
from webtest import TestApp
from sqlalchemy import create_engine, event

from renom_img.server.utility.DAO import Base
from renom_img.server.server import get_app


class SaneTestApp(TestApp):
    def _check_status(self, status, res):
        # don't check status by default
        pass

    def _check_errors(self, res):
        # don't steal application's error message
        errors = res.errors
        if errors:
            print(res.errors, file=sys.stderr)


bottle.debug(True)


@pytest.fixture(scope="session", autouse=True)
def session():
    dirname = tempfile.mkdtemp()
    engine = create_engine('sqlite:///%s/renom_img_v2_0.db' % dirname, echo=True)

    Base.metadata.create_all(engine)

    try:
        yield 1
    finally:
        try:
            engine.dispose()
        except Exception:
            traceback.print_exc()

        try:
            shutil.rmtree(dirname, ignore_errors=True)
        except Exception:
            traceback.print_exc()

        try:
            shutil.rmtree('datasrc', ignore_errors=True)
            shutil.rmtree('storage', ignore_errors=True)
        except Exception:
            traceback.print_exc()


@pytest.fixture
def app():
    app = get_app()
    app.catchall = False  # raise exception on error
    return SaneTestApp(app)


# @pytest.fixture
# def sitedir(tmpdir):
#     d = tmpdir.mkdir('site')
#
#     d.mkdir('datasrc')
#     d.mkdir('storage')
#     d.mkdir('storage/trained_weight')
#
#     cwd = d.chdir()
#
#     try:
#         yield d
#     finally:
#         cwd.chdir()
