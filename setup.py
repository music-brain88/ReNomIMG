import os
import sys
import re
import shutil
import pathlib
from setuptools import setup, find_packages
import distutils.command.build

if sys.version_info < (3, 4):
    raise RuntimeError('renomimg requires Python3')

DIR = str(pathlib.Path(__file__).resolve().parent)

requires = [
    "bs4", "bottle", "glob2", "lxml", "Pillow",
    "PyYAML", "watchdog", "xmltodict"
]


entry_points = {
    'console_scripts': [
        'renomimg = server:main',
    ]
}


setup(
    name="renomimg",
    version="0.4b",
    entry_points=entry_points,
    packages=['renomimg'],
    install_requires=requires,
    include_package_data=True,
    zip_safe=True,
)
