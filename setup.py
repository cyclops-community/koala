from setuptools import setup, find_packages
import os


VERSION_PATH = os.path.join(os.path.dirname(__file__), 'koala', 'VERSION')
with open(VERSION_PATH) as version_file:
    VERSION = version_file.read().strip()


setup(
    name='koala',
    version=VERSION,
    packages=find_packages(exclude=[]),
    package_data={
        'koala': ['VERSION'],
    },
    install_requires=[
        'numpy>=1.17',
    ],
)
