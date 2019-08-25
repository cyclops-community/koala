from setuptools import setup, find_packages
import os


VERSION_PATH = os.path.join(os.path.dirname(__file__), 'pepsi', 'VERSION')
with open(VERSION_PATH) as version_file:
    VERSION = version_file.read().strip()


setup(
    name='pepsi',
    version=VERSION,
    packages=find_packages(exclude=[]),
    package_data={
        'pepsi': ['VERSION'],
    },
    install_requires=[
        'numpy>=1.17',
    ],
)
