from setuptools import setup

setup(
   name='interact',
   version='0.1.0',
   description='Machine learning interact for flexible modeling',
   author='lruczu',
   author_email='lukasz.rucz@gmail.com',
   install_requires=[
        'numpy==1.16.3',
        'pandas==0.24.1',
        #'sklearn==0.22.1',
        'tensorflow==2.6.4'
   ],
   packages=[
        'interact',
        'interact.exceptions',
        'interact.fields',
        'interact.layers',
        'interact.models',
        'interact.utils',
   ],
)
