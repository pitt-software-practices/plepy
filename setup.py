from setuptools import setup

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
   name='plepy',
   version='1.0',
   description='Identifiability tool using profile likelihood that interfaces with Pyomo',
   author='Monica Shapiro',
   author_email='monshapiro@gmail.com',
   packages=['plepy'],  #same as name
   install_requires=requirements, #external packages as dependencies
)