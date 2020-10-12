from setuptools import setup

setup(
   name='plepy',
   version='1.0',
   description='Identifiability tool using profile likelihood that interfaces with Pyomo',
   author='Monica Shapiro',
   author_email='monshapiro@gmail.com',
   packages=['plepy'],  #same as name
   install_requires=[
       'numpy ~= 1.19',
       'pyomo ~= 5.7',
       'scipy ~= 1.5',
       'matplotlib ~= 3.3',
       'seaborn ~= 0.10',
       'pandas ~= 1.1'
   ], #external packages as dependencies
   tests_require=['pytest']
)