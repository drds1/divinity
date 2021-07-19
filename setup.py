#!/usr/bin/env python
"""A Nose plugin to support IPython doctests.
"""
# upload to pip
# pip install .
# python setup.py sdist
# twine upload dist/divinity-0.1.tar.gz

from setuptools import setup

setup(
    name="divinity",
    version="0.1",
    url="https://github.com/dstarkey23/divinity",
    author="David Starkey",
    author_email="davidstarkey29@gmail.com",
    description="Auto Timeseries Forecast Package",
    license="Apache",
    packages=["divinity"],
    entry_points={},
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "statsmodels",
        "pre-commit",
        "pydocstyle",
        "pytest",
        "scikit-learn",
    ],
)


# setup(name='pycecream',
#      version='1.5.2',
#      description='python implementation of the cream accretion disc fitting code '
#                  'https://academic.oup.com/mnras/article-abstract/456/2/1960/1066664?redirectedFrom=PDF'
#      ,
#      long_description= 'add pycecream.dream light curve merging feature'
#      ,url='https://github.com/dstarkey23/pycecream',
#      author='dstarkey23',
#      author_email='ds207@st-andrews.ac.uk',
#      license='MIT',
#      packages=['pycecream'],
#      package_data={'': ['creaminpar.par','cream_f90.f90']},
#      install_requires=[
#      'pandas',
#      'numpy',
#      'matplotlib',
#      'scipy',
#      'astropy_stark',
#      'glob3',
#      'PyQt5',
#      'corner',
#      'seaborn'
#      ],
# extras_require={
#        'tests': [
#            'nose2==0.9.1',
#            'pre-commit==1.20.0',
#            'flake8==3.7.9',
#            'pydoc-markdown==2.0.4',
#            'tabulate==0.8.5',
#            'six==1.12.0'
#        ]
#    },
#      zip_safe=False)
