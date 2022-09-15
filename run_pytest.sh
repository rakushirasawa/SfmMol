#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:.
pytest --cov --cov-branch --cov-report html --cov-report term-missing --cov-report xml
#codecov --token XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXX
dirname=`pwd | xargs basename`
pdoc3 --html --force -o docs  ${dirname,,}
