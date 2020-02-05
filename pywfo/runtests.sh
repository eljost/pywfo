#!/bin/bash

pytest -k test_cytosin --durations 0 -v
pytest -k test_cytosin --durations 0 -v --profile --profile-svg
