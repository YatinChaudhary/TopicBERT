#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
sudo apt -yq install python3-venv
python3 -m venv projectenv
source projectenv/bin/activate
pip install wrapt --upgrade --ignore-installed
pip install --upgrade pip==20.1.1
pip install -r requirements.txt
deactivate