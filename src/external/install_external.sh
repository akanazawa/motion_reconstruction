#!/usr/bin/env bash
echo "Note you have to have torch 0.4.0!"
pip install -U torch==0.4.0
git clone https://github.com/daniilidis-group/neural_renderer.git
cd neural_renderer
python setup.py develop
