#!/bin/bash

arg1=${1:-"default"}

python train.py run_cfg@_global_="$arg1" "${@:2}"
    