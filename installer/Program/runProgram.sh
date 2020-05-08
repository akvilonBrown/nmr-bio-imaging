#!/bin/bash
if [ ! -d "$1" ]; then
  echo Path to source  "$1" folder was not found!
  exit
fi
if [ ! -d "$2" ]; then
  echo Path to result $2 folder was not found!
  exit
fi

python3 solution.py "$1" "$2"
