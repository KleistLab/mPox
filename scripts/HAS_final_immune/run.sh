#!/bin/bash
for i in {0..144}
do
   Python3 run.py $((i*100)) $((i*100+100)) run_immune 1
done