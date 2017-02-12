#!/bin/bash
qsub \
-I \
-N interactive_job \
-M daich@umich.edu \
-m abe \
-A engin_fluxg \
-q fluxg \
-l qos=flux,nodes=1:gpus=1,pmem=16gb,walltime=00:04:00:00 \
-j oe \
-V \

