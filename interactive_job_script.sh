#!/bin/bash
qsub \
-I \
-N interactive_job \
-M daich@umich.edu \
-m abe \
-A mdstproject_fluxg \
-q fluxg \
-l qos=flux,nodes=2:gpus=2,pmem=8gb,walltime=00:04:00:00 \
-j oe \
-V \

