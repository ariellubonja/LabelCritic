#!/bin/sh
#PBS -l select=1:ncpus=20:ngpus=0
#PBS -l walltime=23:59:00
#PBS -j oe
#PBS -N AbdomenAtlas11Download
#PBS -q cpu

cd /fastwork/psalvador/JHU/AnnotationVLM
bash InstallEnv.sh > installation.log 2>&1