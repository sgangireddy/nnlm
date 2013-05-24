#!/bin/sh
########################################
# GE job script for ECDF Cluster       #
#                                      #
# by ECDF System Team                  #
# ecdf-systems-team@lists.ed.ac.uk     #
#                                      #
########################################

# Grid Engine options
#module load blcr

#$ -N train_mlprnn.py 
#$ -cwd
#$ -l h_rt=48:00:00
#$ -pe memory-2G 2
#$ -P inf_hcrc_cstr_udialogue 
# -q ecdf@@clock-2week
#$ -m beas
#$ -M s1264845@sms.ed.ac.uk

# Initialise environment module
. /etc/profile.d/modules.sh

ulimit -v 2097152  

#python env variable
source /exports/work/inf_hcrc_cstr_students/s1136550/software/path.sh

# Run the program
tgl.sh --cpu --numlib mkl train_mlprnn.py $1 $2 $3 $4
