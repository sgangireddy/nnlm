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

#$ -N testing_tunedweights.py 
#$ -cwd
#$ -l h_rt=334:59:59
#$ -pe memory-2G 2
#$ -P inf_hcrc_cstr_udialogue 
#$ -q ecdf@@clock-2week
#$ -m beas
#$ -M s1264845@sms.ed.ac.uk

# Initialise environment module
. /etc/profile.d/modules.sh

ulimit -v 2097152  

#python env variable
source /exports/work/inf_hcrc_cstr_students/s1136550/software/path.sh

# Run the program
tgl.sh --cpu --numlib mkl train_nn_lm_modified.py -f $1 -c $2 -h $3 -p $4 -m $5 -l $6 -r $7 -e $8 -v $9 
