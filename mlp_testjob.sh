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
#$ -l h_rt=10:00:00
#$ -pe memory-2G 1
#$ -P inf_hcrc_cstr_udialogue
#$ -M s.gangireddy@sms.ed.ac.uk
#$ -m beas 

# Initialise environment module
. /etc/profile.d/modules.sh

ulimit -v 2097152  

#python env variable
source /exports/work/inf_hcrc_cstr_students/s1136550/software/path.sh

# Run the program
tgl.sh --cpu --numlib mkl testing_tunedweights.py -f $1 -c $2 -h $3 -p $4 -m $5 -n $6
