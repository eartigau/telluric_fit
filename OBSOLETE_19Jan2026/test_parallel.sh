#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --account=def-rdoyon

cd /home/eartigau/projects/rrg-rdoyon/eartigau/tapas/test_fit/tapas_tellu/

alias lbl_env="source /home/eartigau/lbl_env/bin/activate"
module purge
module load StdEnv/2023
module load python/3.11
module load scipy-stack/2025a
echo "test parallel -- start"
python test_parallel.py
echo "test parallel -- end"



#module load StdEnv/2020 python/3.9 hdf5
#source /project/6102120/apero/spirou_bin/env/apero_drs_07/bin/activate
#source /project/6102120/apero/spirou_bin/settings/spirou_mini2_07295/spirou_mini2_07295.bash.setup
#cd /project/6102120/apero
#echo 'RESET'
#apero_reset.py --warn=False
#echo 'PRE-CHECK'
#apero_precheck.py mini_run3.ini
#echo 'PRECESSING TEST'
#apero_processing.py mini_run3.ini --test 'True'
#echo 'STARTING APERO PROCESSING 70 cores'
#apero_processing.py mini_run2.ini --cores 70
#sleep 30
#
