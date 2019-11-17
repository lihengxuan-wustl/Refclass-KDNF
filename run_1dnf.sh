#PBS -q old
#PBS -l nodes=1:ppn=1,walltime=168:00:00
#PBS -t 0-3

# set up the environment to use my python
export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
source activate py2

# cd into the directory with the python script
cd /home/hengxuanli/juba-lab/refclass-experiment-cython

python run_1dnf_lending.py -i $PBS_ARRAYID
