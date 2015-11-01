#!/bin/bash
#PBS -l nodes=1:ppn=12:gpus=1:titan
#PBS -l walltime=100:00:00
#PBS -l mem=64GB
#PBS -N level1GraspLift
#PBS -M ajr619@nyu.edu
#PBS -j oe

module purge

SRCDIR=$HOME/kaggle-winning-grasp-lift-eeg/
RUNDIR=$SCRATCH/kaggle-winning-grasp-lift-eeg/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR

cd $PBS_O_WORKDIR
cp -R $SRCDIR/* $RUNDIR

cd $RUNDIR

module load scikit-learn/intel/0.16.1;
module load numpy/intel/1.9.2;
module load virtualenv/12.1.1;
module load cuda/7.0.28;

virtualenv .venv
virtualenv -p `which python` .venv

source .venv/bin/activate;
pip install pandas;
pip install pyriemann;
pip install mne;
pip install XGBoost;
pip install Theano;
pip install keras;
pip install Lasange;
pip install Lasagne;
pip install nolearn;
pip install hyperopt;
pip install progressbar;

cd lvl1;

./genAll.sh


