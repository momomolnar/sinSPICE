#PBS -A P54048000
#PBS -N MURAM_CIII_synth
#PBS -q main
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=127

# Load modules to match compile-time environment
#module purge

module load conda
conda activate chianti

cd /glade/u/home/mmolnar/Projects/sinSPICE

#python -i example.py
python -i example_CIII.py --ion "c" --ion_level 3 --lambda0 977 --numWaves 200 --dLambda 0.015
#python -i example_CIII.py --ion "mg" --ion_level 9 --lambda0 706.0 --numWaves 100 --dLambda 0.015
#python -i example_CIII.py --ion "o" --ion_level 6 --lambda0 1031.9 --numWaves 100 --dLambda 0.015