#!/bin/sh
# execution script for SLURM cluster 

DOUBLE_TES=false

for name in li1p li1l li2p
do
  for (( idx=0; idx<=4; idx++))
  do
    echo "Submitting job ${name} version ${idx}"
    JOB_NAME="SAC_${name}_${idx}"
    SCRIPT_PATH="/users/felix.wagner/cryoenv/tests/train_sac.py"
    CONTAINER_PATH="/users/felix.wagner/mosquitto_latest.sif"
    OUTPUT_FILE="/users/felix.wagner/outputs/${JOB_NAME}.out"
    SBATCH_OPTIONS=" -c 1 --mem=4G --output=${OUTPUT_FILE} --job-name=${JOB_NAME} --time=240"
    SINGULARITY_OPTIONS=" -c -B /eos/ -H /users/felix.wagner"
    PYTHON_OPTIONS=" -u "
    CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --double_tes ${DOUBLE_TES}"

    sbatch ${SBATCH_OPTIONS} --wrap="time singularity exec ${SINGULARITY_OPTIONS} ${CONTAINER_PATH} python3 ${PYTHON_OPTIONS} ${SCRIPT_PATH} ${CMD_ARGUMENTS}"
  done
done

exit 0
