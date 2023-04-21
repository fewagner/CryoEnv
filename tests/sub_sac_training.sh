#!/bin/sh
# execution script for SLURM cluster 

# for double tes: put to time=600

for name in li1p li1l li2p
do
  for (( idx=105; idx<=109; idx++))
  do
    echo "Submitting job ${name} version ${idx}"
    JOB_NAME="SAC_${name}_${idx}"
    SCRIPT_PATH="/users/felix.wagner/cryoenv/tests/train_sac.py"
    CONTAINER_PATH="/users/felix.wagner/mosquitto_latest.sif"
    OUTPUT_FILE="/users/felix.wagner/outputs/${JOB_NAME}.out"
    SBATCH_OPTIONS=" -c 1 --mem=4G --output=${OUTPUT_FILE} --job-name=${JOB_NAME} --time=240"
    SINGULARITY_OPTIONS=" -c -B /eos/ -H /users/felix.wagner"
    PYTHON_OPTIONS=" -u "
    CMD_ARGUMENTS=" --version ${idx} --detector ${name} --scale 0.2 --lr 1e-3  --batch_size 64  --gamma 0.99  --gradient_steps 20  --tau 20.  --sweep " 

    sbatch ${SBATCH_OPTIONS} --wrap="time singularity exec ${SINGULARITY_OPTIONS} ${CONTAINER_PATH} python3 ${PYTHON_OPTIONS} ${SCRIPT_PATH} ${CMD_ARGUMENTS}"
  done
done

exit 0
