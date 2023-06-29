#!/bin/sh
# execution script for SLURM cluster 


for name in li1p li1l li2p
do
  for idx in 4 14 19 24 29 34 39 44 49 54 59 64 69 74 79 84 89 94 99 104 109 
  do
    echo "Submitting job ${name} version ${idx}"
    JOB_NAME="INF_${name}_${idx}"
    SCRIPT_PATH="/users/felix.wagner/cryoenv/tests/inference_model.py"
    CONTAINER_PATH="/users/felix.wagner/mosquitto_latest.sif"
    OUTPUT_FILE="/users/felix.wagner/outputs/${JOB_NAME}.out"
    SBATCH_OPTIONS=" -c 1 --mem=4G --output=${OUTPUT_FILE} --job-name=${JOB_NAME} --time=30"
    SINGULARITY_OPTIONS=" -c -B /eos/ -H /users/felix.wagner"
    PYTHON_OPTIONS=" -u "
    CMD_ARGUMENTS=" --version ${idx} --detector ${name} --do_dt --buffer_save_path_inf_dt /users/felix.wagner/cryoenv/tests/buffers_inf_dt_40mretrain --path_checkpoint /users/felix.wagner/cryoenv/tests/output_40m/run1_checkpoint-89500" 

    sbatch ${SBATCH_OPTIONS} --wrap="time singularity exec ${SINGULARITY_OPTIONS} ${CONTAINER_PATH} python3 ${PYTHON_OPTIONS} ${SCRIPT_PATH} ${CMD_ARGUMENTS}"
  done
done

exit 0
