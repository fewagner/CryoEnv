#!/bin/sh
# execution script for SLURM cluster 


for name in li1p li1l li2p
do
  for idx in 0 1 2 3 10 11 12 13 15 16 17 18 20 21 22 23 25 26 27 28 30 31 32 33 35 36 37 38 40 41 42 43 45 46 47 48 50 51 52 53 55 56 57 58 60 61 62 63 65 66 67 68 70 71 72 73 75 76 77 78 80 81 82 83 85 86 87 88 90 91 92 93 95 96 97 98 100 101 102 103 105 106 107 108 # 4 14 19 24 29 34 39 44 49 54 59 64 69 74 79 84 89 94 99 104 109 
  do
    echo "Submitting job ${name} version ${idx}"
    JOB_NAME="INF_${name}_${idx}"
    SCRIPT_PATH="/users/felix.wagner/cryoenv/tests/inference_model.py"
    CONTAINER_PATH="/users/felix.wagner/mosquitto_latest.sif"
    OUTPUT_FILE="/users/felix.wagner/outputs/${JOB_NAME}.out"
    SBATCH_OPTIONS=" -c 1 --mem=4G --output=${OUTPUT_FILE} --job-name=${JOB_NAME} --time=30"
    SINGULARITY_OPTIONS=" -c -B /eos/ -H /users/felix.wagner"
    PYTHON_OPTIONS=" -u "
    
    CMD_ARGUMENTS=" --version ${idx} --detector ${name} --do_sac --buffer_save_path_inf_sac /users/felix.wagner/cryoenv/tests/buffers_inf_sac" 
    sbatch ${SBATCH_OPTIONS} --wrap="time singularity exec ${SINGULARITY_OPTIONS} ${CONTAINER_PATH} python3 ${PYTHON_OPTIONS} ${SCRIPT_PATH} ${CMD_ARGUMENTS}"
    
    # CMD_ARGUMENTS=" --version ${idx} --detector ${name} --do_dt --buffer_save_path_inf_dt /users/felix.wagner/cryoenv/tests/buffers_inf_dt_10m --path_checkpoint /scratch-cbe/users/felix.wagner/rltests/output/checkpoint-90000" 
    # sbatch ${SBATCH_OPTIONS} --wrap="time singularity exec ${SINGULARITY_OPTIONS} ${CONTAINER_PATH} python3 ${PYTHON_OPTIONS} ${SCRIPT_PATH} ${CMD_ARGUMENTS}"

    # CMD_ARGUMENTS=" --version ${idx} --detector ${name} --do_dt --buffer_save_path_inf_dt /users/felix.wagner/cryoenv/tests/buffers_inf_dt_40m --path_checkpoint /scratch-cbe/users/felix.wagner/rltests/output_40m/checkpoint-90000" 
    # sbatch ${SBATCH_OPTIONS} --wrap="time singularity exec ${SINGULARITY_OPTIONS} ${CONTAINER_PATH} python3 ${PYTHON_OPTIONS} ${SCRIPT_PATH} ${CMD_ARGUMENTS}"

  done
done

exit 0
