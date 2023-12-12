#!/bin/sh
# execution script for SLURM cluster 

# for double tes: put to time=600

for name in li1p li1l li2p
do
  for (( idx=5; idx<=9; idx++))
  do
    echo "Submitting job ${name} version ${idx}"
    JOB_NAME="SAC_${name}_${idx}"
    SCRIPT_PATH="/users/felix.wagner/cryoenv/tests/train_sac.py"
    CONTAINER_PATH="/users/felix.wagner/mosquitto_latest.sif"
    OUTPUT_FILE="/users/felix.wagner/outputs/${JOB_NAME}.out"
    SBATCH_OPTIONS=" -c 1 --mem=4G --output=${OUTPUT_FILE} --job-name=${JOB_NAME} --time=480"
    SINGULARITY_OPTIONS=" -c -B /eos/ -H /users/felix.wagner"
    PYTHON_OPTIONS=" -u "
    
    if [ $idx -lt 5 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.1 --lr 3e-4  --batch_size 16  --gamma 0.99  --gradient_steps 20 " 
    elif [ $idx -lt 10 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.1 --lr 3e-4  --batch_size 16  --gamma 0.99  --gradient_steps 20 --double_tes --sweep " 
    elif [ $idx -lt 15 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 1e-3  --batch_size 16  --gamma 0.99  --gradient_steps 20 " 
    elif [ $idx -lt 20 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 3e-4  --batch_size 64  --gamma 0.99  --gradient_steps 20 " 
    elif [ $idx -lt 25 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 3e-4  --batch_size 16  --gamma 0.99  --gradient_steps 100 " 
    elif [ $idx -lt 30 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 3e-4  --batch_size 16  --gamma 0.9  --gradient_steps 20 " 
    elif [ $idx -lt 35 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 1e-3  --batch_size 16  --gamma 0.6  --gradient_steps 100 " 
    elif [ $idx -lt 40 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 1e-3  --batch_size 64  --gamma 0.99  --gradient_steps 20 " 
    elif [ $idx -lt 45 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 3e-4  --batch_size 16  --gamma 0.99  --gradient_steps 20 --sweep " 
    elif [ $idx -lt 50 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 1e-3  --batch_size 16  --gamma 0.99  --gradient_steps 20 --sweep " 
    elif [ $idx -lt 55 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 3e-4  --batch_size 64  --gamma 0.99  --gradient_steps 20 --sweep " 
    elif [ $idx -lt 60 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 3e-4  --batch_size 16  --gamma 0.99  --gradient_steps 100 --sweep " 
    elif [ $idx -lt 65 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 3e-4  --batch_size 16  --gamma 0.9  --gradient_steps 20 --sweep " 
    elif [ $idx -lt 70 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 1e-3  --batch_size 16  --gamma 0.6  --gradient_steps 100 --sweep " 
    elif [ $idx -lt 75 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 1e-3  --batch_size 64  --gamma 0.99  --gradient_steps 20 --sweep " 
    elif [ $idx -lt 80 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 3e-4  --batch_size 16  --gamma 0.99  --gradient_steps 20  --tau 20.  --sweep " 
    elif [ $idx -lt 85 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 1e-3  --batch_size 16  --gamma 0.99  --gradient_steps 20  --tau 20.  --sweep " 
    elif [ $idx -lt 90 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 3e-4  --batch_size 64  --gamma 0.99  --gradient_steps 20  --tau 20.  --sweep " 
    elif [ $idx -lt 95 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 3e-4  --batch_size 16  --gamma 0.99  --gradient_steps 100  --tau 20.  --sweep " 
    elif [ $idx -lt 100 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 3e-4  --batch_size 16  --gamma 0.9  --gradient_steps 20  --tau 20.  --sweep " 
    elif [ $idx -lt 105 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 1e-3  --batch_size 16  --gamma 0.6  --gradient_steps 100  --tau 20.  --sweep " 
    elif [ $idx -lt 110 ] 
    then
        CMD_ARGUMENTS=" --version ${idx} --detector ${name} --rnd_seed ${idx} --scale 0.2 --lr 1e-3  --batch_size 64  --gamma 0.99  --gradient_steps 20  --tau 20.  --sweep " 
    fi
    
    sbatch ${SBATCH_OPTIONS} --wrap="time singularity exec ${SINGULARITY_OPTIONS} ${CONTAINER_PATH} python3 ${PYTHON_OPTIONS} ${SCRIPT_PATH} ${CMD_ARGUMENTS}"
    
  done
done

exit 0
