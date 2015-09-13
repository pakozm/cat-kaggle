#!/bin/bash

data_seeds=(458736 2117 28651)
shuffle_seeds=(48776 5773 12041)
weights_seeds=(244896 4217 60661)
noise_seeds=(68754 1230 27554)

train()
{
    str=$(echo $@ | sed 's/ /_/g')
    OMP_NUM_THREADS=4 april-ann scripts/TRAIN/train_stack_mlp_gbm.lua $@ | tee STACK_MLPS/train_$str.log
}

for j in 0 1 2; do
    train 128 512 256 ${data_seeds[$j]} ${shuffle_seeds[$j]} ${weights_seeds[$j]} ${noise_seeds[$j]}
    train 256 512 256 ${data_seeds[$j]} ${shuffle_seeds[$j]} ${weights_seeds[$j]} ${noise_seeds[$j]}
    train 128 128 256 ${data_seeds[$j]} ${shuffle_seeds[$j]} ${weights_seeds[$j]} ${noise_seeds[$j]}
    train 128  64 256 ${data_seeds[$j]} ${shuffle_seeds[$j]} ${weights_seeds[$j]} ${noise_seeds[$j]}
    train 256 256 128 ${data_seeds[$j]} ${shuffle_seeds[$j]} ${weights_seeds[$j]} ${noise_seeds[$j]}
done
