#!/bin/bash

data_seeds=(53968 2117 28591)
shuffle_seeds=(4852869 28773 14041)
weights_seeds=(5968 4417 6081)
noise_seeds=(72659 12290 20274)

train()
{
    str=$(echo $@ | sed 's/ /_/g')
    OMP_NUM_THREADS=2 april-ann scripts/TRAIN/train_mlp_ram.lua $@ | tee RAM_MLPS/train_$str.log
}

train 128 4 128 ${data_seeds[0]} ${shuffle_seeds[0]} ${weights_seeds[0]} ${noise_seeds[0]}

for j in 1 2; do
    train 128 64 256 ${data_seeds[$j]} ${shuffle_seeds[$j]} ${weights_seeds[$j]} ${noise_seeds[$j]}
    train 128 64 4 ${data_seeds[$j]} ${shuffle_seeds[$j]} ${weights_seeds[$j]} ${noise_seeds[$j]}
    train 512 64 256 ${data_seeds[$j]} ${shuffle_seeds[$j]} ${weights_seeds[$j]} ${noise_seeds[$j]}
    train 128 8 256 ${data_seeds[$j]} ${shuffle_seeds[$j]} ${weights_seeds[$j]} ${noise_seeds[$j]}
    train 128 4 128 ${data_seeds[$j]} ${shuffle_seeds[$j]} ${weights_seeds[$j]} ${noise_seeds[$j]}
    train 64 256 32 ${data_seeds[$j]} ${shuffle_seeds[$j]} ${weights_seeds[$j]} ${noise_seeds[$j]}
    train 32 64 256 ${data_seeds[$j]} ${shuffle_seeds[$j]} ${weights_seeds[$j]} ${noise_seeds[$j]}
    train 8 128 4 ${data_seeds[$j]} ${shuffle_seeds[$j]} ${weights_seeds[$j]} ${noise_seeds[$j]}
done
