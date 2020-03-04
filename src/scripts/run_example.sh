#! /bin/bash

y=136 # [14, 136, 306, 417, 672, 717, 852, 914, 933, 992]

for transform in scale vertical_position horizontal_position
do
    python3 main.py \
        --output_dir=../outputs/example \
        --image_size=128 \
        --y=$y \
        --transform=$transform \
        --num_trajectories=50 \
        --batch_size=10 \
        --n_steps=50 \
        --renorm \
        --save_images \
        --evaluation=quantitative \
        --num_traversals=50
done