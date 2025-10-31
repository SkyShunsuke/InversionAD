export CUDA_VISIBLE_DEVICES=0
python ./src/evaluate.py \
    --eval_strategy inversion \
    --save_dir results/exp_dit_ad/all \  # REPLACE WITH YOUR SAVE DIRECTORY, e.g., results/exp_dit_ad/all \
    --category audiojack \
    --eval_step 3 \