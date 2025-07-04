export CUDA_VISIBLE_DEVICES=0
python ./src/evaluate.py \
    --eval_strategy inversion \
    --save_dir results/exp_dit_gigant_ad2_enet/all \
    --eval_step 5 \