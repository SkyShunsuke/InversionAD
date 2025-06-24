export CUDA_VISIBLE_DEVICES=0
python ./src/evaluate.py \
    --eval_strategy inversion \
    --save_dir results/exp_unet_realiad/all \
    --eval_step 4 \