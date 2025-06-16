export CUDA_VISIBLE_DEVICES=2
python ./src/evaluate.py \
    --eval_strategy reconstruction \
    --save_dir results/exp_unet_ad/all \
    --eval_step 10 \
    --noise_step 7 \