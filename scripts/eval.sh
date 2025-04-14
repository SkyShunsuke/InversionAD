export CUDA_VISIBLE_DEVICES=0
python ./src/evaluate.py \
    --save_dir ./results/exp_unet_ad/bottle \
    --eval_step 4 