export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python ./src/evaluate_distributed.py \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --eval_strategy inversion \
    --save_dir results/exp_unet_realiad/all \
    --eval_step 4 \