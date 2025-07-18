export CUDA_VISIBLE_DEVICES=0

# python ./src/evaluate.py \
#     --eval_strategy inversion \
#     --save_dir results/exp_dit_hugew_visa_enet/all \
#     --eval_step 2 \

python main.py \
    --task test \
    --devices cuda:0 \
    --eval_strategy inversion \
    --save_dir results/exp_dit_gigant_realiad_enet384/all \
    --eval_step 3 \

# python ./src/evaluate.py \
#     --eval_strategy inversion \
#     --save_dir results/exp_dit_hugew_visa_enet/all \
#     --eval_step 4 \

# python ./src/evaluate.py \
#     --eval_strategy inversion \
#     --save_dir results/exp_dit_hugew_visa_enet/all \
#     --eval_step 5 \

