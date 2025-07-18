# python ./main.py \
#     --task train_odm \
#     --fname ./configs/exp_dit_mpdd_odm/connector.yml \
#     --save_dir ./results/exp_dit_gigant_mpdd_enet_aug1/all \
#     --devices cuda:0 \
#     --port 12341 & \

# python ./main.py \
#     --task train_odm \
#     --fname ./configs/exp_dit_mpdd_odm/bracket_white.yml \
#     --save_dir ./results/exp_dit_gigant_mpdd_enet_aug1/all \
#     --devices cuda:1 \
#     --port 12342 & \

# python ./main.py \
#     --task train_odm \
#     --fname ./configs/exp_dit_mpdd_odm/bracket_brown.yml \
#     --save_dir ./results/exp_dit_gigant_mpdd_enet_aug1/all \
#     --devices cuda:2 \
#     --port 12343 & \

python ./main.py \
    --task train_odm \
    --fname ./configs/exp_dit_mpdd_odm/bracket_black.yml \
    --save_dir ./results/exp_dit_gigant_mpdd_enet_aug1/all \
    --devices cuda:3 \
    --port 12344 \

# python ./main.py \
#     --task train_odm \
#     --fname ./configs/exp_dit_mpdd_odm/tubes.yml \
#     --save_dir ./results/exp_dit_gigant_mpdd_enet_aug1/all \
#     --devices cuda:4 \
#     --port 12345 & \

# python ./main.py \
#     --task train_odm \
#     --fname ./configs/exp_dit_mpdd_odm/metal_plate.yml \
#     --save_dir ./results/exp_dit_gigant_mpdd_enet_aug1/all \
#     --devices cuda:5 \
#     --port 12346 & \
# wait
