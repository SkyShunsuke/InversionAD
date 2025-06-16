python ./src/train_distributed.py \
    --config_path ./configs/exp_unet_ad_dinov2/all_1.yml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --port 12345 

python ./src/train_distributed.py \
    --config_path ./configs/exp_unet_ad_dinov2/all_2.yml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --port 12345 

python ./src/train_distributed.py \
    --config_path ./configs/exp_unet_ad_dinov2/all_3.yml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --port 12345 

python ./src/train_distributed.py \
    --config_path ./configs/exp_unet_ad_dinov2/all_4.yml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --port 12345 