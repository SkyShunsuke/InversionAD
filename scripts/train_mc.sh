python ./main.py \
    --task train_dist \
    --fname ./configs/exp_dit_mpdd/all.yml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --port 12345 