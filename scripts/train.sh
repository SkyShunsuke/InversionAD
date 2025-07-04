# !/bin/bash

# クラスのリスト
AD_CLASSES=(
  "metal_nut" "tile" "screw" "zipper" "grid"
  "pill" "capsule" "transistor" "toothbrush"
  "cable" "carpet" "wood" "bottle" "leather" "hazelnut"
)

# 空きとみなすGPUメモリ使用量の上限（MB）
MAX_USED_MEM=1000

# GPU数
NUM_GPUS=8

# 間隔（秒）
CHECK_INTERVAL=60

# 処理を順番にこなす
for CLASS in "${AD_CLASSES[@]}"; do
  while true; do
    for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
      # 現在のGPUメモリ使用量を取得（MiB単位）
      USED_MEM=$(nvidia-smi --id=$GPU_ID --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)

      # 閾値より少ないなら使用可能とみなす
      if [ "$USED_MEM" -lt "$MAX_USED_MEM" ]; then
        echo "[`date`] Launching training for $CLASS on GPU $GPU_ID (used: ${USED_MEM}MiB)"

        # バックグラウンドで実行
        CUDA_VISIBLE_DEVICES=$GPU_ID python ./src/train.py --config_path ./configs/exp_dit_ad/${CLASS}.yml &

        # 次のクラスへ
        sleep 60
        break 2
      fi
    done

    # どのGPUも空いてなければ少し待つ
    sleep $CHECK_INTERVAL
  done
done

# 最後に全プロセスの終了を待つ
wait
echo "All training jobs completed."
