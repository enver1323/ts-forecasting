# pred_lens=(96 192 336 720)
pred_lens=(96)

config=point_id

for pred_len in ${pred_lens[@]};
do
    python train.py \
        $config \
        --wandb_log=0 \
        --data.dataset.path=data/ETTh1.csv \
        --model.n_channels=7 \
        --model.seq_len=336 \
        --model.pred_len=$pred_len \
        --model.n_points=4 \
        --data.batch_size=32
done