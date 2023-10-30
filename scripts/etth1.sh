pred_lens=(192 336 720)
# pred_lens=(96)

config="point_id_ar"

for pred_len in ${pred_lens[@]};
do
    python train.py \
        $config \
        --wandb_log=0 \
        --data.dataset.path=data/ETTh1.csv \
        --data.loader=etth \
        --model.n_choices=4 \
        --model.n_channels=7 \
        --model.seq_len=96 \
        --model.pred_len=$pred_len \
        --data.batch_size=32
done