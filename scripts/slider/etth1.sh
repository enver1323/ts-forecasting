# pred_lens=(96 192 336 720)
pred_lens=(96)

config="slider"

for pred_len in ${pred_lens[@]};
do
    python -u train_jax.py \
        $config \
        --data.dataset.path=data/ETTh1.csv \
        --data.loader=etth \
        --model.n_channels=7 \
        --model.seq_len=336 \
        --model.label_len=0 \
        --model.pred_len=$pred_len \
        --lr.init=0.0004 \
        --data.batch_size=16
done