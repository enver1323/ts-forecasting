# pred_lens=(192 336 720)
pred_lens=(96)

config="ilinear"

for pred_len in ${pred_lens[@]};
do
    python train_jax.py \
        $config \
        --data.dataset.path=data/ETTh1.csv \
        --data.loader=etth \
        --model.hidden_size=192 \
        --model.seq_len=336 \
        --model.pred_len=$pred_len \
        --lr.init=1e-4 \
        --data.batch_size=32
done