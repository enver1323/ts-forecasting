pred_lens=(96 192 336 720)
# pred_lens=(96)

config="mdlinear"

for pred_len in ${pred_lens[@]};
do
    python -u train_jax.py \
        $config \
        --data.dataset.path=data/ETTm2.csv \
        --data.loader=ettm \
        --model.n_channels=7 \
        --model.seq_len=336 \
        --model.label_len=0 \
        --model.pred_len=$pred_len \
        --model.patch_size=8 \
        --lr.init=0.00003 \
        --data.batch_size=16 \
        --patience=5
done