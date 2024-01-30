pred_lens=(24 36 48 60)
# pred_lens=(96)

config="mdlinear"

for pred_len in ${pred_lens[@]};
do
    python -u train_jax.py \
        $config \
        --data.dataset.path=data/national_illness.csv \
        --data.loader=common \
        --model.n_channels=7 \
        --model.seq_len=104 \
        --model.label_len=0 \
        --model.pred_len=$pred_len \
        --model.patch_size=8 \
        --lr.init=0.0003 \
        --data.batch_size=16 \
        --patience=5
done