# pred_lens=(96 192 336 720)
pred_lens=(96)

config="ssm"

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
        --model.n_blocks=1 \
        --model.d_model=32 \
        --model.d_inner=32 \
        --model.d_state=32 \
        --model.d_conv=4 \
        --model.d_dt=64 \
        --model.patch_size=48 \
        --lr.rec=0.001 \
        --lr.pred=0.001 \
        --lr.n_warmup_epochs=3 \
        --lr.decay=0.8 \
        --patience=5 \
        --data.batch_size=64
done