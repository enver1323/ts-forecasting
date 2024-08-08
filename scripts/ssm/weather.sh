pred_lens=(96 192 336 720)
# pred_lens=(720)

config="ssm"

for pred_len in ${pred_lens[@]};
do
    python -u train_jax.py \
        $config \
        --data.dataset.path=data/weather.csv \
        --data.loader=common \
        --model.n_channels=21 \
        --model.seq_len=336 \
        --model.label_len=0 \
        --model.pred_len=$pred_len \
        --model.n_blocks=1 \
        --model.d_model=32 \
        --model.d_inner=32 \
        --model.d_state=16 \
        --model.d_conv=2 \
        --model.d_dt=16 \
        --model.patch_size=24 \
        --lr.rec=0.001 \
        --lr.pred=0.0006 \
        --data.batch_size=8
done