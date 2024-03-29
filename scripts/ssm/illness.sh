pred_lens=(24 36 48 60)
# pred_lens=(96)

config="ssm"

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
        --model.n_blocks=1 \
        --model.d_model=16 \
        --model.d_inner=32 \
        --model.d_state=32 \
        --model.d_conv=4 \
        --model.d_dt=64 \
        --model.patch_size=12 \
        --lr.rec=0.01 \
        --lr.pred=0.01 \
        --data.batch_size=32
done