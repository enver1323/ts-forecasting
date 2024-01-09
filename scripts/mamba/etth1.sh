# pred_lens=(96 192 336 720)
pred_lens=(96)

config="mamba"

for pred_len in ${pred_lens[@]};
do
    python train_jax.py \
        $config \
        --data.dataset.path=data/ETTh1.csv \
        --data.loader=etth \
        --model.n_channels=7 \
        --model.seq_len=336 \
        --model.label_len=0 \
        --model.pred_len=96 \
        --model.n_blocks=1 \
        --model.d_model=32 \
        --model.d_inner=64 \
        --model.d_state=16 \
        --model.d_conv=4 \
        --model.d_dt=32 \
        --model.patch_size=24 \
        --lr.rec=0.001 \
        --lr.pred=0.0006 \
        --data.batch_size=32
done