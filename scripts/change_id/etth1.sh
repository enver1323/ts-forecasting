# pred_lens=(96 192 336 720)
pred_lens=(96)

config="change_id"

for pred_len in ${pred_lens[@]};
do
    python train_jax.py \
        $config \
        --data.dataset.path=data/ETTh1_cp.csv \
        --data.loader=change_point \
        --model.n_channels=7 \
        --model.seq_len=336 \
        --model.pred_len=96 \
        --model.patch_size=48 \
        --model.stride=16 \
        --model.hidden_size=64 \
        --lr.init=0.00001 \
        --data.batch_size=32
done