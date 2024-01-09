# pred_lens=(96 192 336 720)
pred_lens=(96)

config=lin_adapt

for pred_len in ${pred_lens[@]};
do
    python train.py \
        $config \
        --data.dataset.path=data/ETTh1.csv \
        --data.loader=etth \
        --model.n_channels=7 \
        --model.seq_len=336 \
        --model.pred_len=$pred_len \
        --lr.init=0.005 \
        --data.batch_size=32
done