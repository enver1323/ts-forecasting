# pred_lens=(96 192 336 720)
pred_lens=(96)

config="rnn_dec"

for pred_len in ${pred_lens[@]};
do
    python -u train.py \
        $config \
        --data.dataset.path=data/ETTh1.csv \
        --data.loader=etth \
        --model.n_channels=7 \
        --model.seq_len=336 \
        --model.label_len=0 \
        --model.pred_len=$pred_len \
        --model.d_model=512 \
        --model.dropout=0.5 \
        --model.patch_len=24 \
        --lr.init=0.001 \
        --lr.decay=1.0 \
        --data.batch_size=64 \
        --patience=10
done