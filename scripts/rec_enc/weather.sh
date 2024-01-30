# pred_lens=(96 192 336 720)
pred_lens=(96)

config="rec_enc"

for pred_len in ${pred_lens[@]};
do
    python -u train.py \
        $config \
        --data.dataset.path=data/weather.csv \
        --data.loader=common \
        --model.n_channels=21 \
        --model.seq_len=720 \
        --model.label_len=0 \
        --model.pred_len=$pred_len \
        --model.d_model=512 \
        --model.dropout=0.5 \
        --model.patch_len=48 \
        --lr.init=0.0001 \
        --lr.decay=0.8 \
        --data.batch_size=64 \
        --patience=5
done