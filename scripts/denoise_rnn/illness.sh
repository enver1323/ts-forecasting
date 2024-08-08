pred_lens=(24 36 48 60)
# pred_lens=(24)

config="denoise_rnn"

for pred_len in ${pred_lens[@]};
do
    python -u train.py \
        $config \
        --data.dataset.path=data/national_illness.csv \
        --data.loader=common \
        --model.n_channels=7 \
        --model.seq_len=102 \
        --model.label_len=0 \
        --model.pred_len=$pred_len \
        --model.d_model=128 \
        --model.dropout=0.5 \
        --model.patch_len=6 \
        --lr.init=0.001 \
        --lr.decay=0.8 \
        --data.batch_size=256 \
        --patience=5
done