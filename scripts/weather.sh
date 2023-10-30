# pred_lens=(96 192 336 720)
pred_lens=(96 )

# config="point_id"

# for pred_len in ${pred_lens[@]};
# do
#     python train.py \
#         $config \
#         --wandb_log=1 \
#         --data.dataset.path=data/weather.csv \
#         --model.n_channels=21 \
#         --model.seq_len=336 \
#         --model.pred_len=$pred_len \
#         --model.n_points=12 \
#         --data.batch_size=32
# done


config="point_id_ar"

for pred_len in ${pred_lens[@]};
do
    python train.py \
        $config \
        --wandb_log=0 \
        --data.dataset.path=data/weather.csv \
        --model.n_choices=4 \
        --model.n_channels=21 \
        --model.seq_len=336 \
        --model.pred_len=$pred_len \
        --data.batch_size=32
done