from train_jax import app as train_app

CONFIG_NAME = 'mamba'
CONFIG = {
    'data.dataset.path': 'data/ETTh1_cp.csv',
    'data.loader': 'change_point',
    'model.n_channels': 7,
    'model.seq_len': 336,
    'model.label_len': 0,
    'model.pred_len': 96,
    'model.n_blocks': 1,
    'model.d_model': 32,
    'model.d_inner': 64,
    'model.d_state': 16,
    'model.d_conv': 4,
    'model.d_dt': 32,
    'model.patch_size': 24,
    'lr.rec': 0.001,
    'lr.pred': 0.001,
    'data.batch_size': 32
}
GRID = {
    'model.seq_len': [96, 192, 336, 720],
    'model.pred_len': [96, 336],
    'model.d_model': [32, 64],
    'model.d_inner': [64, 128],
    'model.d_state': [16, 32],
    'model.d_conv': [4, 8],
    'model.d_dt': [32, 64],
    'model.patch_size': [16, 24, 48],
    'lr.rec': [3e-2, 1e-3, 3e-4],
    'lr.pred': [3e-2, 1e-3, 3e-4],
}

best_config = None
best_stat = None


def format_train_config(config):
    config = [f"--{k}={v}" for k, v in config.items()]
    return [CONFIG_NAME] + config


def train_step(cur_conf, next_conf):
    global best_config
    global best_stat

    if len(next_conf) == 0:
        print(f"Current config: {cur_conf}")
        train_config = format_train_config(cur_conf)
        try:
            trainer = train_app(train_config)
            cur_stat = trainer.log.get_stat('test', 'test/pred_loss')
            if best_config is None or best_stat is None or cur_stat < best_stat:
                best_config = cur_conf
                best_stat = cur_stat
                print(f"Best config: {best_config}")
                print(f"Best stat: {best_stat}")
        except Exception as e:
            print(e)
        return

    cur_k = list(next_conf.keys())[0]
    for v in next_conf[cur_k]:
        temp_conf = {**cur_conf, cur_k: v}
        filtered_next_conf = {k: v for k, v in next_conf.items() if k != cur_k}
        train_step(temp_conf, filtered_next_conf)


def write_best_config():
    global best_config
    global best_stat
    with open('best_config.txt', 'w') as f:
        f.write(f"Best config: {best_config}\n")
        f.write(f"Best stat: {best_stat}\n")


def app():
    train_step(CONFIG, GRID)
    print("Best config: ", best_config)
    print("Best stat: ", best_stat)
    write_best_config()


if __name__ == '__main__':
    app()
