from train_jax import app as train_app

CONFIG_NAME = 'ssm'
CONFIG = {
    'data.dataset.path': 'data/national_illness.csv',
    'data.loader': 'common',
    'model.n_channels': 7,
    'model.seq_len': 104,
    'model.label_len': 0,
    'model.pred_len': 24,
    'model.n_blocks': 1,
    'model.d_model': 32,
    'model.d_inner': 32,
    'model.d_state': 32,
    'model.d_conv': 4,
    'model.d_dt': 64,
    'model.patch_size': 4,
    'lr.rec': 0.001,
    'lr.pred': 0.0006,
    'data.batch_size': 16
}
GRID = {
    # 'model.seq_len': [192, 336, 504, 720],
    # 'model.pred_len': [96, 192, 336, 720],
    'model.d_model': [16, 32],
    'model.d_inner': [32, 64],
    'model.d_state': [16, 32, 64],
    'model.d_conv': [2, 4],
    'model.d_dt': [16, 32, 64],
    'model.patch_size': [4, 8],
    # 'model.n_blocks': [1,2,3,4],
    'lr.rec': [1e-3, 6e-4, 3e-4],
    'lr.pred': [1e-3, 6e-4, 3e-4],
}

best_config = {}
best_stat = {}


def write_best_config():
    global best_config
    global best_stat
    with open('best_config.txt', 'a') as f:
        f.write(f"Best config: {best_config}\n")
        f.write(f"Best stat: {best_stat}\n")
        f.write("***************************************************************\n")


def format_train_config(config):
    config = [f"--{k}={v}" for k, v in config.items()]
    return [CONFIG_NAME] + config


def train_step(cur_conf, next_conf):
    global best_config
    global best_stat

    if len(next_conf) == 0:
        print(f"Current config: {cur_conf}")
        cur_pred_len = cur_conf['model.pred_len']
        train_config = format_train_config(cur_conf)
        try:
            trainer = train_app(train_config)
            cur_stat = trainer.log.get_stat('test', 'test/pred_loss')
            cur_best_config = best_config.get(cur_pred_len, None)
            cur_best_stat = best_stat.get(cur_pred_len, None)
            if cur_best_config is None or cur_best_stat is None or cur_stat < cur_best_stat:
                best_config[cur_pred_len] = cur_conf
                best_stat[cur_pred_len] = cur_stat
                print(f"Best config: {best_config}")
                print(f"Best stat: {best_stat}")
                write_best_config()
        except Exception as e:
            print(e)
        return

    cur_k = list(next_conf.keys())[0]
    for v in next_conf[cur_k]:
        temp_conf = {**cur_conf, cur_k: v}
        filtered_next_conf = {k: v for k, v in next_conf.items() if k != cur_k}
        train_step(temp_conf, filtered_next_conf)


def app():
    train_step(CONFIG, GRID)
    print("Best config: ", best_config)
    print("Best stat: ", best_stat)
    write_best_config()


if __name__ == '__main__':
    app()
