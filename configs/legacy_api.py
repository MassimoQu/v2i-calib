import os
import datetime
from pathlib import Path

import yaml
from easydict import EasyDict

cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()


def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)
        merge_new_config(config=config, new_config=new_config)
    return config


class Logger:
    def __init__(self, name):
        path = os.path.join(cfg.ROOT_DIR, 'outputs', 'logs')
        if not os.path.exists(path):
            os.makedirs(path)
        log_path = os.path.join(
            path, f"{name}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
        )
        self.log_file = open(log_path, "w")

    def info(self, msg):
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()
