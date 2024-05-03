import tomllib


config = None


def load_config():
    with open('config.toml', 'rb') as f:
        global config
        config = tomllib.load(f)
