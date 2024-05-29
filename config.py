import json


class SimulationConfig:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

# utils
def load_config_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return SimulationConfig(data)