import json


class SimulationConfig:
    def __init__(self, screen_width, screen_height, grid_size, fps, ant_count, ant_lifetime, pheromone_deposit_rate,
                 pheromone_saturation, pheromone_process_interval, pheromone_diffusion_rate, pheromone_decay_rate,
                 pheromone_cull_threshold, food_piles_count):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_size = grid_size
        self.fps = fps
        self.ant_count = ant_count
        self.ant_lifetime = ant_lifetime
        self.pheromone_deposit_rate = pheromone_deposit_rate
        self.pheromone_saturation = pheromone_saturation
        self.pheromone_process_interval = pheromone_process_interval
        self.pheromone_diffusion_rate = pheromone_diffusion_rate
        self.pheromone_decay_rate = pheromone_decay_rate
        self.pheromone_cull_threshold = pheromone_cull_threshold
        self.food_piles_count = food_piles_count

# utils
def load_config_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return SimulationConfig(**data)
