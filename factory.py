
import random
from collections import defaultdict
import pygame
from simulation import PheromoneManager, FoodManager, Colony

class SimulationFactory:
    @staticmethod
    def create_simulation(config):
        pygame.init()
        screen = pygame.display.set_mode((config.screen_width, config.screen_height))
        pheromone_manager = PheromoneManager(config)
        food_manager = FoodManager(config)
        colonies = [
            Colony(random.randint(50, config.screen_width - 50), random.randint(50, config.screen_height - 50), (255, 0, 0),
                   config, pheromone_manager, food_manager),
            #Colony(random.randint(50, config.screen_width - 50), random.randint(50, config.screen_height - 50), (0, 0, 255),
             #      config, pheromone_manager, food_manager),
            #Colony(random.randint(50, config.screen_width - 50), random.randint(50, config.screen_height - 50), (0, 0, 0),
             #      config, pheromone_manager, food_manager)
        ]
        return screen, pheromone_manager, food_manager, colonies
