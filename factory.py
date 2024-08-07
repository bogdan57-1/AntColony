import random
import pygame
from simulation import PheromoneManager, FoodManager, Colony


class SimulationFactory:
    @staticmethod
    def create_simulation(config):
        pygame.init()
        screen = pygame.display.set_mode((config.screen_width, config.screen_height))

        # Initialize pheromone manager and food manager
        pheromone_manager = PheromoneManager(config)
        food_manager = FoodManager(config)

        # Initialize colonies with unique IDs and colors
        colonies = []
        for colony_id in range(config.num_colonies):
            x = random.randint(50, config.screen_width - 50)
            y = random.randint(50, config.screen_height - 50)
            color = config.colony_colors[colony_id]  # Use predefined colors for each colony
            colony = Colony(x, y, colony_id, color, config, pheromone_manager, food_manager)
            colonies.append(colony)

        return screen, pheromone_manager, food_manager, colonies
