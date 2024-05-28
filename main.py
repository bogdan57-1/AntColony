import pygame
import random
import math
from collections import defaultdict

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
GRID_SIZE = 4
ANT_COUNT = 50  # spawn count
FOOD_COUNT = 20
FPS = 30
ANT_LIFETIME = 1000  # in number of updates
PHEROMONE_DEPOSIT_RATE = 1  # rate at which pheromone is deposited on a cell
PHEROMONE_SATURATION = 10  # max pheromone amount per tile

PHEROMONE_PROCESS_INTERVAL = 5  # process pheromones every 5 frames
PHEROMONE_DIFFUSION_RATE = 0.0  # fraction of pheromones that get diffused to adjacent tiles
PHEROMONE_DECAY_RATE = 0.95 # Rate at which pheromones decay

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Helper functions
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

class Ant:
    def __init__(self, x, y, color, pheromone_manager, lifetime):
        self.x = x
        self.y = y
        self.color = color
        self.food = False
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 2
        self.pheromone_manager = pheromone_manager
        self.lifetime = lifetime

        # specific brain stuff
        self.impatience = 0
        self.boredom_threshold = random.normalvariate(30, 10)  # hyperparameter

    def move(self):
        # Basic movement
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # Boundary checking
        if self.x < 0 or self.x > SCREEN_WIDTH or self.y < 0 or self.y > SCREEN_HEIGHT:
            self.angle += math.pi

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 2)

    def brain(self):
        # update internal state
        self.impatience += 1

        # maybe change direction
        if self.impatience > self.boredom_threshold:
            self.angle = random.uniform(self.angle - math.pi / 2, self.angle + math.pi / 2)
            self.impatience = 0

    def update(self):
        self.brain()
        self.move()
        self.pheromone_manager.add_pheromone(self.color, self.x, self.y)
        self.lose_health()

    def lose_health(self):
        self.lifetime -= 1

class Colony:
    def __init__(self, x, y, color, ant_count, pheromone_manager):
        self.x = x
        self.y = y
        self.color = color
        self.ants = [Ant(x, y, color, pheromone_manager, ANT_LIFETIME) for _ in range(ant_count)]
        self.food_collected = 0

    def update(self):
        ants_to_remove = []
        for ant in self.ants:
            ant.update()

            # check for dead ants to remove
            if ant.lifetime <= 0:
                ants_to_remove.append(ant)

        for ant in ants_to_remove:
            self.ants.remove(ant)
            del ant  # smart pointer stuff

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 10)
        for ant in self.ants:
            ant.draw(screen)

class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.amount = 100

    def draw(self, screen):
        pygame.draw.circle(screen, GREEN, (int(self.x), int(self.y)), 5)

class PheromoneManager:
    def __init__(self, decay_rate):
        self.pheromones = defaultdict(lambda: defaultdict(float))
        self.max_values = defaultdict(float)  # for rendering
        self.decay_rate = decay_rate
        self.frame_count = 0  # to track frames for processing intervals

    def create_pheromone_map(self, color):
        self.max_values[color] = 0

    def add_pheromone(self, color, x, y):
        grid_x, grid_y = int(x / GRID_SIZE), int(y / GRID_SIZE)
        if 0 <= grid_x < SCREEN_WIDTH // GRID_SIZE and 0 <= grid_y < SCREEN_HEIGHT // GRID_SIZE:
            if self.pheromones[color][(grid_x, grid_y)] < PHEROMONE_SATURATION:
                self.pheromones[color][(grid_x, grid_y)] += min(PHEROMONE_DEPOSIT_RATE,
                                                                PHEROMONE_SATURATION-int(self.pheromones[color][(grid_x, grid_y)]))

            self.max_values[color] = max(self.max_values[color], self.pheromones[color][(grid_x, grid_y)])

    def process_pheromones(self):
        if self.frame_count % PHEROMONE_PROCESS_INTERVAL != 0:
            self.frame_count += 1
            return

        self.frame_count += 1

        for color in self.pheromones:
            new_pheromones = defaultdict(float)
            for (grid_x, grid_y), strength in self.pheromones[color].items():
                strength *= PHEROMONE_DECAY_RATE
                if strength <= PHEROMONE_DEPOSIT_RATE / 100:  # Just to cull very small values
                    continue
                new_pheromones[(grid_x, grid_y)] = strength
                # Diffuse pheromones to neighbors
                diffusion_amount = strength * PHEROMONE_DIFFUSION_RATE / 8
                if diffusion_amount > PHEROMONE_DEPOSIT_RATE / 100:
                    #print(diffusion_amount)
                    new_pheromones[(grid_x, grid_y)] -= diffusion_amount*8
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                        nx, ny = grid_x + dx, grid_y + dy
                        if 0 <= nx < SCREEN_WIDTH // GRID_SIZE and 0 <= ny < SCREEN_HEIGHT // GRID_SIZE:
                            new_pheromones[(nx, ny)] += diffusion_amount

            self.pheromones[color] = new_pheromones
            print(len(self.pheromones[color]))

    def draw_pheromones(self, screen):
        for color, pheromone_map in self.pheromones.items():
            for (grid_x, grid_y), strength in pheromone_map.items():
                if strength > 0:
                    alpha = min(255, int(strength / PHEROMONE_SATURATION * 255))  # Normalize strength to [0, 255]
                    pheromone_color = (*color, alpha)
                    surface = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
                    surface.fill(pheromone_color)
                    screen.blit(surface, (grid_x * GRID_SIZE, grid_y * GRID_SIZE))

def run_experiment():
    # pygame init
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Ant Colony Simulation")
    clock = pygame.time.Clock()

    # Pheromone manager
    pheromone_manager = PheromoneManager(PHEROMONE_DECAY_RATE)

    # colonies and food init
    colonies = [
        Colony(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50), RED, ANT_COUNT, pheromone_manager),
        # Uncomment to add more colonies
        # Colony(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50), BLUE, ANT_COUNT, pheromone_manager),
        # Colony(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50), BLACK, ANT_COUNT, pheromone_manager)
    ]

    for colony in colonies:
        pheromone_manager.create_pheromone_map(colony.color)

    food_piles = [Food(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50)) for _ in range(FOOD_COUNT)]

    # pygame stuff
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYUP:
                running = False

        screen.fill(WHITE)

        pheromone_manager.process_pheromones()
        pheromone_manager.draw_pheromones(screen)

        for colony in colonies:
            colony.update()
            colony.draw(screen)

        for food in food_piles:
            food.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

# Perform the experiments
run_experiment()
