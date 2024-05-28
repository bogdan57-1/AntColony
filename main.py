import pygame
import random
import math
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
GRID_SIZE = 4
ANT_COUNT = 50  # spawn count
FOOD_COUNT = 20
FPS = 30
PHEROMONE_DECAY_RATE = 0.0005  # Rate at which pheromones decay
PHEROMONE_DEPOSIT_RATE = 5  # rate at which pheromone is deposited on a cell
ANT_LIFETIME = 1000  # in number of updates
PHEROMONE_SATURATION = 50  # max pheromone amount per tile
PHEROMONE_DIFFUSION_RATE = 0.05  # fraction of pheromones that get diffused to adjacent tiles
PHEROMONE_PROCESS_INTERVAL = 5  # process pheromones every 5 frames

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

        # Lay pheromone
        self.pheromone_manager.add_pheromone(self.color, self.x, self.y)

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
        self.pheromones = {}
        self.max_values = {}  # for rendering
        self.decay_rate = decay_rate
        self.frame_count = 0  # to track frames for processing intervals

        # CUDA setup
        self.mod = SourceModule("""
        __global__ void decay_diffuse(float *pheromones, int width, int height, float decay_rate, float diffusion_rate) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int idx = y * width + x;

            if (x >= width || y >= height) return;

            float pheromone = pheromones[idx];
            if (pheromone > 0) {
                pheromones[idx] -= decay_rate;
                if (pheromones[idx] < 0) pheromones[idx] = 0;

                float diffusion_amount = pheromone * diffusion_rate / 8;
                if (x > 0) pheromones[idx - 1] += diffusion_amount;  // Left
                if (x < width - 1) pheromones[idx + 1] += diffusion_amount;  // Right
                if (y > 0) pheromones[idx - width] += diffusion_amount;  // Up
                if (y < height - 1) pheromones[idx + width] += diffusion_amount;  // Down
                if (x > 0 && y > 0) pheromones[idx - width - 1] += diffusion_amount;  // Up-Left
                if (x < width - 1 && y > 0) pheromones[idx - width + 1] += diffusion_amount;  // Up-Right
                if (x > 0 && y < height - 1) pheromones[idx + width - 1] += diffusion_amount;  // Down-Left
                if (x < width - 1 && y < height - 1) pheromones[idx + width + 1] += diffusion_amount;  // Down-Right
            }
        }
        """)

        self.decay_diffuse = self.mod.get_function("decay_diffuse")

    def create_pheromone_map(self, color):
        self.pheromones[color] = np.zeros((SCREEN_WIDTH // GRID_SIZE, SCREEN_HEIGHT // GRID_SIZE), dtype=np.float32)
        self.max_values[color] = 0

    def add_pheromone(self, color, x, y):
        grid_x, grid_y = int(x / GRID_SIZE), int(y / GRID_SIZE)
        if 0 <= grid_x < self.pheromones[color].shape[0] and 0 <= grid_y < self.pheromones[color].shape[1]:
            if self.pheromones[color][grid_x, grid_y] < PHEROMONE_SATURATION:
                self.pheromones[color][grid_x, grid_y] += PHEROMONE_DEPOSIT_RATE

            self.max_values[color] = max(self.max_values[color], self.pheromones[color][grid_x, grid_y])

    def process_pheromones(self):
        if self.frame_count % PHEROMONE_PROCESS_INTERVAL != 0:
            self.frame_count += 1
            return

        self.frame_count += 1

        for color in self.pheromones:
            pheromone_array = self.pheromones[color].flatten()
            pheromones_gpu = cuda.mem_alloc(pheromone_array.nbytes)
            cuda.memcpy_htod(pheromones_gpu, pheromone_array)

            width, height = self.pheromones[color].shape
            block_size = (16, 16, 1)
            grid_size = (width // block_size[0] + 1, height // block_size[1] + 1, 1)

            self.decay_diffuse(pheromones_gpu, np.int32(width), np.int32(height), np.float32(self.decay_rate),
                               np.float32(PHEROMONE_DIFFUSION_RATE), block=block_size, grid=grid_size)

            cuda.memcpy_dtoh(pheromone_array, pheromones_gpu)
            self.pheromones[color] = pheromone_array.reshape((width, height))

            self.max_values[color] = np.max(self.pheromones[color])

    def draw_pheromones(self, screen):
        for color, pheromone_map in self.pheromones.items():
            for (grid_x, grid_y), strength in np.ndenumerate(pheromone_map):
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
        Colony(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50), RED, ANT_COUNT,
               pheromone_manager),
        # Uncomment to add more colonies
        # Colony(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50), BLUE, ANT_COUNT, pheromone_manager),
        # Colony(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50), BLACK, ANT_COUNT, pheromone_manager)
    ]

    for colony in colonies:
        pheromone_manager.create_pheromone_map(colony.color)

    food_piles = [Food(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50)) for _ in
                  range(FOOD_COUNT)]

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
