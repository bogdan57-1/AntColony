import pygame
import random
import math
from collections import defaultdict
from config import SimulationConfig
from factory import SimulationFactory
from config import load_config_from_file

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
GRID_SIZE = 7
ANT_COUNT = 50  # spawn count

FPS = 30
ANT_LIFETIME = 3000  # in number of updates
PHEROMONE_DEPOSIT_RATE = 100  # rate at which pheromone is deposited on a cell
PHEROMONE_SATURATION = 100  # max pheromone amount per tile

PHEROMONE_PROCESS_INTERVAL = 5  # process pheromones every 5 frames
PHEROMONE_DIFFUSION_RATE = 0.2  # fraction of pheromones that get diffused to adjacent tiles
PHEROMONE_DECAY_RATE = 0.95  # Rate at which pheromones decay

PHEROMONE_CULL_THRESHOLD = PHEROMONE_DEPOSIT_RATE / 100  # below which tiles will be deleted

FOOD_PILES_COUNT = 50

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


# Helper functions
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def gridCoord(grid_x, grid_y):
    # return the center coordinate of the grid
    return grid_x * GRID_SIZE - GRID_SIZE // 2, grid_y * GRID_SIZE - GRID_SIZE // 2


class Ant:
    def __init__(self, x, y, color, pheromone_manager, lifetime, home, food_manager, colony):
        self.x = x
        self.y = y
        self.color = color
        self.food = False
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 2
        self.pheromone_manager = pheromone_manager
        self.lifetime = random.uniform(500, lifetime)
        self.viewRange = 50
        self.pheromone_detection_threshold = PHEROMONE_CULL_THRESHOLD
        self.foodCarry = 0
        self.biteSize = 10  # how much food can pick up at one time
        self.home = home
        self.layPheromone = False
        self.food_manager = food_manager
        self.colony = colony

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

    def detectPheromones(self):
        grid_x, grid_y = int(self.x / GRID_SIZE), int(self.y / GRID_SIZE)
        max_strength = 0
        best_tile = None

        for dx in range(-self.viewRange // GRID_SIZE, self.viewRange // GRID_SIZE + 1):
            for dy in range(-self.viewRange // GRID_SIZE, self.viewRange // GRID_SIZE + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < SCREEN_WIDTH // GRID_SIZE and 0 <= ny < SCREEN_HEIGHT // GRID_SIZE:
                    world_x, world_y = nx * GRID_SIZE + GRID_SIZE // 2, ny * GRID_SIZE + GRID_SIZE // 2
                    if is_in_cone(self.x, self.y, world_x, world_y, self.angle, self.viewRange):
                        strength = self.pheromone_manager.pheromones[self.color][(nx, ny)]
                        if (strength > max_strength
                                and strength > self.pheromone_detection_threshold
                                and (nx, ny) != (grid_x, grid_y)):
                            max_strength = strength
                            best_tile = (world_x, world_y)

        return best_tile

    def randomWalk(self):
        self.speed = 2

        # update internal state
        self.impatience += 1

        # maybe change direction
        if self.impatience > self.boredom_threshold:
            self.angle = random.uniform(self.angle - math.pi / 2, self.angle + math.pi / 2)
            self.impatience = 0

    def walkToTarget(self, x, y):
        self.angle = math.atan2(y - self.y, x - self.x)

    def checkIfHomeBehaviour(self):
        if distance(self.x, self.y, self.home['x'], self.home['y']) < 5:
            # deposit food
            self.colony.depositFood(self.foodCarry)
            self.foodCarry = 0
            self.food = False
            self.layPheromone = False

    def carriesFoodBehaviour(self):
        self.walkToTarget(self.home['x'], self.home['y'])
        self.layPheromone = True
        self.checkIfHomeBehaviour()

    def sniffPheromonesBehaviour(self):
        detected_pheromone_tile = self.detectPheromones()
        if detected_pheromone_tile:
            target_x, target_y = detected_pheromone_tile
            self.walkToTarget(target_x, target_y)

    def brain(self):
        detected_pheromone_tile = self.detectPheromones()
        nearby_food = self.food_manager.check_for_food(self.x, self.y, self.angle, self.viewRange)

        if nearby_food and not self.food:
            self.walkToTarget(nearby_food.x, nearby_food.y)
            if distance(self.x, self.y, nearby_food.x, nearby_food.y) < 5:
                self.food = True
                self.foodCarry = self.biteSize
                nearby_food.pickUp(self.biteSize)
        elif self.foodCarry > 0:
            self.carriesFoodBehaviour()
        elif random.randint(0, 10) <= 8:
            self.sniffPheromonesBehaviour()
        else:
            self.randomWalk()

    def update(self):
        self.brain()
        self.move()

        if self.layPheromone:
            self.pheromone_manager.add_pheromone(self.color, self.x, self.y)

        self.lose_health()

    def lose_health(self):
        self.lifetime -= 1


class Colony:
    def __init__(self, x, y, color, ant_count, pheromone_manager, food_manager):
        self.x = x
        self.y = y
        self.color = color
        self.ants = [Ant(x, y, color, pheromone_manager, ANT_LIFETIME, {'x': x, 'y': y}, food_manager, self) for _ in
                     range(ant_count)]
        self.food_collected = 0

        # logging
        self.stats = {
            'food': 0,
            'currentAnts': len(self.ants),
            'deadAnts': 0,
            'cyclesToSpawn': 0
        }

    def depositFood(self, amount):
        self.stats['food'] += amount

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
            self.stats['deadAnts'] += 1

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 10)
        for ant in self.ants:
            ant.draw(screen)

    def getStats(self):
        return self.stats


class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.amount = 100

    def draw(self, screen):
        pygame.draw.circle(screen, GREEN, (int(self.x), int(self.y)), 5)

    def pickUp(self, amount):
        if self.amount > 0:
            self.amount -= min(amount, self.amount)


class FoodManager:
    def __init__(self):
        self.grid = defaultdict(list)
        self._populate_grid()

    def _populate_grid(self):
        for _ in range(FOOD_PILES_COUNT):
            food = Food(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50))
            grid_x, grid_y = int(food.x / GRID_SIZE), int(food.y / GRID_SIZE)
            self.grid[(grid_x, grid_y)].append(food)

    def _remove_from_grid(self, food):
        grid_x, grid_y = int(food.x / GRID_SIZE), int(food.y / GRID_SIZE)
        if food in self.grid[(grid_x, grid_y)]:
            self.grid[(grid_x, grid_y)].remove(food)
            if not self.grid[(grid_x, grid_y)]:
                del self.grid[(grid_x, grid_y)]

    def _get_nearby_food_piles(self, x, y, dist):
        grid_x, grid_y = int(x / GRID_SIZE), int(y / GRID_SIZE)
        nearby_food = []
        for dx in range(-dist // GRID_SIZE, dist // GRID_SIZE + 1):
            for dy in range(-dist // GRID_SIZE, dist // GRID_SIZE + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if (nx, ny) in self.grid:
                    nearby_food.extend(self.grid[(nx, ny)])
        return nearby_food

    def check_for_food(self, x, y, direction, view_range):
        nearby_food_piles = self._get_nearby_food_piles(x, y, view_range)
        for food in nearby_food_piles:
            if is_in_cone(x, y, food.x, food.y, direction, view_range):
                return food
        return None

    def update_and_draw(self, screen):
        empty_piles = []
        for (grid_x, grid_y), piles in self.grid.items():
            for pile in piles:
                if pile.amount <= 0:
                    empty_piles.append(pile)
                else:
                    pile.draw(screen)

        for pile in empty_piles:
            self._remove_from_grid(pile)


def is_in_cone(x1, y1, x2, y2, direction, max_range, angle=math.pi / 4):
    dx, dy = x2 - x1, y2 - y1
    dist = math.sqrt(dx ** 2 + dy ** 2)
    if dist > max_range:
        return False
    angle_to_target = math.atan2(dy, dx)
    delta_angle = abs(direction - angle_to_target)
    if delta_angle > math.pi:
        delta_angle = 2 * math.pi - delta_angle
    return delta_angle < angle


class PheromoneManager:
    def __init__(self, args):
        self.pheromones = defaultdict(lambda: defaultdict(float))
        self.args = args
        self.max_values = defaultdict(float)  # for rendering
        self.decay_rate = args.pheromone_decay_rate
        self.frame_count = 0  # to track frames for processing intervals

    def create_pheromone_map(self, color):
        self.max_values[color] = 1

    def add_pheromone(self, color, x, y):
        grid_x, grid_y = int(x / self.args.grid_size), int(y / self.args.grid_size)
        if 0 <= grid_x < SCREEN_WIDTH // GRID_SIZE and 0 <= grid_y < SCREEN_HEIGHT // GRID_SIZE:
            if self.pheromones[color][(grid_x, grid_y)] < PHEROMONE_SATURATION:
                self.pheromones[color][(grid_x, grid_y)] += min(PHEROMONE_DEPOSIT_RATE,
                                                                PHEROMONE_SATURATION - int(
                                                                    self.pheromones[color][(grid_x, grid_y)]))

    def process_pheromones(self):
        if self.frame_count % PHEROMONE_PROCESS_INTERVAL != 0:
            self.frame_count += 1
            return

        self.frame_count += 1

        for color in self.pheromones:
            new_pheromones = defaultdict(float)
            self.max_values[color] = 0

            for (grid_x, grid_y), strength in self.pheromones[color].items():

                if self.max_values[color] < strength:
                    self.max_values[color] = strength

                strength *= PHEROMONE_DECAY_RATE
                if strength <= PHEROMONE_CULL_THRESHOLD:  # Just to cull very small values
                    continue
                new_pheromones[(grid_x, grid_y)] = strength
                # Diffuse pheromones to neighbors
                diffusion_amount = strength * PHEROMONE_DIFFUSION_RATE * (strength / PHEROMONE_SATURATION) / 8
                if diffusion_amount > PHEROMONE_CULL_THRESHOLD:
                    # print(diffusion_amount)
                    new_pheromones[(grid_x, grid_y)] -= diffusion_amount * 8
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
                    alpha = min(255,
                                int(strength / (self.max_values[color] + 1) * 255))  # Normalize strength to [0, 255]
                    pheromone_color = (*color, alpha)
                    surface = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
                    surface.fill(pheromone_color)
                    screen.blit(surface, (grid_x * GRID_SIZE, grid_y * GRID_SIZE))

    def get_pheromone_strength(self, x, y):  # for mouse input
        grid_x, grid_y = int(x / GRID_SIZE), int(y / GRID_SIZE)
        strengths = {color: self.pheromones[color][(grid_x, grid_y)] for color in self.pheromones if
                     (grid_x, grid_y) in self.pheromones[color]}
        return strengths

    def get_nearest_pheromone_tile(self, color, x, y, min_strength, max_range, direction=None):
        nearest_tile = None
        nearest_distance = max_range
        grid_x, grid_y = int(x / GRID_SIZE), int(y / GRID_SIZE)

        for (px, py), strength in self.pheromones[color].items():
            if strength >= min_strength:
                dist = distance(grid_x, grid_y, px, py)
                if dist < nearest_distance:
                    if direction is None or is_in_cone(x, y, px * GRID_SIZE, py * GRID_SIZE, direction, max_range):
                        nearest_distance = dist
                        nearest_tile = (px * GRID_SIZE + GRID_SIZE // 2, py * GRID_SIZE + GRID_SIZE // 2)

        return nearest_tile


def run_experiment_with_config(config):
    screen, pheromone_manager, food_manager, colonies = SimulationFactory.create_simulation(config)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    running = True
    cycles = 0
    while running:
        cycles += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYUP:
                running = False

        screen.fill((255, 255, 255))

        pheromone_manager.process_pheromones()
        pheromone_manager.draw_pheromones(screen)

        for colony in colonies:
            colony.update()
            colony.draw(screen)

        food_manager.update_and_draw(screen)

        # Get mouse position and corresponding grid position
        mouse_x, mouse_y = pygame.mouse.get_pos()
        pheromone_strengths = pheromone_manager.get_pheromone_strength(mouse_x, mouse_y)

        # Display pheromone strengths on the screen
        strength_text = ', '.join([f'{color}: {strength:.2f}' for color, strength in pheromone_strengths.items()])
        text_surface = font.render(f'Pheromone Strengths: {strength_text}', True, (0, 0, 0))
        screen.blit(text_surface, (10, 10))

        # show max pheromone value
        maxval_text = f'{pheromone_manager.max_values[(255, 0, 0)]:0.2f}'
        mamxval_text_surface = font.render(f'Max value: {maxval_text}', True, (0, 0, 0))
        screen.blit(mamxval_text_surface, (10, 25))

        # get colony stats and print on screen
        i = 0
        for colony in colonies:
            stats = colony.getStats()
            stats_text = (f'Colony {colony.color}: Food: {stats["food"]}, '
                          f'Current Ants: {stats["currentAnts"]}, Dead Ants: {stats["deadAnts"]}')
            stats_surface = font.render(stats_text, True, colony.color)
            screen.blit(stats_surface, (10, 40 + i * 15))
            i += 1

        cycles_text = f'Cycles:{cycles}'
        cycles_surface = font.render(cycles_text, True, (0, 0, 0))
        screen.blit(cycles_surface, (10, 85))

        pygame.display.flip()
        clock.tick(config.fps)

    pygame.quit()


if __name__ == "__main__":
    config_path = 'path_to_your_config_file.json'
    config = load_config_from_file(config_path)
    run_experiment_with_config(config)
