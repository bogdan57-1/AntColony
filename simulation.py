import pygame
import random
import math
from collections import defaultdict
from scipy.stats import norm

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


# Helper functions
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def gridCoord(grid_x, grid_y, grid_size):
    # return the center coordinate of the grid
    return grid_x * grid_size - grid_size // 2, grid_y * grid_size - grid_size // 2


class Ant:
    def __init__(self, x, y, color, pheromone_manager, lifetime, home, food_manager, colony, args):
        self.x = x
        self.y = y
        self.color = color
        self.food = False
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = args.ant_speed
        self.pheromone_manager = pheromone_manager
        self.lifetime = random.uniform(500, lifetime)
        self.viewRange = 20
        self.pheromone_detection_threshold = args.pheromone_cull_threshold
        self.foodCarry = 0
        self.biteSize = 10  # how much food can pick up at one time
        self.home = home
        self.layPheromone = False
        self.food_manager = food_manager
        self.colony = colony
        self.args = args

        # specific brain stuff
        self.impatience = 0
        self.boredom_threshold = random.normalvariate(30, 10)  # hyperparameter

        # logging
        self.stats = {
            'distanceTravelled': 0
        }

    def move(self):
        # Basic movement
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # Boundary checking
        if self.x < 0 or self.x > self.args.screen_width or self.y < 0 or self.y > self.args.screen_height:
            self.angle += math.pi

        # log
        self.stats['distanceTravelled'] += self.speed

    def draw(self, screen):
        col = self.color
        if self.food:
            col = GREEN

        pygame.draw.circle(screen, col, (int(self.x), int(self.y)), 2)

    def detectPheromones(self):
        grid_x, grid_y = int(self.x / self.args.grid_size), int(self.y / self.args.grid_size)
        max_strength = 0
        best_tile = None

        for dx in range(-self.viewRange // self.args.grid_size, self.viewRange // self.args.grid_size + 1):
            for dy in range(-self.viewRange // self.args.grid_size, self.viewRange // self.args.grid_size + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.args.screen_width // self.args.grid_size and 0 <= ny < self.args.screen_height // self.args.grid_size:
                    world_x, world_y = nx * self.args.grid_size + self.args.grid_size // 2, ny * self.args.grid_size + self.args.grid_size // 2
                    if is_in_cone(self.x, self.y, world_x, world_y, self.angle, self.viewRange):
                        strength = self.pheromone_manager.pheromones[self.color][(nx, ny)]
                        if (strength > max_strength
                                and strength > self.pheromone_detection_threshold
                                and (nx, ny) != (grid_x, grid_y)):
                            max_strength = strength
                            best_tile = (world_x, world_y)

        return best_tile

    def randomWalk(self):
        self.speed = self.args.ant_speed

        # update internal state
        self.impatience += 1

        # maybe change direction
        if self.impatience > self.boredom_threshold:
            self.angle = random.uniform(self.angle - math.pi / 2, self.angle + math.pi / 2)
            self.impatience = 0

    def walkToTarget(self, x, y, noise=None):
        self.speed = self.args.ant_speed
        self.angle = math.atan2(y - self.y, x - self.x)
        if noise:
            self.angle += noise

    def checkIfHomeBehaviour(self):
        if distance(self.x, self.y, self.home['x'], self.home['y']) < 5:
            # deposit food
            self.colony.depositFood(self.foodCarry)
            self.foodCarry = 0
            self.food = False
            self.layPheromone = False

    def carriesFoodBehaviour(self):
        noise = random.normalvariate(0, math.pi / 2) # have a bit of noise each time unless close to home
        if random.randint(0, 10) < 3: # once in a while bias towards pheromone trail
            detected_pheromone_tile = self.detectPheromones()
            if detected_pheromone_tile:
                target_x, target_y = detected_pheromone_tile
                pheromone_angle = math.atan2(target_y - self.y, target_x - self.x)
                noise = random.normalvariate(abs(pheromone_angle-self.angle), math.pi / 8)

        if is_in_cone(self.x, self.y, self.home['x'], self.home['y'], self.angle, self.viewRange):
            self.walkToTarget(self.home['x'], self.home['y'])
        else:
            self.walkToTarget(self.home['x'], self.home['y'], noise)

        self.layPheromone = True
        self.checkIfHomeBehaviour()

    def sniffPheromonesBehaviour(self):
        detected_pheromone_tile = self.detectPheromones()
        if detected_pheromone_tile:
            target_x, target_y = detected_pheromone_tile
            self.walkToTarget(target_x, target_y)

    def brain(self):
        nearby_food = self.food_manager.check_for_food(self.x, self.y, self.angle, self.viewRange)

        if nearby_food and not self.food:
            self.walkToTarget(nearby_food.x, nearby_food.y)
            if distance(self.x, self.y, nearby_food.x, nearby_food.y) < 5:
                self.food = True
                self.foodCarry = self.biteSize
                nearby_food.pickUp(self.biteSize)
                self.angle += math.pi
        elif self.foodCarry > 0:
            self.carriesFoodBehaviour()
        elif random.randint(0, 10) <= 7:
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
    def __init__(self, x, y, color, args, pheromone_manager, food_manager):
        self.x = x
        self.y = y
        self.color = color
        self.pheromone_manager = pheromone_manager
        self.food_manager = food_manager
        self.ants = [Ant(x, y, color, pheromone_manager, args.ant_lifetime, {'x': x, 'y': y}, food_manager, self, args)
                     for _ in
                     range(args.ant_count)]
        self.food_collected = 0
        self.args = args
        self.cycles_to_spawn = args.cycles_to_spawn

        # logging
        self.stats = {
            'food': 0,
            'currentAnts': len(self.ants),
            'deadAnts': 0,
            'cycles_to_spawn': self.cycles_to_spawn
        }

    def depositFood(self, amount):
        self.stats['food'] += amount

    def spawnAnts(self, amount):
        for _ in range(amount):
            home = {'x': self.x, 'y': self.y}
            self.ants.append(
                Ant(self.x, self.y, self.color, self.pheromone_manager, self.args.ant_lifetime, home,
                    self.food_manager, self, self.args))
        self.stats['currentAnts'] = len(self.ants)

    def update(self):
        self.cycles_to_spawn -= 1
        self.stats['cycles_to_spawn'] = self.cycles_to_spawn

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
            self.stats['currentAnts'] = len(self.ants)

        if self.cycles_to_spawn <= 0:
            self.cycles_to_spawn = self.args.cycles_to_spawn
            self.spawnAnts(self.args.ants_spawn_number)

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
    def __init__(self, args):
        self.grid = defaultdict(list)
        self.args = args
        self.food_piles_count = args.food_piles_count
        self.grid_size = args.grid_size
        self._populate_grid()

    def _populate_grid(self):
        for _ in range(self.food_piles_count):
            food = Food(random.randint(50, self.args.screen_width - 50),
                        random.randint(50, self.args.screen_height - 50))
            grid_x, grid_y = int(food.x / self.grid_size), int(food.y / self.grid_size)
            self.grid[(grid_x, grid_y)].append(food)

    def _remove_from_grid(self, food):
        grid_x, grid_y = int(food.x / self.grid_size), int(food.y / self.grid_size)
        if food in self.grid[(grid_x, grid_y)]:
            self.grid[(grid_x, grid_y)].remove(food)
            if not self.grid[(grid_x, grid_y)]:
                del self.grid[(grid_x, grid_y)]

    def _get_nearby_food_piles(self, x, y, dist):
        grid_x, grid_y = int(x / self.grid_size), int(y / self.grid_size)
        nearby_food = []
        for dx in range(-dist // self.grid_size, dist // self.grid_size + 1):
            for dy in range(-dist // self.grid_size, dist // self.grid_size + 1):
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


def is_in_cone(x1, y1, x2, y2, direction, max_range, angle=math.pi / 4, min_range = 1):
    dx, dy = x2 - x1, y2 - y1
    dist = math.sqrt(dx ** 2 + dy ** 2)
    if dist > max_range or dist < min_range:
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
        if 0 <= grid_x < self.args.screen_width // self.args.grid_size and 0 <= grid_y < self.args.screen_height // self.args.grid_size:
            if self.pheromones[color][(grid_x, grid_y)] < self.args.pheromone_saturation:
                self.pheromones[color][(grid_x, grid_y)] += min(self.args.pheromone_deposit_rate,
                                                                self.args.pheromone_saturation - int(
                                                                    self.pheromones[color][(grid_x, grid_y)]))

    def process_pheromones(self):
        if self.frame_count % self.args.pheromone_process_interval != 0:
            self.frame_count += 1
            return

        self.frame_count += 1

        for color in self.pheromones:
            new_pheromones = defaultdict(float)
            self.max_values[color] = 0

            for (grid_x, grid_y), strength in self.pheromones[color].items():

                if self.max_values[color] < strength:
                    self.max_values[color] = strength

                strength *= self.args.pheromone_decay_rate
                if strength <= self.args.pheromone_cull_threshold:  # Just to cull very small values
                    continue
                new_pheromones[(grid_x, grid_y)] = strength
                # Diffuse pheromones to neighbors
                diffusion_amount = strength * self.args.pheromone_diffusion_rate / 8
                if diffusion_amount > self.args.pheromone_cull_threshold:
                    new_pheromones[(grid_x, grid_y)] -= diffusion_amount * 8
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                        nx, ny = grid_x + dx, grid_y + dy
                        if 0 <= nx < self.args.screen_width // self.args.grid_size and 0 <= ny < self.args.screen_height // self.args.grid_size:
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
                    surface = pygame.Surface((self.args.grid_size, self.args.grid_size), pygame.SRCALPHA)
                    surface.fill(pheromone_color)
                    screen.blit(surface, (grid_x * self.args.grid_size, grid_y * self.args.grid_size))

    def get_pheromone_strength(self, x, y):  # for mouse input
        grid_x, grid_y = int(x / self.args.grid_size), int(y / self.args.grid_size)
        strengths = {color: self.pheromones[color][(grid_x, grid_y)] for color in self.pheromones if
                     (grid_x, grid_y) in self.pheromones[color]}
        return strengths

"""    def get_nearest_pheromone_tile(self, color, x, y, min_strength, max_range, direction=None):
        nearest_tile = None
        nearest_distance = max_range
        grid_x, grid_y = int(x / self.args.grid_size), int(y / self.args.grid_size)

        for (px, py), strength in self.pheromones[color].items():
            if strength >= min_strength:
                dist = distance(grid_x, grid_y, px, py)
                if dist < nearest_distance:
                    if direction is None or is_in_cone(x, y, px * self.args.grid_size, py * self.args.grid_size,
                                                       direction, max_range):
                        nearest_distance = dist
                        nearest_tile = (px * self.args.grid_size + self.args.grid_size // 2,
                                        py * self.args.grid_size + self.args.grid_size // 2)

        return nearest_tile
"""