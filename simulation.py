import pygame
import random
import math
from collections import defaultdict
import numpy as np

# colours for colonies moved to config file, but these are useful as well
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


# helper functions
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# returns true if a particular object_2 is within object_1's cone field of view.
# requires the direction and max range of object_1.
# the angle parameter represents the shape of the cone. defaults to 45 degrees.
# the scan will be performed as min_range<=obj_2<=max_range.
def is_in_cone(x1, y1, x2, y2, direction, max_range, angle=math.pi / 4, min_range=1):
    dx, dy = x2 - x1, y2 - y1
    dist = math.sqrt(dx ** 2 + dy ** 2)
    if dist > max_range or dist < min_range:
        return False
    angle_to_target = math.atan2(dy, dx)
    delta_angle = abs(direction - angle_to_target)
    if delta_angle > math.pi:
        delta_angle = 2 * math.pi - delta_angle
    return delta_angle < angle


# grid to world transformation
def gridCoord(grid_x, grid_y, grid_size):
    # return the center coordinate of the grid
    return grid_x * grid_size - grid_size // 2, grid_y * grid_size - grid_size // 2


# bias using exponential modelling
def bias_probabilities_exp(pheromones, alpha=2, inverse=False):
    pheromones = np.array(pheromones)

    # handle edge case where there is only 1 value, or only equal values
    if len(np.unique(pheromones)) == 1:
        return np.full(len(pheromones), 1.0 / len(pheromones))

    normalized_pheromones = (pheromones - np.min(pheromones)) / (np.max(pheromones) - np.min(pheromones))

    if inverse:
        transformed_pheromones = np.exp(-alpha * normalized_pheromones)
    else:
        transformed_pheromones = np.exp(alpha * normalized_pheromones)

    probabilities = transformed_pheromones / np.sum(transformed_pheromones)
    return probabilities


# bias using power transform
def bias_probabilities_power(pheromones, alpha=2, inverse=False):
    pheromones = np.array(pheromones)

    beta = np.average(pheromones)

    if inverse:
        transformed_pheromones = (2 * beta - pheromones) ** alpha
    else:
        transformed_pheromones = pheromones ** alpha

    probabilities = transformed_pheromones / np.sum(transformed_pheromones)
    return probabilities


# takes a list of tiles and probabilistically samples one or more depending on the tiles' pheromone strength.
# have not used it with multiple samples.
# supports two different sampling modes: 'power' and 'exp'.
# inverse=True will bias towards smaller strengths.
def sampleTile(tiles, samples=1, inverse=False, mode='exp'):
    if not tiles:
        return []

    strengths = np.array([tile[2] for tile in tiles])

    if not np.any(strengths > 0):
        return []  # no pheromone tiles nearby

    if mode == 'exp':
        probabilities = bias_probabilities_exp(strengths, alpha=3, inverse=inverse)
    elif mode == 'power':
        probabilities = bias_probabilities_power(strengths, alpha=3, inverse=inverse)

    sampled_indices = np.random.choice(len(tiles), size=samples, p=probabilities)  # error if mode set incorrectly
    sampled_tiles = [tiles[i] for i in sampled_indices]  # samples = 1 for now

    return sampled_tiles


class Ant:
    def __init__(self, x, y, colony_id, pheromone_manager, lifetime, home, food_manager, colony, args):
        self.x = x
        self.y = y
        self.colony_id = colony_id
        self.food = False
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = args.ant_speed
        self.pheromone_manager = pheromone_manager
        self.lifetime = random.uniform(500, lifetime)
        self.viewRange = args.grid_size * 5
        self.pheromone_detection_threshold = args.pheromone_cull_threshold
        self.foodCarry = 0
        self.biteSize = 10  # how much food can be picked up at one time
        self.home = home
        self.layPheromone = False
        self.food_manager = food_manager
        self.colony = colony
        self.args = args

        # specific brain stuff
        self.impatience = 0
        self.boredom_threshold = random.normalvariate(50, 10)  # hyperparameter

        # logging
        self.stats = {
            'distanceTravelled': 0
        }

    # called on update to move the ant to a new coordinate
    def move(self):
        # basic movement
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # check for boundaries - careful, it can leave the screen, it just turns back
        if self.x < 0 or self.x > self.args.screen_width or self.y < 0 or self.y > self.args.screen_height:
            self.angle += math.pi

        # log
        self.stats['distanceTravelled'] += self.speed

    # draw the ant. use green colour if it carries food.
    def draw(self, screen):
        color = self.colony.color
        if self.food:
            color = GREEN

        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 2)

    # scans around the ant and returns the nearby pheromone tiles (for both types)
    def detectPheromones(self):
        grid_x, grid_y = int(self.x / self.args.grid_size), int(self.y / self.args.grid_size)
        pheromone_tiles = {
            'food': [],
            'home': []
        }

        grid_range = self.viewRange // self.args.grid_size  # how many grids away to look
        grid_count_w = self.args.screen_width // self.args.grid_size  # how many grids there are; width
        grid_count_h = self.args.screen_height // self.args.grid_size  # height

        # check tile neighbours in range for pheromones
        # could also change to np.roll later if there's performance issues
        for dx in range(-grid_range, grid_range + 1):
            for dy in range(-grid_range, grid_range + 1):
                cx, cy = grid_x + dx, grid_y + dy  # current relevant grid
                if 0 <= cx < grid_count_w and 0 <= cy < grid_count_h:
                    world_x, world_y = gridCoord(cx, cy, self.args.grid_size)  # grid to world
                    # home pheromones
                    strength_home = self.pheromone_manager.get_pheromone_strength(cx, cy, self.colony_id, 'home')
                    if (cx, cy) != (grid_x, grid_y) and strength_home > 0:
                        pheromone_tiles['home'].append((world_x, world_y, strength_home))
                    # food pheromones
                    strength_food = self.pheromone_manager.get_pheromone_strength(cx, cy, self.colony_id, 'food')
                    if (cx, cy) != (grid_x, grid_y) and strength_food > 0:
                        pheromone_tiles['food'].append((world_x, world_y, strength_food))

        return pheromone_tiles

    # default wandering behaviour
    # the ant moves with a preset speed around, and changes direction when it gets 'bored'
    # function is used as part of food searching behaviour
    def randomWalk(self):
        self.speed = self.args.ant_speed

        # update internal state
        self.impatience += 1

        # maybe change direction
        if self.impatience > self.boredom_threshold:
            self.angle = random.uniform(self.angle - math.pi / 2, self.angle + math.pi / 2)
            self.impatience = 0

    # used to direct the ant towards a particular coordinate by adjusting its angle of movement
    # also supports the addition of a noise parameter to affect the angle, however this is ill designed and deprecated
    # as it did not achieve the desired outcome
    def walkToTarget(self, x, y, noise=None):
        self.speed = self.args.ant_speed
        self.angle = math.atan2(y - self.y, x - self.x)
        if noise:
            self.angle += noise

    # behaviour for determining whether the ant has reached the colony
    # if true, then it will deposit the food and switch its pheromone lay state
    def checkIfHomeBehaviour(self):
        if distance(self.x, self.y, self.home['x'], self.home['y']) < 5:
            # deposit food
            self.colony.depositFood(self.foodCarry)
            self.foodCarry = 0
            self.food = False
            self.layPheromone = 'home'

    # if the ant is carrying food it should be aiming to return home.
    # on return, the ant lays 'food' pheromones which will guide other ants towards the food source in the future.
    # the principle here is to bias the movement towards a 'home' trail that is leaving the colony, hence
    # sampling with the inverse mode which favours lower strength tiles - however, this may not have the desired effect
    # once multiple ants are picking up on a trail.
    # the ant will also check its viewing cone - if it 'sees' the colony it will aim towards it.
    def carriesFoodBehaviour(self, detected_pheromones):
        self.layPheromone = 'food'
        current_pos = np.array([self.x, self.y])
        new_direction = np.zeros(2)

        # to be filled
        home_str = 0
        home_vec = np.zeros(2)
        home_pheromone_tile = None
        ###

        # follow 'home' trail
        if detected_pheromones['home']:
            home_pheromone_tile = sampleTile(detected_pheromones['home'], inverse=True)[0]  # maybe empty
        if home_pheromone_tile:
            home_ph_pos = np.array([home_pheromone_tile[0], home_pheromone_tile[1]])
            home_str = home_pheromone_tile[2]
            home_vec = home_ph_pos - current_pos

        if home_vec.any():
            dist = np.linalg.norm(home_vec)
            if dist > 0:
                home_vec /= dist
            new_direction += home_vec # * home_str # is this even what i want?

        if np.any(new_direction):  # should be all zeros if nothing happened
            self.angle = np.arctan2(new_direction[1], new_direction[0])

        # reach home if nearby
        if is_in_cone(self.x, self.y, self.home['x'], self.home['y'], self.angle, self.viewRange):
            self.walkToTarget(self.home['x'], self.home['y'])

        self.checkIfHomeBehaviour()

    # deprecated function
    def sniffPheromonesBehaviour(self):
        detected_tiles = self.detectPheromones()['food']  # returns all tiles around with strength > 0
        if detected_tiles:
            # probabilistically samples pheromone tiles
            target_x, target_y, _ = sampleTile(detected_tiles)[0]
            self.walkToTarget(target_x, target_y)

    # main wandering behaviour.
    # the idea here is to bias the ant towards food trails (sample in inverse?) and away from home trails.
    # this is needs heavy adjusting as obviously a path to a food source will have both. maybe split this into different
    # behaviours which alternate depending on conditions.
    def searchFoodBehaviour(self, detected_pheromones):
        self.layPheromone = 'home'
        current_pos = np.array([self.x, self.y])
        new_direction = np.zeros(2)

        # to be filled
        food_str = 0
        to_food_vec = np.zeros(2)
        home_str = 0
        away_home_vec = np.zeros(2)

        food_pheromone_tile = None
        home_pheromone_tile = None
        #####

        if detected_pheromones['food']:
            food_pheromone_tile = sampleTile(detected_pheromones['food'])[0]  # maybe empty
        if food_pheromone_tile:
            food_pos = np.array([food_pheromone_tile[0], food_pheromone_tile[1]])
            food_str = food_pheromone_tile[2]
            to_food_vec = food_pos - current_pos

        # see if there's any home pheromone tiles in the way
        """
        home_pheromone_tiles = []
        for tile in detected_pheromones['home']:
            if is_in_cone(self.x, self.y, tile[0], tile[1], self.angle, self.viewRange,
                          min_range=self.args.grid_size * 4):
                home_pheromone_tiles.append(tile)
        # if there's any home pheromone tiles
        if home_pheromone_tiles:
            home_pheromone_tile = sampleTile(home_pheromone_tiles)[0]
        if home_pheromone_tile:
            home_ph_pos = np.array([home_pheromone_tile[0], home_pheromone_tile[1]])
            home_str = home_pheromone_tile[2]
            away_home_vec = current_pos - home_ph_pos
        """
        # normalise vectors and add to direction
        # idea was to get both vectors and add them with a weight corresponding the strength of the pheromone
        if to_food_vec.any():
            dist = np.linalg.norm(to_food_vec)
            if dist > 0:
                to_food_vec /= dist
            new_direction += to_food_vec # * food_str
        """    
        if away_home_vec.any():
            dist = np.linalg.norm(away_home_vec)
            if dist > 0:
                away_home_vec /= dist
            new_direction += away_home_vec * home_str
        """
        # bias towards the food (and away from home - didn't work correctly, needs adjusting, no time left)
        if np.any(new_direction):  # should be all zeros if nothing happened
            self.angle = np.arctan2(new_direction[1], new_direction[0])
        else:
            self.randomWalk()

    # behaviour to pick up any detected food within its viewing range
    def pickUpFoodBehaviour(self, nearby_food):
        self.walkToTarget(nearby_food.x, nearby_food.y)
        if distance(self.x, self.y, nearby_food.x, nearby_food.y) < 5:
            self.food = True
            self.foodCarry = self.biteSize
            nearby_food.pickUp(self.biteSize)
            self.angle += math.pi

    # the logic center of the ant. It checks for nearby food and pheromones and decides which behaviour to adopt.
    def brain(self):
        nearby_food = self.food_manager.check_for_food(self.x, self.y, self.angle, self.viewRange)
        detected_pheromones = self.detectPheromones()

        if nearby_food and not self.food:  # pick up food
            self.pickUpFoodBehaviour(nearby_food)
        elif self.foodCarry > 0:  # carry food
            self.carriesFoodBehaviour(detected_pheromones)
        else:  # search for food
            self.searchFoodBehaviour(detected_pheromones)

    # ant update function. responsibility passed to colony class
    def update(self):
        self.brain()  # think
        self.move()  # and move

        if self.layPheromone:  # lay pheromone depending on state
            self.pheromone_manager.add_pheromone(self.colony_id, self.x, self.y, self.layPheromone)

        self.lose_health()  # ants will lose health over time

    def lose_health(self):
        self.lifetime -= 1


class Colony:
    def __init__(self, x, y, colony_id, color, args, pheromone_manager, food_manager):
        self.x = x
        self.y = y
        self.colony_id = colony_id
        self.color = color
        self.pheromone_manager = pheromone_manager
        self.food_manager = food_manager
        self.ants = [
            Ant(x, y, colony_id, pheromone_manager, args.ant_lifetime, {'x': x, 'y': y}, food_manager, self, args)
            for _ in range(args.ant_count)]
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

    # used by ants to signal that food has been deposited into the colony
    def depositFood(self, amount):
        self.food_collected += amount
        self.stats['food'] += amount

    # since ants can die, there is a need to be able to respawn some over time
    def spawnAnts(self, amount):
        for _ in range(amount):
            home = {'x': self.x, 'y': self.y}
            self.ants.append(
                Ant(self.x, self.y, self.colony_id, self.pheromone_manager, self.args.ant_lifetime, home,
                    self.food_manager, self, self.args))
        self.stats['currentAnts'] = len(self.ants)

    # colony is updated by the main script.
    # updates the ants, removes any 'dead' ones, and managers respawn mechanic.
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

    # food piles can be picked up incrementally, depending on the ant's bite size.
    def pickUp(self, amount):
        if self.amount > 0:
            self.amount -= min(amount, self.amount)


# class to manage the food piles
class FoodManager:
    def __init__(self, args):
        self.grid = defaultdict(list)  # at a particular set of coordinates there can be multiple food piles stacked
        self.args = args
        self.food_piles_count = args.food_piles_count
        self.grid_size = args.grid_size  # the food also spawns on grid tiles to make it easier to manage
        self._populate_grid()

    # spawn the food
    def _populate_grid(self):
        for _ in range(self.food_piles_count):
            food = Food(random.randint(50, self.args.screen_width - 50),
                        random.randint(50, self.args.screen_height - 50))
            grid_x, grid_y = int(food.x / self.grid_size), int(food.y / self.grid_size)
            self.grid[(grid_x, grid_y)].append(food)

    # remove a particular food object from the grid
    def _remove_from_grid(self, food):
        grid_x, grid_y = int(food.x / self.grid_size), int(food.y / self.grid_size)
        if food in self.grid[(grid_x, grid_y)]:
            self.grid[(grid_x, grid_y)].remove(food)
            if not self.grid[(grid_x, grid_y)]:  # if the list is empty, remove the dictionary entry to save space
                del self.grid[(grid_x, grid_y)]

    # helper function to get food piles within a distance (on the grid)
    def _get_nearby_food_piles(self, x, y, dist):
        grid_x, grid_y = int(x / self.grid_size), int(y / self.grid_size)
        nearby_food = []
        for dx in range(-dist // self.grid_size, dist // self.grid_size + 1):
            for dy in range(-dist // self.grid_size, dist // self.grid_size + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if (nx, ny) in self.grid:
                    nearby_food.extend(self.grid[(nx, ny)])
        return nearby_food

    # used by the ant to check for nearby food.
    # it abstracts the underlying food grid by only requesting the ant's position, direction and view range
    def check_for_food(self, x, y, direction, view_range):
        nearby_food_piles = self._get_nearby_food_piles(x, y, view_range)
        for food in nearby_food_piles:
            if is_in_cone(x, y, food.x, food.y, direction, view_range):
                return food
        return None

    # update function for piles. also draws them on the screen.
    # called by the main function. there was no need to have two separate update and draw functions since they are
    # called consecutively
    def update_and_draw(self, screen):
        empty_piles = []
        for (grid_x, grid_y), piles in self.grid.items():
            for pile in piles:
                if pile.amount <= 0:
                    empty_piles.append(pile)  # can't change the list while iterating
                else:
                    pile.draw(screen)

        for pile in empty_piles:
            self._remove_from_grid(pile)


# class to manage the pheromone tiles
class PheromoneManager:
    def __init__(self, args):
        self.args = args
        self.decay_rate = args.pheromone_decay_rate
        self.frame_count = 0  # to track frames for processing intervals

        self.grid_w = args.screen_width // args.grid_size
        self.grid_h = args.screen_height // args.grid_size

        # switched to numpy arrays for performance
        self.pheromones = {
            'home': np.zeros((args.num_colonies, self.grid_w, self.grid_h)),
            'food': np.zeros((args.num_colonies, self.grid_w, self.grid_h))
        }

        # also max values for debugging, switched to np for consistency
        self.max_values = {
            'home': np.zeros(args.num_colonies),
            'food': np.zeros(args.num_colonies)
        }

    # used by ant to signal a release of pheromones
    def add_pheromone(self, colony_id, x, y, pheromone_type):
        grid_x, grid_y = int(x / self.args.grid_size), int(y / self.args.grid_size)

        # bounds checking
        if (not (0 <= grid_x < self.grid_w)) or (not (0 <= grid_y < self.grid_h)):
            return

        if self.pheromones[pheromone_type][colony_id, grid_x, grid_y] < self.args.pheromone_saturation:  # hard cap
            self.pheromones[pheromone_type][colony_id, grid_x, grid_y] += min(
                self.args.pheromone_deposit_rate,
                self.args.pheromone_saturation - int(self.pheromones[pheromone_type][colony_id, grid_x, grid_y])
            )  # add value or difference up to saturation

    # called by main function to process pheromones
    # will only do so once every few frames to improve performance
    def process_pheromones(self):
        # process every few frames for performance
        if self.frame_count % self.args.pheromone_process_interval != 0:
            self.frame_count += 1
            return

        self.frame_count += 1

        # main loop, pheromones are processed for all colonies inside the function
        for pheromone_type in ['home', 'food']:
            self.diffuse_and_decay_pheromones(pheromone_type)

    # diffuse and decay the whole grid (all colonies) for a given pheromone type
    def diffuse_and_decay_pheromones(self, pheromone_type):

        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        with open('pheromone_debug.log', 'a') as log_file:  # used to debug, no longer required

            # decay with simple numpy operation for all colonies
            self.pheromones[pheromone_type] *= self.decay_rate

            # diffuse per colony
            for colony_id in range(self.args.num_colonies):  # For each colony
                pheromone_layer = self.pheromones[pheromone_type][colony_id]

                # diffusion accumulator
                diffused = np.zeros_like(pheromone_layer)
                neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]

                # debug
                debug = False
                if np.any(pheromone_layer > 1):
                    debug = False

                if debug:
                    log_file.write("BEGIN PHEROMONE UPDATE\n")
                    log_file.write(f"Pheromone layer before diffusion:\n{pheromone_layer[pheromone_layer > 0]}\n")

                # for each neighbour mask with rule and calculate diffusion amount
                for dx, dy in neighbor_offsets:
                    # shift the array to check neighbours
                    # note that shifts with dx actually shift on the pixel Y axis and vice versa
                    shifted = np.roll(pheromone_layer, shift=dx, axis=0)
                    shifted = np.roll(shifted, shift=dy, axis=1)

                    # mask to only diffuse to lesser strength neighbours and to stop diffusing once the current cell is
                    # below a certain cull threshold
                    diffusion_mask = (shifted < pheromone_layer) & (
                            pheromone_layer >= self.args.pheromone_cull_threshold)

                    # amount calculated assuming the spread would be to all 8 neighbours,
                    # however this is mostly not the case, but the limit is easier to calculate
                    diffusion_amount = pheromone_layer * self.args.pheromone_diffusion_rate / 8

                    # tricky part
                    # the mask will mark the i,j cell in pheromone_layer under the conditions
                    # therefore to spread to the actual neighbour we need to unroll with -dx,-dy
                    diffused += np.roll(diffusion_amount * diffusion_mask, shift=(-dx, -dy), axis=(0, 1))
                    # don't forget to subtract the diffused amount from the actual cell
                    # here we don't need to shift because the mask already targets cell i,j
                    pheromone_layer -= diffusion_amount * diffusion_mask

                if debug:
                    log_file.write(f"Diffused grid:\n{diffused[diffused > 0]}\n")

                # add the diffused values
                self.pheromones[pheromone_type][colony_id] += diffused

                if debug:
                    log_file.write(
                        f"Updated pheromone layer:\n{self.pheromones[pheromone_type][colony_id][self.pheromones[pheromone_type][colony_id] > 0]}\n")
                    log_file.write("END OF PHEROMONE UPDATE\n")
                debug = False

            # also update max for debugging
            self.max_values[pheromone_type][colony_id] = np.max(self.pheromones[pheromone_type][colony_id])

    # draws pheromone tiles on the screen.
    # the food pheromones are drawn using the colony colour, while the home pheromones use the inverted colour
    def draw_pheromones(self, screen):
        grid_size = self.args.grid_size
        for pheromone_type in ['home', 'food']:
            for colony_id in range(self.args.num_colonies):
                pheromone_layer = self.pheromones[pheromone_type][colony_id]
                for (grid_x, grid_y), strength in np.ndenumerate(pheromone_layer):  # unpack coordinate and value
                    if strength > 0:
                        color = self.args.colony_colors[colony_id]  # Use the defined color for each colony
                        alpha = min(255, int(strength * 255 / self.args.pheromone_saturation))
                        if pheromone_type == 'food':
                            pheromone_color = (*color, alpha)
                        else: # type == 'home'
                            negate_color = (255 - color[0], 255 - color[1], 255 - color[2])
                            pheromone_color = (*negate_color, alpha)
                        surface = pygame.Surface((grid_size, grid_size), pygame.SRCALPHA)
                        surface.fill(pheromone_color)
                        screen.blit(surface, (grid_x * grid_size, grid_y * grid_size))

    def get_pheromone_strength(self, grid_x, grid_y, colony_id, pheromone_type):
        return self.pheromones[pheromone_type][colony_id, grid_x, grid_y]
