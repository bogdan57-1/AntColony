import pygame
from factory import SimulationFactory
from config import load_config_from_file

def run_experiment_with_config(args):

    # init with factory
    screen, pheromone_manager, food_manager, colonies = SimulationFactory.create_simulation(args)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    running = True
    cycles = 0

    while running:
        for event in pygame.event.get():
            # exit events
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYUP:
                running = False

        # flush screen buffer first
        screen.fill((255, 255, 255))

        # update and draw pheromones
        pheromone_manager.process_pheromones()
        pheromone_manager.draw_pheromones(screen)

        # update and draw each colony
        for colony in colonies:
            colony.update()
            colony.draw(screen)

        # update and draw food piles
        food_manager.update_and_draw(screen)

        # debug info
        mouse_x, mouse_y = pygame.mouse.get_pos()
        grid_x, grid_y = mouse_x // args.grid_size, mouse_y // args.grid_size
        pheromone_strength_food = pheromone_manager.get_pheromone_strength(grid_x, grid_y, 0, 'food')  # Single float value
        pheromone_strength_home = pheromone_manager.get_pheromone_strength(grid_x, grid_y, 0, 'home')

        strength_text_food = f'Pheromone Strength (food): {pheromone_strength_food:.2f}'
        text_surface = font.render(strength_text_food, True, (0, 0, 0))
        screen.blit(text_surface, (10, 10))

        strength_text_home = f'Pheromone Strength (home): {pheromone_strength_home:.2f}'
        text_surface = font.render(strength_text_home, True, (0, 0, 0))
        screen.blit(text_surface, (10, 25))

        # max values for home and food pheromones for debugging
        for pheromone_type in ['home', 'food']:
            maxval_text = f'Max {pheromone_type.capitalize()} Pheromone Value: {pheromone_manager.max_values[pheromone_type][0]:.2f}'  # 0 is assumed to be the colony ID
            maxval_text_surface = font.render(maxval_text, True, (0, 0, 0))
            screen.blit(maxval_text_surface, (10, 25 + 15 * (1 if pheromone_type == 'home' else 2))) # little hack

        # colony stats
        for i, colony in enumerate(colonies):
            stats = colony.getStats()
            stats_text = (f'Colony {colony.colony_id}: Food: {stats["food"]}, '
                          f'Current Ants: {stats["currentAnts"]}, Dead Ants: {stats["deadAnts"]}, '
                          f'Cycles To Spawn: {stats["cycles_to_spawn"]}')
            stats_surface = font.render(stats_text, True, colony.color)
            screen.blit(stats_surface, (10, 70 + i * 15))

        # simulation time info
        currTime = clock.get_time()
        cycles_text = f'Cycles: {cycles}; ms/cycle: {currTime}'
        cycles_surface = font.render(cycles_text, True, (0, 0, 0))
        screen.blit(cycles_surface, (10, 85))

        # update screen and clock
        pygame.display.flip()
        clock.tick(args.fps)
        cycles += 1

    pygame.quit()

if __name__ == "__main__":
    config_path = "config1.json"
    args = load_config_from_file(config_path)
    run_experiment_with_config(args)
