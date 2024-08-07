import pygame
from factory import SimulationFactory
from config import load_config_from_file

def run_experiment_with_config(args):
    # Set up the simulation environment
    screen, pheromone_manager, food_manager, colonies = SimulationFactory.create_simulation(args)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    running = True
    cycles = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYUP:
                running = False

        # Clear the screen with a white background
        screen.fill((255, 255, 255))

        # Process and draw pheromones
        pheromone_manager.process_pheromones()
        pheromone_manager.draw_pheromones(screen)

        # Update and draw each colony
        for colony in colonies:
            colony.update()
            colony.draw(screen)

        # Update and draw the food manager
        food_manager.update_and_draw(screen)

        # Display pheromone strength under mouse position
        mouse_x, mouse_y = pygame.mouse.get_pos()
        grid_x, grid_y = mouse_x // args.grid_size, mouse_y // args.grid_size
        pheromone_strength = pheromone_manager.get_pheromone_strength(grid_x, grid_y, 0, 'food')  # Single float value

        # Create a text surface to display pheromone strength
        strength_text = f'Pheromone Strength: {pheromone_strength:.2f}'
        text_surface = font.render(strength_text, True, (0, 0, 0))
        screen.blit(text_surface, (10, 10))

        # Display maximum pheromone values for each type
        for pheromone_type in ['home', 'food']:
            maxval_text = f'Max {pheromone_type.capitalize()} Pheromone Value: {pheromone_manager.max_values[pheromone_type][0]:.2f}'  # 0 is assumed to be the colony ID
            maxval_text_surface = font.render(maxval_text, True, (0, 0, 0))
            screen.blit(maxval_text_surface, (10, 25 + 15 * (1 if pheromone_type == 'home' else 2)))

        # Display statistics for each colony
        for i, colony in enumerate(colonies):
            stats = colony.getStats()
            stats_text = (f'Colony {colony.colony_id}: Food: {stats["food"]}, '
                          f'Current Ants: {stats["currentAnts"]}, Dead Ants: {stats["deadAnts"]}, '
                          f'Cycles To Spawn: {stats["cycles_to_spawn"]}')
            stats_surface = font.render(stats_text, True, colony.color)
            screen.blit(stats_surface, (10, 40 + i * 15))

        # Display simulation cycles and time per cycle
        currTime = clock.get_time()
        cycles_text = f'Cycles: {cycles}; ms/cycle: {currTime}'
        cycles_surface = font.render(cycles_text, True, (0, 0, 0))
        screen.blit(cycles_surface, (10, 85))

        # Refresh the screen
        pygame.display.flip()
        clock.tick(args.fps)
        cycles += 1

    # Clean up Pygame resources
    pygame.quit()

if __name__ == "__main__":
    # Load configuration from file
    config_path = r'C:\Users\bogda\Desktop\AntColonyPython\config1.json'
    args = load_config_from_file(config_path)
    run_experiment_with_config(args)
