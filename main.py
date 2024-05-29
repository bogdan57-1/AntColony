import pygame
from factory import SimulationFactory
from config import load_config_from_file


def run_experiment_with_config(args):
    screen, pheromone_manager, food_manager, colonies = SimulationFactory.create_simulation(args)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    running = True
    cycles = 0
    initTime = clock.get_time()
    while running:
        currTime = clock.get_time()
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
                          f'Current Ants: {stats["currentAnts"]}, Dead Ants: {stats["deadAnts"]}, '
                          f'Cycles To Spawn: {stats["cycles_to_spawn"]}')
            stats_surface = font.render(stats_text, True, colony.color)
            screen.blit(stats_surface, (10, 40 + i * 15))
            i += 1

        cycles_text = f'Cycles:{cycles}; ms/cycle:{currTime}'
        cycles_surface = font.render(cycles_text, True, (0, 0, 0))
        screen.blit(cycles_surface, (10, 85))

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    config_path = r'C:\Users\bogda\Desktop\AntColonyPython\config1.json'
    args = load_config_from_file(config_path)
    run_experiment_with_config(args)
