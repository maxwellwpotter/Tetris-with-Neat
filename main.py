import numpy as np

from tetris import Tetris
import neat

BOARD_WIDTH = 10
BOARD_HEIGHT = 20
GENERATIONS = 1000

# Possible Inputs:
# Current Piece, next piece, bumpiness

def fitness(network):
    env = Tetris()
    done = False
    while not done:
        next_states = env.get_next_states()
        flattenedBoard = np.array(env._get_complete_board()).flatten()
        outputs = network.computeOutput(flattenedBoard)
        outputs = [t for t in enumerate(outputs)]
        outputs.sort(key=lambda input : input[1])

        i = 0
        best_state = None
        while best_state is None:
            best_index = outputs[i][0]
            i -= -1
            best_action = (int(best_index / 4), int(best_index % 4))
            for action, state in next_states.items():
                if action == best_action:
                    best_state = state
                    break

            if best_action is None:
                print(best_state, "is not a valid action")

        reward, done = env.play(best_action[0], best_action[1], render=True, render_delay=None)

    return env.get_game_score()

print("starting")

gen = neat.Generation(40, BOARD_WIDTH * BOARD_HEIGHT, BOARD_WIDTH * 4)
fitnesses = [fitness(n) for n in gen.nns]
for i in range(GENERATIONS):
    gen = neat.Generation(40, prev=gen, fitness=fitness)
    print("Fitness for generation {} is".format(i), gen.bestPrev)
    print("Finished generation {}.".format(i+1))






