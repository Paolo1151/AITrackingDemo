from enum import Enum
from collections import deque

import numpy as np
import time
import os


class Position(Enum):
    NW = 0
    NN = 1
    NE = 2
    WW = 3
    C = 4
    EE = 5
    SW = 6
    SS = 7
    SE = 8


class Movement(Enum):
    N = 0
    S = 1
    E = 2
    W = 3


state_set = [
    Position.NW,
    Position.NN,
    Position.NE,
    Position.WW,
    Position.C,
    Position.EE,
    Position.SW,
    Position.SS,
    Position.SE
]

action_set = [
    Movement.N,
    Movement.S,
    Movement.E,
    Movement.W
]

trans_tensor = np.array([
    # North Transition Matrix
    [[1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0]],

    # South Transition Matrix
    [[0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1]],

    # East Transition Matrix
    [[0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 1]],

    # West Transition Matrix
    [[1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0]]
])

reward_place = Position(np.random.choice(list(set(state_set) - {Position.NW}), 1)[0])

collected = False


def reward(state, action):
    if state == Position.NW and (action == Movement.N or action == Movement.W):
        return -50
    elif state == Position.NN and (action == Movement.N):
        return -50
    elif state == Position.NE and (action == Movement.N or action == Movement.E):
        return -50
    elif state == Position.WW and (action == Movement.W):
        return -50
    elif state == Position.EE and (action == Movement.E):
        return -50
    elif state == Position.SW and (action == Movement.S or action == Movement.W):
        return -50
    elif state == Position.SS and (action == Movement.S):
        return -50
    elif state == Position.SE and (action == Movement.S or action == Movement.E):
        return -50
    elif state == reward_place:
        return 100
    else:
        return 0


def generate_q_matrix_list_finite(reward_func, h=20, q_matrix_list=[], prev_q_matrix=None, n=1):
    if n > h:
        return q_matrix_list
    else:
        reward_matrix = generate_reward_matrix(reward_func)
        if prev_q_matrix is not None:
            q_max = np.amax(prev_q_matrix, 1)
            q_addend = []
            for state in state_set:
                row = []
                for action in action_set:
                    row.append(np.dot(trans_tensor[action.value, state.value], q_max))
                q_addend.append(row)
            q_new = np.add(reward_matrix, q_addend)
        else:
            q_new = reward_matrix
        q_matrix_list.append(q_new)
        return generate_q_matrix_list_finite(reward_func, h, q_matrix_list, q_new, n + 1)


def generate_q_matrix_infinite(reward_func, iterations=20, prev_q_matrix=None, n=20):
    if iterations <= 0:
        return prev_q_matrix
    else:
        reward_matrix = generate_reward_matrix(reward_func)
        if prev_q_matrix is not None:
            q_max = np.amax(prev_q_matrix, 1)
            q_addend = []
            for state in state_set:
                row = []
                for action in action_set:
                    row.append(np.dot(trans_tensor[action.value, state.value], q_max))
                q_addend.append(row)
            q_new = np.add(reward_matrix, q_addend)
        else:
            q_new = reward_matrix
        return generate_q_matrix_infinite(reward_func, iterations - 1, q_new)


def generate_reward_matrix(reward_func):
    reward_matrix = []
    for state in state_set:
        row = []
        for action in action_set:
            row.append(reward_func(state, action))
        reward_matrix.append(row)
    return np.array(reward_matrix)


def interact_with_environment(initial_state=Position.NW):
    global collected
    prev_moves = deque([Movement.N, Movement.N, Movement.N, Movement.N, Movement.N], maxlen=5)
    q_matrix = generate_q_matrix_infinite(reward)
    q_matrix = flag_matrix(q_matrix)
    state = initial_state
    turn = 1
    new_map = refresh_map(state)
    clear_console()
    print(f'Turn 0: ')
    print(f'+++++++++++++\n' +
          f'+ {new_map[0]} + {new_map[1]} + {new_map[2]} +\n' +
          f'+++++++++++++\n' +
          f'+ {new_map[3]} + {new_map[4]} + {new_map[5]} +\n' +
          f'+++++++++++++\n' +
          f'+ {new_map[6]} + {new_map[7]} + {new_map[8]} +\n' +
          f'+++++++++++++\n')
    time.sleep(2)
    while not collected:
        best_action = generate_best_action(state, q_matrix)
        state = change_state(state, best_action)
        if state == reward_place:
            collected = True
        clear_console()
        print(f'Turn {turn}: ')
        turn += 1
        new_map = refresh_map(state)
        print(f'+++++++++++++\n' +
                f'+ {new_map[0]} + {new_map[1]} + {new_map[2]} +\n' +
                f'+++++++++++++\n' +
                f'+ {new_map[3]} + {new_map[4]} + {new_map[5]} +\n' +
                f'+++++++++++++\n' +
                f'+ {new_map[6]} + {new_map[7]} + {new_map[8]} +\n' +
                f'+++++++++++++\n')
        time.sleep(2)
        if prev_moves[0] == best_action:
            if prev_moves[1] == best_action:
                if prev_moves[2] == best_action:
                    if prev_moves[3] == best_action:
                        if prev_moves[4] == best_action:
                            raise Exception('STUCK!')
        prev_moves.append(best_action)


def flag_matrix(q_matrix):
    q_matrix[Position.NW.value][Movement.N.value] = -5000
    q_matrix[Position.NW.value][Movement.W.value] = -5000
    q_matrix[Position.NN.value][Movement.N.value] = -5000
    q_matrix[Position.NE.value][Movement.N.value] = -5000
    q_matrix[Position.NE.value][Movement.E.value] = -5000
    q_matrix[Position.WW.value][Movement.W.value] = -5000
    q_matrix[Position.EE.value][Movement.E.value] = -5000
    q_matrix[Position.SW.value][Movement.S.value] = -5000
    q_matrix[Position.SW.value][Movement.W.value] = -5000
    q_matrix[Position.SS.value][Movement.S.value] = -5000
    q_matrix[Position.SE.value][Movement.S.value] = -5000
    q_matrix[Position.SE.value][Movement.E.value] = -5000
    return q_matrix


def reduce_trust(q_matrix, state, action, weight):
    q_matrix[state.value][action.value] = q_matrix[state.value][action.value] - weight
    return q_matrix


def generate_best_action(state, q_matrix):
    choice = Movement(np.argmax(q_matrix[state.value]))
    return choice


def change_state(state, action):
    probabilities = trans_tensor[action.value][state.value]
    choice = np.dot(probabilities, np.array(range(0, 9)).T)
    return Position(choice)


agent_sprite = '*'

reward_sprite = '$'


def refresh_map(agent_pos):
    new_map = [
        ' ', ' ', ' ',
        ' ', ' ', ' ',
        ' ', ' ', ' '
    ]
    new_map[reward_place.value] = reward_sprite
    new_map[agent_pos.value] = agent_sprite
    return new_map


def clear_console():
    os.system('cls')


def main():
    while not collected:
        interact_with_environment()
    print('Reward Collected!')


if __name__ == '__main__':
    main()
