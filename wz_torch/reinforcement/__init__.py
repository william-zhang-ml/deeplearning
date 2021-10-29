from typing import Tuple
from gym import Env


class PitfallEnv(Env):
    """ Maze where the agent must walk to the goal while avoiding pitfalls. """
    def __init__(self) -> None:
        """ Constructor. """
        self.row = 0
        self.col = 0
        self.board = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11]
        ]
        self.blocked = [6]
        self.pit = [7]
        self.goal = [11]
        self.pit_reward = -10
        self.goal_reward = 10
        self.wait_reward = -1

    @property
    def num_rows(self) -> int:
        """ Number of board rows.

        :return: number of board rows
        :rtype:  int
        """
        return len(self.board)

    @property
    def num_cols(self) -> int:
        """ Number of board columns.

        :return: number of board columns
        :rtype:  int
        """
        return len(self.board[0])

    def reset(self) -> int:
        """ Reset the environment.

        :return: initial state
        :rtype:  int
        """
        self.row, self.col = 0, 0
        state = self.board[self.row][self.col]
        return state

    def step(self, action: int) -> Tuple[object, float, bool, dict]:
        """ Advance the environment based on agent action.

        :param   action: up (0), down (1), left (2), right (3)
        :type    action: int
        :return:         state, reward, whether game over, debug info
        :rtype:          Tuple[object, float, bool, dict]
        """
        # do not allow actions if game has ended
        try:
            assert self.board[self.row][self.col] not in self.pit
            assert self.board[self.row][self.col] not in self.goal
        except AssertionError:
            raise RuntimeError('Game has already ended!')

        # figure out if the action moves the agent to a new state
        if action == 0:  # up
            row = max(0, self.row - 1)
            col = self.col
        elif action == 1:  # down
            row = min(self.num_rows - 1, self.row + 1)
            col = self.col
        elif action == 2:  # left
            row = self.row
            col = max(0, self.col - 1)
        elif action == 3:  # right
            row = self.row
            col = min(self.num_cols - 1, self.col + 1)
        else:
            raise ValueError('Invalid action submitted.')

        if self.board[row][col] not in self.blocked:
            self.row, self.col = row, col

        # gather environment outputs for caller
        state = self.board[self.row][self.col]
        if state in self.goal:
            reward = self.goal_reward
            done = True
        elif state in self.pit:
            reward = self.pit_reward
            done = True
        else:
            reward = self.wait_reward
            done = False

        return state, reward, done, {}
