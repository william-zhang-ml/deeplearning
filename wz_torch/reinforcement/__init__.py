from gym import Env


class PitfallEnv(Env):
    """ Maze where the agent must walk to the goal while avoiding pitfalls. """
    def __init__(self):
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
