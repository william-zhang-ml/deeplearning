class RampThenPlateau:
    """ Wrapper for a variable that increases over time until a point.
        One example use is a KL loss weight for variational autoencoders. """
    def __init__(self,
                 plateau_val: float,
                 num_burn: int,
                 start_val: float = 0) -> None:
        """ Constructor.

        :param plateau_val: final value
        :type  plateau_val: float
        :param num_burn:    number of increments until reaching plateau
        :type  num_burn:    int
        :param start_val:   initial value, defaults to 0
        :type  start_val:   float, optional
        """
        self.plateau_val = plateau_val
        self.num_burn = num_burn
        self.start_val = start_val
        self.increment = (plateau_val - start_val) / num_burn
        self.value = start_val

    def __repr__(self) -> str:
        """ Representation. """
        return (
            'RampThenPlateau('
            f'{self.plateau_val}, '
            f'{self.num_burn}, '
            f'{self.start_val})'
        )

    def step(self) -> float:
        """ Get current value and increment

        :return: current variable value
        :rtype:  float
        """
        to_ret = self.value
        self.value = min(self.value + self.increment, self.plateau_val)
        return to_ret
