"""
Classes for environments
"""

import numpy as np


class BanditEnv(object):
    # TODO give history everywhere

    def __init__(self):
        raise (NotImplementedError)

    def play(self, arm):
        raise (NotImplementedError)


class BernoulliBandit(object):


    def __init__(self, nb_arms, means, seed=None, keep_history=True):
        """
            nb_arms [int] : number of arms
            means [list[float]] : list of means of the arms
            seed [int] : seed for the random number generator
            keep_history [bool] : whether to keep history of the plays
        """

        assert (len(means) == nb_arms)
        self.means = means

        self.keep_history = keep_history
        self.history = []

        self.rand = np.random.default_rng(seed)

    def play(self, arm):
        """
        Play an arm and return a random reward
            arm [int] : index of the arm to play
        """

        mean = self.means[arm]
        reward = self.rand.binomial(1, mean)

        if self.keep_history:
            self.history.append((arm, reward))
        
        return reward

    def reset(self):
        pass


class GaussianBandit(object):

    def __init__(self, nb_arms, means, stds=None, seed=None, keep_history=True):
        """
            nb_arms [int] : number of arms
            means [list[float]] : list of means of the arms
            stds [list[float]] : list of standard deviations of the arms.
                If None, all arms have std 1.
            seed [int] : seed for the random number generator
            keep_history [bool] : whether to keep history of the plays
        """

        assert (len(means) == nb_arms)
        self.means = means

        if stds == None:
            stds = [1]*nb_arms
        assert (len(stds) == nb_arms)
        self.stds = stds

        self.keep_history = keep_history
        self.history = []

        self.rand = np.random.default_rng(seed)

    def play(self, arm):
        """
        Play an arm and return a random reward
            arm [int] : index of the arm to play
        """

        mean, std = self.means[arm], self.stds[arm]
        reward = self.rand.normal(mean, std)

        if self.keep_history:
            self.history.append((arm, reward))
        return reward

    def reset(self):
        pass
