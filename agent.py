"""
Classes for agents
"""

import numpy as np


class BanditAgent(object):
    # TODO give history everywhere

    def __init__(self):
        raise (NotImplementedError)

    def observe(self):
        raise (NotImplementedError)

    def best_action(self):
        raise (NotImplementedError)


class UCBBandit(BanditAgent):
    # TODO add history of UCBs

    def __init__(self, nb_arms, ucb_version="ucb2", keep_history=True):
        self.nb_arms = nb_arms
        self.estimations = [0.]*nb_arms
        self.ucbs = [np.inf]*nb_arms
        self.draws = [0]*nb_arms
        self.time = 0

        self.keep_history = keep_history
        self.history = []

        self.ucb_version = ucb_version

    def observe(self, arm, reward):
        if self.keep_history:
            self.history.append((arm, reward))

        n = self.draws[arm]
        self.estimations[arm] = (
            n/(n+1))*self.estimations[arm] + (1/(n+1))*reward

        self.draws[arm] += 1
        self.time += 1

        self._update_ucb()

    def best_action(self):
        return np.argmax(self.ucbs)

    def _ucb_uncertainty(self, arm):
        t = self.time
        Ti = self.draws[arm]
        if Ti == 0.:
            return np.inf

        # UCB with confidence 1/n²
        elif self.ucb_version == "ucb":
            return np.sqrt(4*np.log(t)/Ti)

        # asymptotic optimal
        elif self.ucb_version == "ucb2":
            def f(n): return (1 + n*(np.log(np.log(n))) if n > 3 else 1)
            return np.sqrt(2*np.log(f(t))/Ti)

        # MOSS algorithm, minimax optimal
        elif self.ucb_version == "moss":
            def logp(x): return np.log(np.max([1, x]))
            return np.sqrt(4*logp(t/(Ti*self.nb_arms))/Ti)

        elif self.ucb_version == "kl-ucb":
            raise NotImplementedError

        else:
            raise ValueError(
                f"{self.ucb_version} not recognized as an UCB algorithm")

    def _update_ucb(self):
        for arm in range(self.nb_arms):
            self.ucbs[arm] = self.estimations[arm] + self._ucb_uncertainty(arm)

    def reset(self):
        pass

class ThompsonSamplingBandit(BanditAgent):
    pass

