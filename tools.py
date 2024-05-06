import numpy as np
import matplotlib as plt
import env
import agent


# ####################################################### #
# The functions in this file have not all been tested yet #
#                   and may not work                      #
# ####################################################### #

def generate_bandits(nb_bandits: int, bandit_version: str, nb_arms: int, means: list, stds=None, seeds=None):

    # Check means
    assert (len(means) == nb_arms)

    # Check seeds
    if seeds == None:
        seeds = [None]*nb_bandits
    elif len(seeds) == 1:
        seed_init = seeds[0]
        seeds = [seed_init + i for i in range(nb_bandits)]
    else:
        assert (len(seeds) == nb_bandits)

    # Check stds
    if stds == None:
        stds = [1]*nb_arms
    elif len(stds) == 1:
        stds = stds*nb_arms
    else:
        assert (len(stds) == nb_arms)

    # create bandits
    if bandit_version == "gaussian":
        bandits = [env.GaussianBandit(
            nb_arms, means, stds, seeds[i]) for i in range(nb_bandits)]
    elif bandit_version == "bernoulli":
        bandits = [env.BernoulliBandit(nb_arms, means, seeds[i])
                   for i in range(nb_bandits)]
    else:
        raise Exception(
            f"{bandit_version} is not a recognized bandit version.")

    return bandits


def generate_agents(nb_agents: int, agent_version: str, nb_arms: int, ucb_version: str = "ucb2"):

    if agent_version == "ucb":
        agents = [agent.UCBBandit(nb_arms, ucb_version)
                  for i in range(nb_agents)]
        return agents
    else:
        raise Exception(f"{agent_version} not recognized as a bandit version")


def run_experiment(bandits: list, agents: list, nb_traj: int, length_traj: int):

    for traj in range(nb_traj):
        print(f"Iteration {traj+1}", end="\r")
        for _ in range(length_traj):
            arm = agents[traj].best_action()
            reward = bandits[traj].play(arm)
            agents[traj].observe(arm, reward)


def get_arm_proportion(agents, log_normalized=False):
    if log_normalized:
        return np.array([agent.draws/np.log(agent.time) for agent in agents])
    else:
        return np.array([agent.draws for agent in agents])


def get_arm_proportions_time(agents, log_normalized=False):
    nb_arms = agents[0].nb_arms
    time = agents[0].time

    draws_proportions = []

    if log_normalized:
        log_normalization = np.array([max(1, np.log(t)) for t in range(time)])

    for agent in agents:
        draws = np.zeros((nb_arms, time))
        for t in range(time):
            arm_chosen = agent.history[t][0]
            draws[arm_chosen, t] = 1
        draws = draws.cumsum(axis=1)

        if log_normalized:
            draws /= log_normalization

        draws_proportions.append(draws)

    return draws_proportions


class Run(object):

    def __init__(self):
        self.trajs = []
        self.trajs_loged = []

    def run_trajectories(self, nb_arms, means, traj_length, nb_traj, stds=None, seed_init=0, seeds=None, keep_history=True):

        for traj in range(nb_traj):
            ag = agent.UCBBandit(nb_arms, keep_history=keep_history)
            bandit = env.GaussianBandit(
                nb_arms, means, stds, seed=seed_init + traj, keep_history=keep_history)
            sum_draws = [0]
            sum_draws_loged = [0]
            print(f"Iteration {traj}", end="\r")

            for iter in range(traj_length):
                arm = ag.best_action()
                reward = bandit.play(arm)
                ag.observe(arm, reward)
                sum_draws.append(sum_draws[-1] + arm)
                sum_draws_loged.append(sum_draws[-1]/np.log(iter+2))

            self.trajs.append(sum_draws)
            self.trajs_loged.append(sum_draws_loged)

    def plot_trajectories(self, print_average=True, logscale=False):
        avg = []
        traj_length = len(self.trajs[0])

        for i in range(traj_length):
            avg_draws = np.mean([trajl[i] for trajl in self.trajs_loged])
            avg.append(avg_draws)

        for trajl in self.trajs_loged:
            plt.plot(trajl)
        plt.plot(avg, color="black", linewidth=5)
