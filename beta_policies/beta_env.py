import numpy as np
import matplotlib.pyplot as plt
from distribution import Distribution
import matplotlib.patches as patches
import enum

class BetaTestingDistribution(object):
    """
    All the rewards have to be in the form of r(x,a), and should not depend on the arrived state x'.
    """

    def __init__(self):
        pass

    def generate_bandit(self):
        pass

    def print_entRM(self):
        pass

    def sample(self, arm, nb_samples):
        pass

class BernouilliDistributions(object):

    def __init__(self, nb_arms, means=None, scales=None, random=False, max_scale=2, seed=None):
        """
            nb_arms [int] : number of arms
            means [list[float]] : list of means of the arms
            scales [list[float]] : list of scales of the arms
            random [bool] : whether to generate random means and scales
            max_scale [float] : maximum scale for the random scales
            seed [int] : seed for the random number generator
        """

        self.nb_arms = nb_arms
        self.seed = seed

        if random:
            self.rand   = np.random.default_rng(self.seed)
            self.means  = self.rand.uniform(0, 1, nb_arms)
            self.scales = self.rand.uniform(0, max_scale, nb_arms)  
        else:
            assert (means  is not None)
            assert (scales is not None)
            assert (len(means)  == nb_arms)
            assert (len(scales) == nb_arms)

            self.means  = means
            self.scales = scales

        self.distributions = [Distribution({0:1-mean, scale:mean}) for mean, scale in zip(self.means, self.scales)]

    def generate_new_bandit(self, nb_arms=None, means=None, scales=None, random=False, max_scale=2):
        """
            nb_arms [int] : number of arms
            means [list[float]] : list of means of the arms
            scales [list[float]] : list of scales of the arms
            random [bool] : whether to generate random means and scales
            max_scale [float] : maximum scale for the random scales
        """
        if nb_arms is None:
            nb_arms = self.nb_arms

        self.__init__(nb_arms, means, scales, random, max_scale, self.seed)
        return self.distributions

    def print_entRM(self, beta_min, beta_max, nb_points):
        """
            beta_min [float] : minimum value on the x-axis
            beta_max [float] : maximum value of the x-axis
            nb_points [int] : number of points to plot
        """
        fig, ax = plt.subplots()
        abscissas = np.linspace(beta_min, beta_max, nb_points)
        for (i, dist) in enumerate(self.distributions):
            #dist.print_entRM(beta_min, beta_max, nb_points)
            ordinates = [dist.entRM(beta) for beta in abscissas]
            ax.plot(abscissas, ordinates, label=f"Arm {i}")
        ax.legend()
        plt.plot()

class GaussianDistributions(object):

    def __init__():
        pass    

####################################################################
#
#                        MDP ENVIRONMENTS
#
####################################################################

 
class TestEnv(object):
    """Environment to test Bellman Operator on Distributions
    States :
        Start :
        - 0 : the beginning
        Intermediates :
        - 1 : with a reward Ber(0.9)
        - 2 : with a reward 2*Ber(0.2)
        - 3 : with a reward Z
        - 4 : with a reward 0
        - 5 : with a reward 0
        Reward States :
        - 6 : with a reward -1
        - 7 : with a reward 0
        - 8 : with a reward 1
        - 9 : with a reward 2
        End :
        - 10 : the end
    
        Intermediates states rewards are obtained by forcing all actions to have the same transition, stochasticly going to rewards state with appropriate reward.
        Reward states all lead to the end with a fixed reward fixed above

        For the beginning, 3 actions are avaialble:
        - 0 : go to 1 (Ber(0.9))
        - 1 : ½ go to 3 (Z), ¼ go to 4 (0), ¼ go to 5 (0)
        - 2 : go to 2 (2*Ber(0.2))
    """

    def __init__(self):
        self._state_shape = (11)
        self._action_shape = (3)

    @property
    def state_space(self):
        return self._state_shape

    @property
    def action_space(self):
        return self._action_shape

    def transition(self, pos, action, step):

        #Beginning
        if pos == 0:
            if action == 0:
                return [[1, 1, 0]]
            elif action == 1:
                return [[0.5, 3, 0], [0.25, 4, 0], [0.25, 5, 0]]
            elif action == 2:
                return [[1, 2, 0]]

        #Intermediate States
        elif pos == 1:
            #Ber(0.9)
            return [[0.9, 8, 0], [0.1, 7, 0]]
        elif pos == 2:
            #2*Ber(0.2)
            return [[0.2, 9, 0], [0.8, 7, 0]]
        elif pos == 3:
            #Z
            return [[0.5, 6, 0], [0.5, 8, 0]]
        elif pos == 4:
            #0
            return [[1, 7, 0]]
        elif pos == 5:
            #0
            return [[1, 7, 0]]
        
        #Reward States
        elif pos == 6:
            #-1
            return [[1, 10, -1]]
        elif pos == 7:
            #0
            return [[1, 10, 0]]
        elif pos == 8:
            #1
            return [[1, 10, 1]]
        elif pos == 9:
            #2
            return [[1, 10, 2]]

        #End
        elif pos == 10:
            return [[1, 10, 0]]
        else:
            raise Exception("State out of bound")
        
class AugmentedTestEnv(object):
    """Environment to test Bellman Operator on Distributions
    States :
        Start :
        - 0 : the beginning
        - 11 : other beginning
        - 12 : even better beginning
        Intermediates :
        - 1 : with a reward Ber(0.9)
        - 2 : with a reward 2*Ber(0.2)
        - 3 : with a reward Z
        - 4 : with a reward 0
        - 5 : with a reward 0
        Reward States :
        - 6 : with a reward -1
        - 7 : with a reward 0
        - 8 : with a reward 1
        - 9 : with a reward 2
        End :
        - 10 : the end
    
        Intermediates states rewards are obtained by forcing all actions to have the same transition, stochasticly going to rewards state with appropriate reward.
        Reward states all lead to the end with a fixed reward fixed above

        For the beginning [0], 3 actions are avaialble:
        - 0 : go to 1 (Ber(0.9))
        - 1 : ½ go to 3 (Z), ¼ go to 4 (0), ¼ go to 5 (0)
        - 2 : go to 2 (2*Ber(0.2))

        For the other [11]: 
        TODO : add a fix reward ?
        - 0 : ½ go to 1, ½ go to 2
        - 1 : ¼ go to Z, ¾ go to 2
        - 2 : 3

        further start [12]:
        - 0 : goes to 11 + 1
        - 1 : goes to 0
        - 2 : ½ goes to 11, ½ goes to 3 (Z).
    """

    def __init__(self):
        self._state_shape = (13)
        self._action_shape = (3)

    @property
    def state_space(self):
        return self._state_shape

    @property
    def action_space(self):
        return self._action_shape

    def transition(self, pos, action, step):
        
        #Further start
        if pos == 12:
            if action == 0:
                return [[1, 11, 0]]
            if action == 1:
                return [[1, 0, 0]]
            if action == 2:
                return [[0.5, 11, 0], [0.5, 3, 0]]

        #Beginning
        if pos == 11:
            #TODO : add deterministic reward ?
            if action == 0:
                return [[0.5, 1, 0], [0.5, 2, 0]]
            if action == 1:
                return [[0.25, 3, 0], [0.75, 2, 0]]
            if action == 2:
                return [[1, 3, 0]]

        if pos == 0:
            if action == 0:
                return [[1, 1, 0]]
            elif action == 1:
                return [[0.5, 3, 0], [0.25, 4, 0], [0.25, 5, 0]]
            elif action == 2:
                return [[1, 2, 0]]

        #Intermediate States
        elif pos == 1:
            #Ber(0.9)
            return [[0.9, 8, 0], [0.1, 7, 0]]
        elif pos == 2:
            #2*Ber(0.2)
            return [[0.2, 9, 0], [0.8, 7, 0]]
        elif pos == 3:
            #Z
            return [[0.5, 6, 0], [0.5, 8, 0]]
        elif pos == 4:
            #0
            return [[1, 7, 0]]
        elif pos == 5:
            #0
            return [[1, 7, 0]]
        
        #Reward States
        elif pos == 6:
            #-1
            return [[1, 10, -1]]
        elif pos == 7:
            #0
            return [[1, 10, 0]]
        elif pos == 8:
            #1
            return [[1, 10, 1]]
        elif pos == 9:
            #2
            return [[1, 10, 2]]

        #End
        elif pos == 10:
            return [[1, 10, 0]]
        else:
            raise Exception("State out of bound")
    

####################################################################
#
#                        CLIFF ENVIRONMENT
#
####################################################################

class Action(enum.Enum):
    """Inplements the 4 directions"""
    Down = 0
    Up = 1
    Left = 2
    Right = 3


def action_to_dx(action):
    """given a action, returns the coordinate move associated"""
    if action.name == "Down":
        return (0, -1)
    if action.name == "Up":
        return (0, 1)
    if action.name == "Left":
        return (-1, 0)
    if action.name == "Right":
        return (1, 0)
    else:
        raise Exception("Error in action_to_dx")


def pair_sum(pair1, pair2, p_min, p_max):
    """util function to perform addition on pairs and clip them"""
    return (np.clip(pair1[0] + pair2[0], p_min[0], p_max[0]), np.clip(pair1[1] + pair2[1], p_min[1], p_max[1]))


class TerminalCliff(object):
    """
    Implements the cliff environment similar to the one in Sutton & Barto,
    to test with RL algorithms. It looks like :

        C C C C C \n
        C C C C C \n
        B C C C C \n
        X X X X E  

    B is where the environment starts, E where it ends. C are the regular cells, and X the cliff.
    The agent has to go from B to E without falling in X.
    At each step, the agent has 4 possible actions, the 4 direction. For every action, the agent
    has 0.7 to go it that direction, and 0.1 in each other directions. A action leading to out of
    bound makes the agent stay on the same place.

    Attributes:
        nb_col: number of columns of the environement
        nb_line: number of lines of the environment

        horizon: the upper limit of steps
        reward_fall: the reward given when falling off the cliff
        right_proba: the probability of going to the chosen direction
        retry: if True, when falling, the agent goes back to the beginning. Otherwise, it is stuck.

        state_shape: (nb_col, nb_line)
        action_shape: the number of actions

        self.x: x-pos of the agent
        self.y: y-pos of the agent
    """

    def __init__(self,  nb_col=5, nb_line=4, horizon=15, reward_fall=-0.5, discount=1, right_proba=0.7):

        self.nb_col = nb_col
        self.nb_line = nb_line
        self.horizon = horizon
        self.reward_fall = reward_fall
        self.discount = discount
        self.right_proba = right_proba
        self.wrong_proba = (1-right_proba)/3

        self._state_shape = ((nb_col* nb_line) + 1)
        self._action_shape = (4)

        self.x = 0
        self.y = 1
        self.end = (nb_col-1, 0)

        self.dspace = None

    @property
    def state_space(self):
        return self._state_shape

    @property
    def action_space(self):
        return self._action_shape

    def state_to_pos(self, state):
        """given a state, returns the position"""
        return (state % self.nb_col, state // self.nb_col)

    def pos_to_state(self, pos):
        """given a position, returns the state"""
        return pos[0] + pos[1]*self.nb_col

    def transition(self, pos, action, step):
        """an in-class method of the transition function"""
        # TODO : regarder pour les discout de reward : que à l’objectif, ou aussi quand ça tombe ?
        x, y = self.state_to_pos(pos)

        if pos == self.state_space-1: # terminal state
            return [[1, pos, 0]]
        if x == self.nb_col-1 and y == 0:  # the goal
            bonus_reward = (self.horizon - step)/(2*self.horizon)
            return [[1, self.state_space-1, 0.5 + bonus_reward]]
        elif y == 0:  # the cliff
            return [[1, self.state_space-1, self.reward_fall]]
        else:
            trans = [[self.wrong_proba, self.pos_to_state(pair_sum((x, y), action_to_dx(
                act), (0, 0), (self.nb_col-1, self.nb_line-1))), 0] for act in list(Action)]
            trans[action][0] = self.right_proba
            return trans

    def display_actions(self, action_array):
        action_symbols = ["↓", "↑", "←", "→"]

        for y in reversed(range(1, self.nb_line)):
            for x in range(self.nb_col):
                print(action_symbols[action_array[self.pos_to_state((x,y))]], end=" ")
            print("")

        # first row
        print("\033[1;31m" + "X "*(self.nb_col-1), end="")
        print("\033[1;32mE\033[0m")

    def print_actions(self, policy, title=None, ax=None):
        x_size, y_size = self.state_shape

        # Mapping actions to directions
        action_dict = [(0, -1), (0, 1), (-1, 0),  (1, 0)]

        # Create a figure and axis
        if ax == None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # Create the grid
        for x in range(x_size + 1):
            ax.plot([x, x], [0, y_size], color='black')
        for y in range(y_size + 1):
            ax.plot([0, x_size], [y, y], color='black')

        # Color the cliff
        for x in range(x_size-1):
            rect = patches.Rectangle(
                (x, y_size-1), 1, 1, linewidth=1, edgecolor='black', facecolor='red', alpha=0.5)
            ax.add_patch(rect)

        # Color the goal and the beginning
        rect = patches.Rectangle(
            (x_size-1, y_size-1), 1, 1, linewidth=1, edgecolor='black', facecolor='green', alpha=0.5)
        ax.add_patch(rect)
        rect = patches.Rectangle(
            (0, 2), 1, 1, linewidth=1, edgecolor='black', facecolor='blue', alpha=0.5)
        ax.add_patch(rect)

        # Add the policy arrows
        for y in range(y_size):
            for x in range(x_size):
                action_probs = policy[x][y]
                for action, prob in enumerate(action_probs):
                    dx, dy = action_dict[action]
                    ax.arrow(x + 0.5, y_size - y - 0.5, dx * prob/2, -dy * prob/2,
                             head_width=0.06, head_length=0.05, fc='blue', ec='blue', alpha=0.7)

        # Set limits and labels
        ax.set_xlim(0, x_size)
        ax.set_ylim(0, y_size)
        ax.set_xticks(np.arange(x_size + 1))
        ax.set_yticks(np.arange(y_size + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Show the plot
        ax.invert_yaxis()
        plt.grid(True)
        if title == None:
            plt.title(f"Policy for the cliff environment")
        else:
            plt.title(title)
        
        if ax == None:
            plt.show()