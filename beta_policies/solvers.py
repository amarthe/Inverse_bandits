import numpy as np
from distribution import Distribution





#########################################################################################
#                                                                                       #
#                                 Banditsolver                                          #
#                                                                                       #
#########################################################################################


class BetaPolicyBanditSolver(object):

    def __init__(self, distributions, accuracy:float=1e-3):
        """
            distributions [list[Distribution]] : list of distributions of the arms
            accuracy [float] : accuracy of the solver
        """
        self.distributions = distributions
        self.accuracy = accuracy

        self.intervals_computed = 0
        self.crossings = []

        supports = [dist.support() for dist in self.distributions]
        self.rmax = max([supp[1] for supp in supports])
        self.rmin = min([supp[0] for supp in supports])

        if self.rmin != self.rmax:
            means = sorted([dist.mean for dist in self.distributions])
            self.mean_interval = 8*(means[-1] - means[-2])/(self.rmax-self.rmin)**2
        else:
            self.mean_interval = 1e7

        #Eliminate duplicate distributions
        #TODO : verify there are no issues with r_min and the modification of lowest_reward
        self.eliminate_duplicates(lowest_reward=self.rmin)

    def compute_safe_interval(self, beta:float, dist1=None, dist2=None)->tuple:
        """
        Compute an interval around beta on which one distribution is always better than the other one w.r.t. the entropic risk measure.
            beta [float] : risk parameter for the entropic risk measure
            dist1 [Distribution] : first distribution
            dist2 [Distribution] : second distribution
        in the code, we always have u1 > u2.
        If dist1 and dist2 are None, we consider the two best distributions.
        Beta should never be exactly 0.
        """

        #If the two distributions are not given, we take the two best ones.
        if (dist1 is None) or (dist2 is None):
            assert ((dist1 is None) and (dist2 is None))
            idx_best_two, _ = self.evaluate_max_two(beta)
            dist1, dist2 = self.distributions[idx_best_two[0]], self.distributions[idx_best_two[1]]    

        #Compute the entropic risk measures and the supports
        u1 = dist1.entRM(beta)
        u2 = dist2.entRM(beta)
        u1, u2 = max(u1, u2), min(u1, u2)

        rmax = self.rmax
        rmin = self.rmin
        #Check for deterministic optimal/worst distribution (such distributions makes a division by 0 in the bound computation)
        if u1 == rmax or u2 == rmin:
            return (-1e7, 1e7)
        
        #Compute the bound
        if beta > 0:
            delta_sup = np.abs(u1 - u2)/(rmax-u1)
            delta_inf = np.abs(u1 - u2)/(rmax-u2)
        else:
            delta_sup = np.abs(u1 - u2)/(u1-rmin)
            delta_inf = np.abs(u1 - u2)/(u2-rmin)

        lower_term = np.abs(beta)*delta_sup
        upper_term = np.abs(beta)*delta_inf

        bound = (beta - lower_term, beta + upper_term)

        #Update the number of intervals computed #might be better to do it in the solve functions.
        self.intervals_computed += 1

        return bound

    def evaluate(self, beta:float, arm:int)->float:
        """
        Evaluate the entropic risk measure of an arm at a given beta.
            beta [float] : risk parameter for the entropic risk measure
            arm [int] : index of the arm
        """
        return self.distributions[arm].entRM(beta)

    def evaluate_max(self, beta:float)->list:
        """
        Returns the index of the arm with the highest entropic risk measure at a given beta, along its value.
            beta [float] : risk parameter for the entropic risk measure
        """
        values = [dist.entRM(beta) for dist in self.distributions]
        idx = np.argmax(values)
        return idx, values[idx]

    def evaluate_max_two(self, beta:float)->list:
        """
        Returns the indices of the two arms with the highest entropic risk measure at a given beta, along their values.
            beta [float] : risk parameter for the entropic risk measure
        """
        values = np.array([dist.entRM(beta) for dist in self.distributions])
        idx = (np.argsort(values)[-2:])[::-1]
        return idx, values[idx]

    def find_crossing(self, beta_low:float, beta_high:float, arm1:int, arm2:int, eps:float=0.)->float:
        """
        Find the beta risk parameter at which the two distributions cross w.r.t. the entropic risk measure. Assumes that on the beta_low and beta_high, none of the true distribution is the best on both. Uses dichotomy method.
            beta_low [float] : lower bound for the search
            beta_high [float] : upper bound for the search
            dist1 [Distribution] : first distribution
            dist2 [Distribution] : second distribution
            eps [float] : accuracy of the search
        """
        if eps == 0.:
            eps = self.accuracy

        dist1 = self.distributions[arm1]
        dist2 = self.distributions[arm2]

        #Check that the two distributions are optimal on the two ends, and not only one.
        if np.sign(dist1.entRM(beta_low) - dist2.entRM(beta_low)) == np.sign(dist1.entRM(beta_high) - dist2.entRM(beta_high)):
            return None

        #Dichotomy methods
        else:
            while beta_high - beta_low > eps:
                beta = (beta_low + beta_high)/2
                if np.sign(dist1.entRM(beta_low) - dist2.entRM(beta_low)) == np.sign(dist1.entRM(beta) - dist2.entRM(beta)):
                    beta_low = beta
                else:
                    beta_high = beta
            return beta

    def compute_crossing_limits(self)->list:
        """
        Computes the upper limit above which the entropic risk measure of one distribution is always better than the other one.
        TODO : update with the general formulae
        """
        print("This function is deprecated and should not be used")
        reward_maxs = [dist.support()[1] for dist in self.distributions]
        probas = [self.distributions[i][reward_maxs[i]] for i in range(len(self.distributions))]

        #idx of the distribution with the 2 highest rewards possible
        best_two = np.argsort(reward_maxs)[-2:]
        r1max, r2max = reward_maxs[best_two[1]], reward_maxs[best_two[0]] #r1 > r2
        p1max, p2max = probas[best_two[1]], probas[best_two[0]]

        #prendre les 1-xi ou xi est les deux plus grandes proba.
        #prendre le minimum des rewards
        best_two_proba = np.sort(probas)[-2:]
        p1min, p2min = 1-best_two_proba[1], 1-best_two_proba[0]
        r1min, r2min = min(reward_maxs), 0

        return np.log((p2min - p1min)/(1-p1min))/(r1min), -np.log(p1max)/(r1max-r2max) #TODO: change for β < 0

    def solve(self, lower_limit=None, upper_limit=None)->list:
        """
        Solve the beta-entropic risk measure problem for the given distributions.
        """
        #Reset the crossings for avoiding duplicates TODO : remove this
        self.crossings = []

        if lower_limit is None or upper_limit is None:
            assert (lower_limit is None) and (upper_limit is None)
            lower, upper = self.compute_crossing_limits()
            beta_low = lower -1 if lower_limit is None else lower_limit
            beta_high = upper +1 if upper_limit is None else upper_limit # ±1 to prevent equality case
        else:
            beta_low, beta_high = lower_limit, upper_limit

        #self._solve_recursive(beta_low, beta_high, idx_best_lower, idx_best_higher)
        self._solve_interval(beta_low, beta_high, reset=True)
        self.crossings = sorted(self.crossings)

    def _solve_recursive(self, low:float, high:float, arm_lower:int, arm_higher:int):
        """
        Recursive utility function to solve the beta-entropic risk measure problem.
        WARNING : Only works if only one breakpoint is between two distributions.
        """
        print("This function is deprecated and should not be used")
        beta_cross = self.find_crossing(low, high, arm_lower, arm_higher)

        idx_best_cross, _ = self.evaluate_max(beta_cross)
        #TODO : vérifier que ça marche comme on le souhaite ! (intervalles ?) Vérifier à droite et à gauche que c’est bien les bon bras optimaux.
        if idx_best_cross != arm_lower and idx_best_cross != arm_higher:
            self._solve_recursive(beta_cross, high, idx_best_cross, arm_higher)
            self._solve_recursive(low, beta_cross, arm_lower, idx_best_cross)
        
        else:
            self.crossings.append(beta_cross)

    def _solve_interval(self, beta_low:float, beta_high:float, reset:bool=False)->list:
        """
        Solve the beta-entropic risk measure problem for the given distributions.
        """
        #Check for several actions in the problem
        if len(self.distributions) == 1:
            print("Only one distribution, no crossing")
            return None

        #Reset the crossings for avoiding duplicates
        if reset:
            self.crossings = []
            self.intervals_computed = 0

        #Check for change near 0
        if beta_low < 0 and beta_high > 0:
            best_arm_low, _ = self.evaluate_max(-self.accuracy)
            best_arm_high, _ = self.evaluate_max(self.accuracy)
            if best_arm_low != best_arm_high:
                self.crossings.append(0)

        #solves beta < 0 : - epsilon -> low
        beta = min(-self.accuracy, beta_high, -self.mean_interval)
        best_arm, _ = self.evaluate_max(beta)

        while beta_low < beta:
            bound, _ = self.compute_safe_interval(beta)

            #if interval smaller that accuracy, increase by accuracy and check for change of action
            if np.abs(beta-bound) < self.accuracy:
                beta = beta - self.accuracy
                arm, _ = self.evaluate_max(beta)
                if arm != best_arm:
                    best_arm = arm
                    self.crossings.append(beta) #can do better with a "beta + accuracy/2"
            #else, increase by the bound
            else:
                beta = bound
        
        #solves beta > 0 : + epsilon -> high
        beta = max(self.accuracy, beta_low, self.mean_interval)
        best_arm, _ = self.evaluate_max(beta)

        while beta < beta_high:
            _, bound = self.compute_safe_interval(beta)

            if np.abs(beta-bound) < self.accuracy:
                beta = beta + self.accuracy
                arm, _ = self.evaluate_max(beta)
                if arm != best_arm:
                    best_arm = arm
                    self.crossings.append(beta)
            else:
                beta = bound
        
        self.crossings = sorted(self.crossings)
        return None

    def eliminate_duplicates(self, lowest_reward=-0)->None:
        """
        Eliminate duplicate distributions.
        Currently, the function replaces the duplicate distribution by a new one with a unique atom of -10 or less, hoping for it to always be the worst arm.
        """
        #TODO : vérifier qu’il n’y a pas d’erreur dans l’élimination des doublons.
        temp_distrib = []
        b = True
        i = 1
        for distrib in self.distributions:
            if distrib not in temp_distrib:
                temp_distrib.append(distrib)
            else:
                temp_distrib.append(Distribution({-lowest_reward-10-i:1}))
                i += 1

        self.distributions = temp_distrib




#########################################################################################
#                                                                                       #
#                                    MDPsolver                                          #
#                                                                                       #
#########################################################################################


class MDPsolver():
    def __init__(self, env, horizon, accuracy=1e-3, beta_min=-10, beta_max=10) -> None:
        
        self.env = env
        self.action_space = env.action_space
        self.state_space = env.state_space
        self.horizon = horizon

        self.accuracy = accuracy
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.intervals = [[[] for _ in range(self.state_space)] for _ in range(horizon+1)]
        self.optimal_actions = [[[] for _ in range(self.state_space)] for _ in range(horizon+1)]
        self.optimal_distributions = [[[] for _ in range(self.state_space)] for _ in range(horizon+1)]

        for s in range(self.state_space):
            self.intervals[horizon][s].append((self.beta_min, self.beta_max))
            self.optimal_actions[horizon][s].append(0)
            self.optimal_distributions[horizon][s].append(Distribution({0:1}))
        
        self.intervals_computed = 0
        
    def _solve_one(self, t, s, min, max):

        distributions = self.compute_distributions(t, s, min, max)

        #project si besoin
        #pass
    
        #solve
        bandit_solver = BetaPolicyBanditSolver(distributions, accuracy=self.accuracy)
        bandit_solver.solve(min, max)
        
        #clean the intervals
        crossings = bandit_solver.crossings

        temp_intervals = self._break_to_intervals(crossings, min, max)
        temp_intervals = self.process_intervals([temp_intervals], self.accuracy)

        for (lower, upper) in temp_intervals:
            #update state intervals
            self.intervals[t][s].append((lower, upper))
            #update optimal actions
            optimal_action = bandit_solver.evaluate_max((upper+lower)/2)[0]
            self.optimal_actions[t][s].append(optimal_action)
            #update optimal distributions
            self.optimal_distributions[t][s].append(distributions[optimal_action])

        self.intervals_computed += bandit_solver.intervals_computed

    def solve(self):
        self.intervals_computed = 0
        
        #horizon-1 to 0 (included)
        for t in range(self.horizon-1, -1, -1):

            #computes each interval to look into
            intervals = self.process_intervals(self.intervals[t+1], 2*self.accuracy)
            print(f"Step {t}, computed : {intervals}")

            for s in range(self.state_space):
                for (min,max) in intervals:
                    self._solve_one(t, s, min, max)
        
            print(f"Total intervals computed : {self.intervals_computed}")

    def compute_distributions(self, t, s, min, max):
        distributions = []
        for a in range(self.action_space):
            distrib = self.compute_action_distribution(t, s, a, min, max)
            distributions.append(distrib)
        return distributions

    def compute_action_distribution(self, t, s, a, min, max):
        distribution = Distribution({0.0:0})

        transition = self.env.transition(s, a, t)

        #Otherwise, compute the convexe combinaison of the future distributions
        for (p, s_, r) in transition: 
            future_reward_distribution = self.interval_optimal_distribution(t+1, s_, min, max)
            future_reward_distribution = future_reward_distribution.transfer(1,float(r))
            distribution += p * future_reward_distribution

        #distribution.normalize() #TODO : vérifier que ca pose pas de problèmes
        distribution._clean2()    #TODO : vérifier que ca pose pas de problèmes
        return distribution

    def interval_optimal_distribution(self, t, s, min, max):
        #TODO : gérer les cas où deux intervals consécutifs ont la même distrib optimale.
        for (idx, (lower, upper)) in enumerate(self.intervals[t][s]):
            if self._is_in_interval((min,max), (lower, upper), 2*self.accuracy):
                return self.optimal_distributions[t][s][idx]

        #debug TODO make it true error message
        print(f"h : {t}, s : {s}, interval : ({min},{max})")
        print(f"Intervals to search in : {self.intervals[t][s]}")
        raise ValueError("A single optimal distribution was not found in the interval")

    @staticmethod
    def _is_in_interval(inside, outside, accuracy=1e-4):
        i1, i2 = inside
        o1, o2 = outside
        return (o1-accuracy <= i1 and i2 <= o2+accuracy)

    @staticmethod
    def _break_to_intervals(breakpoints, min, max):
        lower = min
        temp_intervals = []
        for upper in breakpoints:
            temp_intervals.append((lower, upper))
            lower = upper
        temp_intervals.append((lower, max))

        return temp_intervals

    #TODO : rewrite without chatgpt
    @staticmethod
    def process_intervals(interval_lists, threshold):
    # Flatten the list of intervals into a single sorted list of borders
        borders = []
        for interval_list in interval_lists:
            for start, end in interval_list:
                borders.append(start)
                borders.append(end)
        borders.sort()  # Ensure the list is sorted

        # Deduplicate borders within the specified threshold
        deduplicated_borders = []
        if borders:
            deduplicated_borders.append(borders[0])
            for border in borders[1:]:
                if border - deduplicated_borders[-1] > threshold:
                    deduplicated_borders.append(border)

        # Create intervals from deduplicated borders
        merged_intervals = []
        for i in range(len(deduplicated_borders) - 1):
            merged_intervals.append((deduplicated_borders[i], deduplicated_borders[i + 1]))

        return merged_intervals