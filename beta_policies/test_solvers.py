import numpy as np
from beta_env import BernouilliDistributions, TestEnv, AugmentedTestEnv
from solvers import BetaPolicyBanditSolver, MDPsolver
from distribution import Distribution  

def assertEqual(a, b):
    assert(a == b)

def assertAlmostEqual(a, b, places):
    assert(round(a, places) == round(b, places))

debug = False



############################################################
# Tests
############################################################
# d1 = Distribution({0:1})
# d2 = Distribution({0:1})
# d3 = Distribution({0.:0.5, -1.0:0.25, 1.0:0.25})
# distribs = [d1, d2, d3]

# print("beta < 0")
# for dist in distribs:
#     print(f"dist: {dist}, EntRM: {dist.entRM(-5)}")
# print("beta < 0")
# for dist in distribs:
#     print(f"dist: {dist}, EntRM: {dist.entRM(5)}")

# solver = BetaPolicyBanditSolver(distribs, accuracy=1e-3)
# solver.solve(-15, 15)
# print(solver.crossings)
# print(solver.distributions)
# print(solver.evaluate_max(-5))
# print(solver.evaluate_max(5))
# exit()

############################################################
# BetaPolicyBanditSolver
############################################################
#Remark : the _solve method now uses the _solve_interval method, which doens’t need to be tested separately anymore.
#TODO : rajouter un test pour vérifier que le solveur intervalle ne cherche pas en dehors des intervalles.

means = np.linspace(0.1,0.99, 11)
scales = np.linspace(0.1, 2, 11)[::-1]
env = BernouilliDistributions(11, means=means, scales=scales)
distribs = env.distributions

### solver.solve
solver = BetaPolicyBanditSolver(distribs, accuracy=1e-3)
solver.solve(-15, 15)

crossings = solver.crossings

expected_results = [-7.05502, -3.2387, -1.5406, -0.55909, 0.12315, 0.6823, 1.2412, 1.9598, 3.3443]

assertEqual(len(crossings), len(expected_results))
for i in range(len(crossings)):
    assertAlmostEqual(crossings[i], expected_results[i], places=2)

### solver._solve_interval
solver = BetaPolicyBanditSolver(distribs, accuracy=1e-3)
solver._solve_interval(0.1,10, reset=True)
solver._solve_interval(-10,-0.1, reset=False)
crossings = solver.crossings

expected_results = [-7.0550, -3.2387, -1.5406, -0.5590, 0.12315, 0.6823, 1.2412, 1.9598, 3.3443]

assertEqual(len(crossings), len(expected_results))
for i in range(len(crossings)):
    assertAlmostEqual(crossings[i], expected_results[i], places=1)     

############################################################
#  MDPSolver
############################################################

#Note : currently, the test do not try for several intermediate optimal policies

h = 5
env = AugmentedTestEnv()
base_distrib = {
    "ber09" : Distribution({0.:0.1, 1.:0.9}),
    "2ber02" : Distribution({0.:0.8, 2.:0.2}),
    "Z" : Distribution({-1.:0.5, 1.:0.5}),
    "2Z" : Distribution({-2.:0.5, 2.:0.5}),
    "0" : Distribution({0.:1}),
}
mix_distrib = {
    "half2Z" : Distribution({-2.:0.25, 0:0.5, 2.:0.25}),
    "halfZ" : Distribution({-1.:0.25, 0:0.5, 1.:0.25}),
    "halfBer" : 0.5*base_distrib["ber09"] + 0.5*base_distrib["2ber02"],
    "quartBer" : 0.25*base_distrib["ber09"] + 0.75*base_distrib["2ber02"],
    "quartZber" : 0.25*base_distrib["Z"] + 0.75*base_distrib["2ber02"]
}
solver = MDPsolver(env, h, accuracy=1e-3)

### h=4
h -= 1

solver._solve_one(h, 10, -15, 15)

#print(solver.intervals[h][10])
assertEqual(solver.intervals[h][10], [(-15, 15)])
#print(solver.optimal_actions[h][10])
assertEqual(solver.optimal_actions[h][10], [0])
#print(solver.optimal_distributions[h][10])
assertEqual(solver.optimal_distributions[h][10], [Distribution({0.0: 1.0})])

### h=3
h -= 1

#print(solver.env.transition(9, 0, h))
#print(solver.interval_optimal_distribution(h+1, 10, -15, 15))
assertEqual(solver.interval_optimal_distribution(h+1, 10, -15, 15), Distribution({0.0: 1.0}))
distrib = solver.compute_action_distribution(h, 9, 0, -15, 15)
assertEqual(distrib, Distribution({2.0: 1.0}))

distribs = solver.compute_distributions(h, 9, -15, 15)
#print(distribs)
assertEqual(distribs, [Distribution({2.0: 1.0}), Distribution({2.0: 1.0}), Distribution({2.0: 1.0})])

for i in range(6, 10):
    #print(f"Step: h, State: {i}")
    solver._solve_one(h, i, -15, 15)
    #print(solver.intervals[h][i])
    assertEqual(solver.intervals[h][i], [(-15, 15)])
    #print(solver.optimal_actions[h][i])
    assertEqual(solver.optimal_actions[h][i], [0])
    #print(solver.optimal_distributions[h][i])
    assertEqual(solver.optimal_distributions[h][i], [Distribution({i-7: 1.0})])

### h=2
h -= 1

distribs = solver.compute_distributions(h, 1, -15, 15)
assertEqual(distribs, [Distribution({0.0: 0.1, 1.0: 0.9}), Distribution({0.0: 0.1, 1.0: 0.9}), Distribution({0.0: 0.1, 1.0: 0.9})])

for i in range(1,6):
    #print(f"Step: 2, State: {i}")
    solver._solve_one(h, i, -15, 15)
    #print(solver.intervals[h][i])
    assertEqual(solver.intervals[h][i], [(-15, 15)])
    #print(solver.optimal_actions[h][i])
    assertEqual(solver.optimal_actions[h][i], [0])

assertEqual(solver.optimal_distributions[h][1], [base_distrib["ber09"]])
assertEqual(solver.optimal_distributions[h][2], [base_distrib["2ber02"]])
assertEqual(solver.optimal_distributions[h][3], [base_distrib["Z"]])
assertEqual(solver.optimal_distributions[h][4], [base_distrib["0"]])
assertEqual(solver.optimal_distributions[h][5], [base_distrib["0"]])

### h=1
h -= 1

action_distrib = [base_distrib["ber09"], 0.5*base_distrib["Z"] +0.5*base_distrib["0"], base_distrib["2ber02"]]
#print(action_distrib)
#print(solver.compute_distributions(h, 0, -15, 15))
assertEqual(solver.compute_distributions(h, 0, -15, 15), action_distrib)

bandit_solver = BetaPolicyBanditSolver(action_distrib,)
bandit_solver.solve(-15, 15)

solver._solve_one(h, 0, -15, 15)
crossing = bandit_solver.crossings[0]
#print(crossing)
assertEqual(len(solver.intervals[h][0]), 2)
#print(solver.intervals[h][0])
assertAlmostEqual(solver.intervals[h][0][0][1], crossing, 2)
#print(solver.optimal_distributions[h][0])
assertEqual(solver.optimal_distributions[h][0], [base_distrib["ber09"], base_distrib["2ber02"]])

solver._solve_one(h, 11, -15, 15)

action_distrib = [mix_distrib["halfBer"], mix_distrib["quartZber"], base_distrib["Z"]]
assertEqual(solver.compute_distributions(h,11,-15,15), action_distrib)
bandit_solver = BetaPolicyBanditSolver(action_distrib)
bandit_solver.solve(-15, 15)
crossing = bandit_solver.crossings[0]
assertEqual(len(solver.intervals[h][11]), 2)
#print(solver.intervals[h])
assertAlmostEqual(solver.intervals[h][11][0][1], crossing, 2)

h-= 1

# Va planter à cause de la manière manuelle dont on a fait les choses. Les intervalles des états atteint ne sont pas tous calculés.
# intervals = solver.process_intervals(solver.intervals[h+1], 2*solver.accuracy)
# print(intervals)
# for (min,max) in intervals:
#     solver._solve_one(h, 12, min, max)
#     print(min,max)
#print(solver.intervals[0][12])

### Test all in one
solver_2 = MDPsolver(env, 5)
solver_2.solve()

#TODO : rajouter un test qui vérifie ces choses là. (une des changements d’action est valable pour h = 2, mais pas 1 ou 0)
print(f"h = 2 :")
print(f"intervals : {solver_2.intervals[2]}")
print(f"optimal_actions : {solver_2.optimal_actions[2]}")
print(f"optimal_distributions : {solver_2.optimal_distributions[2]}")
solution = [(-10, 0), (0, 1.252909971889041), (1.252909971889041, 1.7789430677680549), (1.7789430677680549, 10)]
print(solver_2.intervals[0][0])
for i in range(len(solution)):
    assertAlmostEqual(solver_2.intervals[0][0][i][0], solution[i][0], 2)
    assertAlmostEqual(solver_2.intervals[0][0][i][1], solution[i][1], 2)
print(solver_2.optimal_actions[0])
print("All tests passed!")