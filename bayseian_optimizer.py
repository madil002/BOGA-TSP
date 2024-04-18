from genetic_algorithm import setup_toolbox, generate_symmetric_distance_matrix
from deap import algorithms, tools
import numpy as np
import random
from GPyOpt.methods import BayesianOptimization

def run_ga(num_cities, cxpb, mutpb, pop_size):
    distance_matrix = generate_symmetric_distance_matrix(num_cities)
    toolbox = setup_toolbox(num_cities, distance_matrix, cxpb, mutpb, pop_size)
    random.seed(42)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=100, stats=stats, halloffame=hof, verbose=False)

    # Best individual
    best_fitness = hof.items[0].fitness.values[0]
    return best_fitness

def objective_function(params, num_cities):
    cxpb, mutpb, pop_size = params[0][0], params[0][1], int(params[0][2])
    return -run_ga(num_cities=num_cities, cxpb=cxpb, mutpb=mutpb, pop_size=pop_size)

def optimize_hyperparameters(num_cities):
    bounds = [
        {'name': 'cxpb', 'type': 'continuous', 'domain': (0.5, 1.0)},
        {'name': 'mutpb', 'type': 'continuous', 'domain': (0.01, 0.2)},
        {'name': 'pop_size', 'type': 'discrete', 'domain': range(50, 200)}
    ]

    obj_func = lambda params: objective_function(params, num_cities)

    optimizer = BayesianOptimization(f=obj_func, domain=bounds, model_type='GP', acquisition_type='EI', exact_feval=True, maximize=True)
    optimizer.run_optimization(max_iter=100)
    print("Optimized Parameters:", optimizer.x_opt)
    print("Optimized Fitness:", -optimizer.fx_opt)

if __name__ == "__main__":
    num_cities = 20
    optimize_hyperparameters(num_cities)