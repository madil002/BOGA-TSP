from genetic_algorithm import setup_toolbox, generate_symmetric_distance_matrix
from tsp_generator import generate_tsp
from deap import algorithms, tools
import numpy as np
import random
from GPyOpt.methods import BayesianOptimization

def run_ga(distance_matrix, cxpb, mutpb, pop_size):
    num_cities = len(distance_matrix)
    toolbox = setup_toolbox(num_cities, distance_matrix, cxpb, mutpb, pop_size)
    random.seed(42)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print(f"Running GA with cxbp={cxpb}, mutpb={mutpb}, pop_size={pop_size}")
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=100, stats=stats, halloffame=hof, verbose=False)

    # Best individual
    best_fitness = hof.items[0].fitness.values[0]
    return best_fitness

def objective_function(params, distance_matrix):
    cxpb, mutpb, pop_size = params[0][0], params[0][1], int(params[0][2])
    fitness = -run_ga(distance_matrix, cxpb, mutpb, pop_size)
    print(f"Fitness achieved: {-fitness}")
    return fitness

def optimize_hyperparameters(file_path):
    distance_matrix = generate_tsp(file_path)
    print("Distance matrix generated successfully.")
    print("Initializing optimization with the following bounds:")
    bounds = [
        {'name': 'cxpb', 'type': 'continuous', 'domain': (0.5, 1.0)},
        {'name': 'mutpb', 'type': 'continuous', 'domain': (0.01, 0.2)},
        {'name': 'pop_size', 'type': 'discrete', 'domain': range(50, 200)}
    ]

    for bound in bounds:
        if isinstance(bound['domain'], range):
            print(f"  - {bound['name']} between {min(bound['domain'])} and {max(bound['domain'])}")
        else:
            print(f"  - {bound['name']} between {bound['domain'][0]} and {bound['domain'][1]}")

    obj_func = lambda params: objective_function(params, distance_matrix)

    optimizer = BayesianOptimization(f=obj_func, domain=bounds, model_type='GP', acquisition_type='EI', exact_feval=True, maximize=True)

    print("Starting optimization process...")
    optimizer.run_optimization(max_iter=10)
    
    print("Optimized Parameters:", optimizer.x_opt)
    print("Optimized Fitness:", -optimizer.fx_opt)

if __name__ == "__main__":
    file_path = "TSPs/test.tsp"
    optimize_hyperparameters(file_path)