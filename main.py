from genetic_algorithm import setup_toolbox, generate_symmetric_distance_matrix
from tsp_generator import generate_tsp
from deap import algorithms, tools
import numpy as np
import random

def main(distance_matrix, cxpb, mutpb, pop_size):
    num_cities = len(distance_matrix)
    toolbox = setup_toolbox(num_cities, distance_matrix, cxpb, mutpb, pop_size)

    random.seed(42)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof

if __name__ == "__main__":
    distance_matrix = generate_tsp("TSPs/wi29.tsp")
    cxpb = 0.894698890
    mutpb = 0.0654135104
    pop_size = 150
    final_pop, logbook, best = main(distance_matrix, cxpb, mutpb, pop_size)
    print("Best individual is:", best[0])
    print("Best fitness is:", best[0].fitness.values[0])