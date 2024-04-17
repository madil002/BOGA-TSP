from genetic_algorithm import setup_toolbox, generate_symmetric_distance_matrix
from deap import algorithms, tools
import numpy as np
import random

def main(num_cities):
    distance_matrix = generate_symmetric_distance_matrix(num_cities)
    toolbox = setup_toolbox(num_cities, distance_matrix)

    random.seed(42)
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof

if __name__ == "__main__":
    num_cities = 10
    final_pop, logbook, best = main(num_cities)
    print("Best individual is:", best[0])
    print("Best fitness is:", best[0].fitness.values[0])