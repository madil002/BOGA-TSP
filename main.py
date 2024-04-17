from genetic_algorithm import toolbox, generate_symmetric_distance_matrix, evalTSP
from deap import algorithms, tools
import random
import numpy as np

def main():
    num_cities = 20
    distance_matrix = generate_symmetric_distance_matrix(num_cities)

    toolbox.register("evaluate", evalTSP, distance_matrix=distance_matrix)
    
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
    final_pop, logbook, best = main()
    print("Best individual is:", best[0])
    print("Best fitness is:", best[0].fitness.values[0])
