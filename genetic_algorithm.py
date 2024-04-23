import random
import numpy as np
from deap import base, creator, tools, algorithms

# Define Minimization and Individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def generate_symmetric_distance_matrix(num_cities):
    np.random.seed(42)
    upper_triangle = np.triu(np.random.randint(10, 100, size=(num_cities, num_cities)), 1)
    matrix = upper_triangle + upper_triangle.T
    np.fill_diagonal(matrix, 0)
    print(matrix)
    return(matrix)

def setup_toolbox(num_cities, distance_matrix, cxpb, mutpb, pop_size):
    toolbox = base.Toolbox()

    # Attribute generator for creating a route
    toolbox.register("indices", random.sample, range(num_cities), num_cities)
    
    # Initialize individuals and population
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define the Fitness Function
    def evalTSP(individual):
        distance = sum(distance_matrix[individual[i - 1], individual[i]] for i in range(1, len(individual)))
        distance += distance_matrix[individual[-1], individual[0]]  # Complete the cycle
        return (distance,)
    
        # Genetic Operators
    toolbox.register("evaluate", evalTSP)
    toolbox.register("mate", tools.cxOrdered)  # Crossover function suitable for TSP
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)  # Shuffle mutation
    toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection

    return toolbox