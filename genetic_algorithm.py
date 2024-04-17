import random
import numpy as np
from deap import base, creator, tools

def generate_symmetric_distance_matrix(num_cities):
    np.random.seed(42)

    upper_triangle = np.triu(np.random.randint(10, 100, size=(num_cities, num_cities)), 1)  # Create an upper triangle of random integers
    matrix = upper_triangle + upper_triangle.T  # Mirror the upper triangle to the lower triangle to ensure symmetry

    np.fill_diagonal(matrix, 0) # Set the diagonal to zero
    print(matrix)
    return matrix

# Define Fitness and Individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #Minimization
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator for creating a route
toolbox.register("indices", random.sample, range(20), 20)

# Initialize individuals and population
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the Fitness Function
def evalTSP(individual, distance_matrix):
    distance = sum(distance_matrix[individual[i - 1], individual[i]] for i in range(1, len(individual)))
    distance += distance_matrix[individual[-1], individual[0]]  # Complete the cycle
    return (distance,)

# Genetic Operators
toolbox.register("evaluate", evalTSP)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)