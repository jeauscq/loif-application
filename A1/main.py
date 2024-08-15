# Instantiate required libraries
from deap_modified import base, creator, tools, algorithms
import array
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def evaluate_individual(individual):
    """
    The specific fitness function for the aimed problem

    individual: it is an array of floats that contains variables Wing longitude
    (L), Wing width (W), Wing Depth (T) and attack angle (alpha).

    score: it is the evaluation of the individual.
    """
    # Define costants
    rho = 1.225  # Air density
    v = 250  # Air velocity

    L, W, T, alpha = individual
    S = L * W  # Wing area
    CL = 2 * np.pi * alpha * (W / L)  # Sustentation coefficient
    CD = CL**2 / (np.pi * (L / W))  # Drag coefficient
    lift = 0.5 * rho * v**2 * S * CL  # Sustentation force
    weight = S * T * 100  # Simplified weight model

    # If the weight or force are out of the defined boundaries,
    # the score is penalized and returns zero
    if weight > 10000 or lift < 50000:
        return float(0),
    score = CL / CD
    return score,


def ea_1dgan():
    """
    Runs a customized version of a simple evolutory algorithm in which a 1
    dimensional GAN is used to create the individuals from the second
    generation until the last one. This GAN substitutes the well known
    crossover and mutation steps.

    hof: a list of the n best individual produced.
    """
    # Hyperparameters are defined
    n_gen = 8  # Number of generations
    pop_size = 350  # Population size
    mutation_gen = 0.01  # Mutation probability

    # Toolbox definition
    toolbox = base.Toolbox()
    # The fitness is defined. Weights are +1, it is a maximization function
    creator.create('Fitness_funct', base.Fitness, weights=(1.0,))
    # Each individual is array type, with a fitness and typecode argument
    creator.create('Individuo', array.array,
                   fitness=creator.Fitness_funct, typecode='f')
    # Define fitness function
    toolbox.register('evaluate', evaluate_individual)
    # Define how to create each gene inside of an individual
    toolbox.register('LLL', random.uniform, a=10, b=30)
    toolbox.register('WWW', random.uniform, a=1, b=5)
    toolbox.register('TTT', random.uniform, a=0.1, b=1)
    toolbox.register('alpha', random.uniform, a=0, b=15)
    # Puts together each indiviual
    toolbox.register('individuo_gen', tools.initCycle, creator.Individuo,
                     (toolbox.LLL, toolbox.WWW, toolbox.TTT, toolbox.alpha),
                     n=1)
    # Puts together the whole population function
    toolbox.register("Poblacion", tools.initRepeat, list,
                     toolbox.individuo_gen, n=pop_size)
    # Creates the initial population
    popu = toolbox.Poblacion()
    # Mate method: one point crossover
    toolbox.register('mate', tools.cxOnePoint)
    # Gaussian method for mutation
    toolbox.register('mutate', tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
    # Defines Tournament as selection method with tournament size of 5
    toolbox.register('select', tools.selTournament, tournsize=5, k=pop_size)
    # Defines the second selection method as Best individuals
    toolbox.register('select2', tools.selBest, k=pop_size, n=pop_size)
    # Hall of Fame: present the best 10 individuals
    hof = tools.HallOfFame(10)
    # General statistics of each generation
    stats = tools.Statistics(lambda indiv: indiv.fitness.values)
    stats.register('avg', np.mean)  # Average of generation
    stats.register('std', np.std)  # Standard deviation in a single generation
    stats.register('min', np.min)  # Minimum fitnes in a generation
    stats.register('max', np.max)  # Maximum fitnes in a generation
    # Executes the whole algorithm
    popu, logbook = algorithms.eaSimple(popu, toolbox, cxpb=0.5,
                                        mutpb=mutation_gen, ngen=n_gen,
                                        stats=stats, halloffame=hof,
                                        verbose=True)
    # Show results
    print('---------------------------')
    print(logbook)
    # Set IEEE style parameters
    mpl.rcParams['figure.figsize'] = (10, 6)  # Adjusted figure size
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['lines.linewidth'] = 2  # Line width
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.5
    generation = logbook.select("gen")
    # Y axis parameter:
    avg = logbook.select("avg")
    std = logbook.select("std")
    # minn = logbook.select("min")
    maxx = logbook.select("max")
    plt.figure(figsize=(10, 6))
    plt.plot(generation, avg, label="Average fitness", color='blue')
    plt.plot(generation, std, label="Standard deviation", color='orange')
    # plt.plot(generation, minn, label="Worst individual", color='red')
    plt.plot(generation, maxx, label="Best individual", color='green')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.legend()
    plt.title("Algorithm's performance through Generations")
    plt.show()


if __name__ == "__main__":
    ea_1dgan()
