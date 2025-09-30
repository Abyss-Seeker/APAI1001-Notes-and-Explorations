import random

def crossover(parent1, parent2):
    idx = random.randint(0, len(parent1) - 1)
    return parent1[:idx]+parent2[idx:] , parent2[:idx]+parent1[idx:]

print(crossover([1,2,3,4,5,6], [6,5,4,3,2,1]))