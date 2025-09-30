import math, random
import numpy as np


"""
PSEUDOCODE
# 遗传算法求解八皇后问题 - 主流程
POPULATION_SIZE = 100  # 种群大小
MAX_GENERATIONS = 10000  # 最大进化代数
MUTATION_RATE = 0.1  # 变异概率

function MAIN():
    population = INITIALIZE_POPULATION(POPULATION_SIZE)  # 初始化种群
    for generation in range(MAX_GENERATIONS):
        fitness_scores = [FITNESS(chromosome) for chromosome in population]  # 计算适应度
        if max(fitness_scores) == 28:  # 找到最优解（28表示无冲突的皇后对数量）
            best_index = fitness_scores.index(28)
            PRINT_SOLUTION(population[best_index])
            return
        new_population = []
        for _ in range(POPULATION_SIZE // 2):  # 生成新一代
            parent1 = SELECT(population, fitness_scores)  # 选择父代1
            parent2 = SELECT(population, fitness_scores)  # 选择父代2
            child1, child2 = CROSSOVER(parent1, parent2)  # 交叉产生子代
            child1 = MUTATE(child1, MUTATION_RATE)  # 变异
            child2 = MUTATE(child2, MUTATION_RATE)
            new_population.append(child1)
            new_population.append(child2)
        population = new_population  # 更新种群
    print("未找到最优解")
"""


# AI written fitness check function
def fitness_function(individual):
    """p
    计算八皇后问题中一个个体的适应度
    适应度 = 不互相攻击的皇后对的数量
    理想解（无冲突）的适应度为28（因为8个皇后共有C(8,2)=28对）

    参数:
        individual: 列表，表示一个染色体，长度为8
                  索引代表列号，值代表该列皇后所在的行号

    返回:
        int: 适应度分数（不互相攻击的皇后对的数量）
    """
    non_attacking_pairs = 0  # 不互相攻击的皇后对计数

    # 遍历所有不同的皇后对（避免重复计算）
    for i in range(8):
        for j in range(i + 1, 8):
            # 检查两个皇后是否在同一行（行冲突）
            if individual[i] == individual[j]:
                continue  # 在同一行，会互相攻击

            # 检查两个皇后是否在同一对角线（对角线冲突）
            # 对角线的判断：|列差| == |行差| 则在同一对角线
            col_diff = j - i  # 列差
            row_diff = abs(individual[i] - individual[j])  # 行差的绝对值

            if col_diff != row_diff:
                non_attacking_pairs += 1  # 既不共行也不共对角线，皇后对不冲突

    return non_attacking_pairs


def initialize_population(population_size):
    return [[random.randint(0,7) for j in range(8)] for i in range(population_size)]


def select(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_p = [i / total_fitness for i in fitness_scores]
    return population[np.random.choice(len(population), p=selection_p)]

def crossover(parent1, parent2):
    idx = random.randint(0, len(parent1) - 1)
    return parent1[:idx]+parent2[idx:] , parent2[:idx]+parent1[idx:]

def mutate(solution, mutation_rate):
    # O(1) - 100% swap
    # idx1, idx2 = random.sample(range(len(solution)), 2)  # 确保选择两个不同的索引
    # solution[idx1], solution[idx2] = solution[idx2], solution[idx1]  # 交换两个位置的值[1,2](@ref)

    # O(n) - SNP
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = random.randint(0, 7)
    return solution

def GeneticAlgorithm(POPULATION_SIZE = 100, MAX_GENERATIONS = 1000, MUTATION_RATE = 0.1):
    population = initialize_population(POPULATION_SIZE)
    for generation in range(MAX_GENERATIONS):
        fitness_scores = [fitness_function(individual) for individual in population]
        if max(fitness_scores) == 28:
            print(population[fitness_scores.index(max(fitness_scores))])
            return population[fitness_scores.index(max(fitness_scores))]
        new_population = []
        for i in range(len(population)//2):
            parent1 = select(population, fitness_scores)
            parent2 = select(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, MUTATION_RATE)
            child2 = mutate(child2, MUTATION_RATE)
            new_population.append(child1)
            new_population.append(child2)
        population = new_population
    print(f"Did not find optimal solution. Best solution is with {max(fitness_scores) = }, being f{population[fitness_scores.index(max(fitness_scores))]}")

GeneticAlgorithm()