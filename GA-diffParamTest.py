import math
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


# 原有的遗传算法函数保持不变
def fitness_function(individual):
    """计算八皇后问题的适应度"""
    non_attacking_pairs = 0
    for i in range(8):
        for j in range(i + 1, 8):
            if individual[i] == individual[j]:
                continue
            col_diff = j - i
            row_diff = abs(individual[i] - individual[j])
            if col_diff != row_diff:
                non_attacking_pairs += 1
    return non_attacking_pairs


def initialize_population(population_size):
    return [[random.randint(0, 7) for _ in range(8)] for _ in range(population_size)]


def select(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_p = [i / total_fitness for i in fitness_scores]
    return population[np.random.choice(len(population), p=selection_p)]


def crossover(parent1, parent2):
    idx = random.randint(0, len(parent1) - 1)
    return parent1[:idx] + parent2[idx:], parent2[:idx] + parent1[idx:]


def mutate(solution, mutation_rate):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = random.randint(0, 7)
    return solution


# 修改后的遗传算法，支持提前终止和适应度记录
def GeneticAlgorithm(POPULATION_SIZE=100, MAX_GENERATIONS=1000, MUTATION_RATE=0.1):
    """
    修改后的遗传算法，一旦找到适应度28的解，后续所有代数的适应度都记为28
    返回：适应度历史列表和总运行时间
    """
    start_time = time.time()
    population = initialize_population(POPULATION_SIZE)
    fitness_history = []  # 记录每一代的最佳适应度
    found_optimal = False  # 标记是否找到最优解
    optimal_generation = MAX_GENERATIONS  # 记录找到最优解的代数

    for generation in range(MAX_GENERATIONS):
        fitness_scores = [fitness_function(individual) for individual in population]
        current_best_fitness = max(fitness_scores)

        # 检查是否找到最优解
        if not found_optimal and current_best_fitness == 28:
            found_optimal = True
            optimal_generation = generation
            print(f"在第 {generation} 代找到最优解！")

        # 如果已经找到最优解，后续所有代适应度记为28
        if found_optimal:
            fitness_history.append(28)
        else:
            fitness_history.append(current_best_fitness)

        # 如果找到最优解，可以提前终止循环以节省时间（可选）
        # 但为了公平比较时间，我们继续运行完整代
        # if found_optimal and generation > optimal_generation + 10:  # 多运行10代确保稳定
        #     break

        # 遗传操作
        new_population = []
        for i in range(len(population) // 2):
            parent1 = select(population, fitness_scores)
            parent2 = select(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, MUTATION_RATE)
            child2 = mutate(child2, MUTATION_RATE)
            new_population.append(child1)
            new_population.append(child2)
        population = new_population

    end_time = time.time()
    total_time = end_time - start_time

    return fitness_history, total_time, found_optimal, optimal_generation


# 定义12组参数配置（种群大小20、50、100、200，每种3个变异率）
parameter_sets = [
    {"POPULATION_SIZE": 20, "MUTATION_RATE": 0.01, "color": "red", "label": "PS=20, MR=0.01"},
    {"POPULATION_SIZE": 20, "MUTATION_RATE": 0.1, "color": "blue", "label": "PS=20, MR=0.1"},
    {"POPULATION_SIZE": 20, "MUTATION_RATE": 0.3, "color": "green", "label": "PS=20, MR=0.3"},
    {"POPULATION_SIZE": 50, "MUTATION_RATE": 0.01, "color": "orange", "label": "PS=50, MR=0.01"},
    {"POPULATION_SIZE": 50, "MUTATION_RATE": 0.1, "color": "purple", "label": "PS=50, MR=0.1"},
    {"POPULATION_SIZE": 50, "MUTATION_RATE": 0.3, "color": "brown", "label": "PS=50, MR=0.3"},
    {"POPULATION_SIZE": 100, "MUTATION_RATE": 0.01, "color": "pink", "label": "PS=100, MR=0.01"},
    {"POPULATION_SIZE": 100, "MUTATION_RATE": 0.1, "color": "gray", "label": "PS=100, MR=0.1"},
    {"POPULATION_SIZE": 100, "MUTATION_RATE": 0.3, "color": "olive", "label": "PS=100, MR=0.3"},
    {"POPULATION_SIZE": 200, "MUTATION_RATE": 0.01, "color": "cyan", "label": "PS=200, MR=0.01"},
    {"POPULATION_SIZE": 200, "MUTATION_RATE": 0.1, "color": "magenta", "label": "PS=200, MR=0.1"},
    {"POPULATION_SIZE": 200, "MUTATION_RATE": 0.3, "color": "teal", "label": "PS=200, MR=0.3"}
]

# 测试参数
MAX_GENERATIONS = 1000
NUM_RUNS = 3  # 每组参数运行3次


def run_enhanced_parameter_testing():
    """增强的参数测试函数，支持多次运行和提前终止记录"""

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 存储运行时间和其他统计数据
    time_data = []
    convergence_data = []  # 存储收敛相关信息

    # 使用tqdm显示进度条
    for params in tqdm(parameter_sets, desc="测试参数组"):
        all_runs_fitness = []  # 存储三次运行的适应度历史
        all_runs_times = []  # 存储三次运行的时间
        optimal_found_counts = 0  # 统计找到最优解的次数
        optimal_generations = []  # 记录找到最优解的代数

        # 对每组参数运行3次
        for run in range(NUM_RUNS):
            fitness_history, total_time, found_optimal, optimal_gen = GeneticAlgorithm(
                POPULATION_SIZE=params["POPULATION_SIZE"],
                MAX_GENERATIONS=MAX_GENERATIONS,
                MUTATION_RATE=params["MUTATION_RATE"]
            )

            all_runs_fitness.append(fitness_history)
            all_runs_times.append(total_time)

            if found_optimal:
                optimal_found_counts += 1
                optimal_generations.append(optimal_gen)

        # 计算平均适应度（跨3次运行）
        avg_fitness = np.mean(all_runs_fitness, axis=0)
        avg_time = np.mean(all_runs_times)

        # 记录统计数据
        time_data.append({
            "label": params["label"],
            "population_size": params["POPULATION_SIZE"],
            "mutation_rate": params["MUTATION_RATE"],
            "avg_time": avg_time,
            "color": params["color"],
            "optimal_rate": optimal_found_counts / NUM_RUNS,  # 找到最优解的比例
            "avg_optimal_generation": np.mean(optimal_generations) if optimal_generations else MAX_GENERATIONS
        })

        # 在第一个子图上绘制适应度曲线
        generations = range(1, MAX_GENERATIONS + 1)
        line = ax1.plot(generations, avg_fitness,
                        color=params["color"],
                        label=params["label"],
                        linewidth=2)[0]

        # 标记找到最优解的平均代数
        if optimal_found_counts > 0:
            avg_gen = time_data[-1]["avg_optimal_generation"]
            ax1.axvline(x=avg_gen, color=params["color"], linestyle='--', alpha=0.5)
            ax1.text(avg_gen, 25, f'{avg_gen:.0f}',
                     color=params["color"], fontsize=8, ha='center')

    # 设置第一个子图（适应度曲线）属性
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Max Optimal Fitness', fontsize=12)
    ax1.set_title('GA Performance on 8-Queens (3-run Average)\nVertical lines show average convergence generation',
                  fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=28, color='r', linestyle='-', alpha=0.7, label='Optimal Solution (28)')
    ax1.set_ylim(20, 29)
    ax1.set_xlim(0, MAX_GENERATIONS)

    # 在第二个子图上绘制运行时间分析
    population_sizes = sorted(set([data["population_size"] for data in time_data]))
    mutation_rates = sorted(set([data["mutation_rate"] for data in time_data]))

    bar_width = 0.25
    x_pos = np.arange(len(population_sizes))

    # 为每个变异率创建一组条形
    for i, mutation_rate in enumerate(mutation_rates):
        times_for_mutation = [data["avg_time"] for data in time_data if data["mutation_rate"] == mutation_rate]
        colors_for_mutation = [data["color"] for data in time_data if data["mutation_rate"] == mutation_rate]
        optimal_rates = [data["optimal_rate"] for data in time_data if data["mutation_rate"] == mutation_rate]

        bars = ax2.bar(x_pos + i * bar_width, times_for_mutation, bar_width,
                       label=f'MR={mutation_rate}', color=colors_for_mutation)

        # 在条形上添加数值标签和成功率
        for j, (bar, time_val, success_rate) in enumerate(zip(bars, times_for_mutation, optimal_rates)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                     f'{time_val:.2f}s\n({success_rate * 100:.0f}%)',
                     ha='center', va='bottom', fontsize=8)

    # 设置第二个子图属性
    ax2.set_xlabel('Population Size', fontsize=12)
    ax2.set_ylabel('Average Running Time (seconds)', fontsize=12)
    ax2.set_title('Running Time and Success Rate Analysis\n(Percentage shows optimal solution finding rate)',
                  fontsize=14)
    ax2.set_xticks(x_pos + bar_width)
    ax2.set_xticklabels(population_sizes)
    ax2.legend(title='Mutation Rate')
    ax2.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig('ga_8queens_enhanced_test.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印详细的统计分析
    print("\n" + "=" * 60)
    print("                   详细统计分析报告")
    print("=" * 60)
    print("种群大小 | 变异率 | 平均时间(秒) | 成功率(%) | 平均收敛代数")
    print("-" * 60)

    for data in time_data:
        success_rate = data["optimal_rate"] * 100
        avg_conv_gen = data["avg_optimal_generation"]
        print(f"{data['population_size']:^8} | {data['mutation_rate']:^6} | {data['avg_time']:^11.2f} | "
              f"{success_rate:^8.1f}% | {avg_conv_gen:^13.0f}")

    # 最佳参数推荐
    best_by_success = max(time_data, key=lambda x: x["optimal_rate"])
    best_by_time = min(time_data, key=lambda x: x["avg_time"])
    best_balanced = max(time_data, key=lambda x: x["optimal_rate"] / x["avg_time"] if x["avg_time"] > 0 else 0)

    print("\n" + "=" * 60)
    print("                   最佳参数推荐")
    print("=" * 60)
    print(f"最高成功率: {best_by_success['label']} (成功率: {best_by_success['optimal_rate'] * 100:.1f}%)")
    print(f"最快求解: {best_by_time['label']} (平均时间: {best_by_time['avg_time']:.2f}秒)")
    print(
        f"最佳平衡: {best_balanced['label']} (效率指数: {best_balanced['optimal_rate'] / best_balanced['avg_time']:.3f})")


# 运行增强测试
if __name__ == "__main__":
    run_enhanced_parameter_testing()