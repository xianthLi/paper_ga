from deap import base, creator, tools, algorithms
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools

from ga import GA
from data import scenes

def generate_population(toolbox, n=256, length=8):
    # 生成所有可能的 8 位二进制组合
    all_combinations = list(itertools.product([0, 1], repeat=length))
    # 转换为个体
    print(all_combinations[:])
    population = [toolbox.individual(lambda: gene) for gene in all_combinations]
    return population

# 主程序
def main(mu=256, ngen=50, cxpb=0.9):
    random.seed(64)
    MU = 200  # 种群大小
    NGEN = 2  # 代数
    CXPB = 0.9

    ga = GA()
    ga.set_scene_list(scenes)

    # 定义问题：双目标最小化
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    # creator.create("Individual", list, fitness=creator.F)
    size = 8
    # 初始化
    toolbox = base.Toolbox()
    # toolbox.register("attr_float", random.randint, 0, 1)
    toolbox.register("individual", tools.initIterate, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", ga.fitness_by_scene8)
    # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=-10, up=10, eta=20.0)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    # toolbox.register("mutate", tools.mutPolynomialBounded, low=-10, up=10, eta=20.0, indpb=0.1)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.02)
    toolbox.register("select", tools.selNSGA2)


    pop = generate_population(toolbox)
    # pop = toolbox.population(n=MU)
    print(pop)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=MU, cxpb=CXPB, mutpb=1-CXPB,
                              ngen=NGEN, stats=stats, halloffame=hof, verbose=True)

    for ind in hof:
        print("Best individual is: %s\nwith fitness: %s" % (ind, ind.fitness))
    plot_pareto_front(hof)
    return pop, stats, hof

def plot_pareto_front(hof):
    front = np.array([ind.fitness.values for ind in hof])
    plt.scatter(front[:,0], front[:,1], c="r")
    for x, y in front:
        plt.annotate(f'({x:.4f}, {y:.4f})', # 这里使用了格式化字符串保留四位小数
                     (x, y),                 # 这是标注文本的坐标位置
                     textcoords="offset points", # 使用偏移
                     xytext=(0,10),         # 每个标签上移 10 个单位，避免被点覆盖
                     ha='center')           # 水平居中对齐
    plt.axis("tight")
    plt.xlabel("cost")
    plt.ylabel("server rate")
    plt.title("Pareto Front")
    plt.show()

def plot_pareto_front(hof):
    front = np.array([ind.fitness.values for ind in hof])
    plt.scatter(front[:,0], front[:,1], c="r")
    for x, y in front:
        plt.annotate(f'({x:.4f}, {y:.4f})', # 这里使用了格式化字符串保留四位小数
                     (x, y),                 # 这是标注文本的坐标位置
                     textcoords="offset points", # 使用偏移
                     xytext=(0,10),         # 每个标签上移 10 个单位，避免被点覆盖
                     ha='center')           # 水平居中对齐
    plt.axis("tight")
    plt.xlabel("cost")
    plt.ylabel("server rate")
    plt.title("Pareto Front")
    plt.show()

if __name__ == "__main__":
    main(256, 50, 0.9)

