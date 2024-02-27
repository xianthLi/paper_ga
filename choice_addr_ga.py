from deap import base, creator, tools, algorithms
import random
import numpy as np

# 供应商、零售商和配送中心的示例数据
suppliers = np.random.rand(6, 2) * 100  # 6个供应商
supplier_caps = np.random.randint(100, 200, size=6)  # 供应商的供应上限
retailers = np.random.rand(27, 2) * 100  # 27个零售商
retailer_demands = np.random.randint(10, 50, size=27)  # 零售商的需求
distribution_centers = np.random.rand(8, 2) * 100  # 8个潜在的配送中心位置
dc_storage_caps = np.random.randint(300, 500, size=8)  # 配送中心的存储上限

"""

addr = [
    116.576495	34.25315
    116.937227	33.546092
    116.51433	34.43965
    116.372942	34.438832
    116.92036	34.221858
    116.883632	34.222028
]
"""

addr = [
    116.576495,	34.25315,
    116.937227,	33.546092,
    116.51433,	34.43965,
    116.372942,	34.438832,
    116.92036,	34.221858,
    116.883632,	34.222028,
]

suppliers = np.array(addr).reshape(6, 2)
supplier_caps = np.array([6000, 3000, 6000, 9000, 3000, 3000])

retailer_addr = [
    116.684534,	35.420201,
    116.794069,	34.408437,
    116.773566,	34.364616,
    116.820553,	34.366062,
    116.94893	,34.35374,
    116.599376,	34.255368,
    116.634658,	34.262554,
    116.655983,	34.290557,
    116.713804,	34.226293,
    116.764246,	34.263714,
    116.894875,	34.280581,
    116.950601,	34.268961,
    116.953905,	34.192724,
    116.954646,	34.196841,
    116.925065,	34.191559,
    116.940176,	34.186675,
    116.935447,	34.182147,
    116.937111,	34.172901,
    116.982374,	34.233714,
    116.970328,	34.208396,
    116.976552,	34.195135,
    116.96124	,34.185382,
    116.950777,	34.17284,
    116.611795,	34.134896,
    116.792416,	34.070618,
    117.026495,	34.042054,
    117.073778,	34.106461,
]
retailers = np.array(retailer_addr).reshape(27, 2)
demands = [
    500, 200, 200, 1000, 500, 200, 500, 500, 500, 1000, 200, 3000, 3000, 500, 1000, 200, 3000, 500, 200, 3000, 1000, 500, 3000, 1000, 1000, 1000, 500
]
retailers_demands = np.array(demands)

distribution_addr = [
    116.726581,	34.296618,
    116.724856, 34.29805,
    116.769412, 34.119163,
    117.044222, 34.151195,
    116.921729, 34.194562,
    116.971891, 34.201668,
    116.955973, 34.235041,
    116.910554, 34.313556,
]

distribution_addr = np.array(distribution_addr).reshape(8, 2)
distribution_caps = np.array([5000, 5000, 5000, 5000, 10000, 10000, 10000, 10000])


distance_cache = dict()


def get_distance_matrix(s_index, d_index):
    """
    计算两者从供应商到配送中心的曼哈顿距离

    Args:
        s_index (_type_): _description_
        d_index (_type_): _description_
    """
    if (s_index, d_index) in distance_cache:
        return distance_cache[(s_index, d_index)]
    else:
        distance = np.linalg.norm(suppliers[s_index] - distribution_addr[d_index], axis=1)
        distance_cache[(s_index, d_index)] = distance
        return distance

per_cost = 0.012


# 适应度函数
def fitness(individual):
    cost = 0
    # 计算供应商到零售商的成本
    s_caps = supplier_caps.copy()
    d_caps = distribution_caps.copy()


    # 计算供应商到零售商的成本
    current_s = 0
    for i, d in enumerate(individual):
        # 0 代表没被分配
        if d == 0:
            continue
        while d_caps[i] > 0:
            if s_caps[current_s] <= 0:
                current_s += 1
            if s_caps[current_s] > d_caps[i]:
                cost += get_distance_matrix(current_s, i) * d_caps[i] * per_cost
                d_caps[i] = 0
                s_caps[current_s] -= d_caps[i]
            else:
                cost += get_distance_matrix(current_s, i) * s_caps[current_s] * per_cost
                d_caps[i] -= s_caps[current_s]
                s_caps[current_s] = 0
    
    # 计算配送中心到零售商的成本
    for i, d in enumerate(individual):
        if d == 0:
            continue
        cost += get_distance_matrix(i, d) * per_cost
    return cost,

# 遗传算法设置
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=8)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# 遗传算法的主循环
def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 100, stats=stats, halloffame=hof, verbose=True)
    return hof[0]

if __name__ == "__main__":
    best_individual = main()
    print(f"Best individual: {best_individual}")
