import numpy as np

from draw import Draw
from ga import GA
from data import scenes

def get_individual_by_num(i):
    """
    将一个数字变成 2 进制

    Args:
        i (_type_): _description_
    """
    binary = "{0:08b}".format(i)
    result = []
    for i in str(binary):
        result.append(int(i))
    return result


def print_individual_info(individual):
    """
    打印长度为 16 的个体计划

    Args:
        individual (_type_): _description_
    """
    ga = GA()
    ga.set_scene_list(scenes)
    print(ga.fitness_by_scene(individual))


def main():
    ga = GA()
    fitness = []
    for i in range(255):
        fitness.append(ga.fitness(get_individual_by_num(i)))
    print("*" * 20 + "result" + "*" * 20)
    print("各个方案的成本: ", fitness)
    print("最低成本: ", min(fitness))
    method = get_individual_by_num(fitness.index(min(fitness)))
    print("最优方案: ", method)
    print("选中了 {} 个配送中心".format(sum(method)))
    for i, c in enumerate(method):
        if c == 1:
            print("第{}个配送中心被选中".format(i+1))

    print()
    print("物流计划如下")

    p1, p2, c1, c2 = ga.get_plan(method)
    print(">>>供应商供货详情")
    for (s_i, d_i) , num in p1.items():
        if num == 0:
            continue
        print("{} 号供应商给 {} 号物流中心供货 {} kg".format(s_i+1, d_i+1, num))

    print(">>>零售商供货详情")
    for (d_i, dd_i), num in p2.items():
        if num == 0:
            continue
        print("{} 号物流中心给 {} 号零售商供货 {} kg".format(d_i+1, dd_i+1, num))

    print(">>>供应商缺货详情:")
    for i, num in enumerate(c2):
        print("{} 号供应商缺货 {} kg".format(i+1, num))

    draw = Draw(ga.supplier_containers, ga.distribution_containers, ga.demands_container, p1, p2)

    draw.run()

if __name__ == "__main__":
    main()
    # print_individual_info([1]*16)