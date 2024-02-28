
import numpy as np
import copy

from data import cos34, per_cost, per_stockout_cost, per_storage_cost, days, scenes
from data import  SupplyContainer, DemandsContainer, DistributionContainer

from draw import Draw


class GA:
    def __init__(self) -> None:
        self.supplier_containers = SupplyContainer()          # 供应商
        self.distribution_containers = DistributionContainer()  # 配送中心
        self.demands_container = DemandsContainer()            # 零售商

        self.distribution_supplier_matrix = None
        self.demand_distribution_matrix = []
        self.scene_list = []
        self.scene = None
        self.init_compute()

    def set_scene_list(self, scene_list):
        """
        设置场景列表
        Args:
            scene_list (_type_): _description_
        """
        self.scene_list = scene_list

    def set_scene(self, scene):
        self.scene = scene
        self.supplier_containers.set_caps_rate(scene["supplier_ratio"])
        self.distribution_containers.set_caps_rate(scene["distribution_ratio"])

    def init_compute(self):
        """
        初始化一些计算过程
        """
        # 计算每个配送中心 的最佳供应商
        for distribution in self.distribution_containers.distributions:
            distribution.compute_supplier_sort(self.supplier_containers.suppliers)
        
        # 计算每个零售商的最佳配送中心
        for demand in self.demands_container.demands:
            demand.compute_distribution_sort(self.distribution_containers.distributions)

        # 一个二维数组的矩阵，计算配送中心到供应商的距离
        self.distribution_supplier_matrix = []
        for distribution in self.distribution_containers.distributions:
            self.distribution_supplier_matrix.append([distribution.addr.distance(supplier.addr) for supplier in self.supplier_containers.suppliers])

        # 一个二维数组的矩阵，计算配送中心到零售商的距离
        for demand in self.demands_container.demands:
            self.demand_distribution_matrix.append([demand.addr.distance(distribution.addr) for distribution in self.distribution_containers.distributions])

    def fitness(self, individual):
        supplier_to_distribution, distribution_to_demand, db_storage, dd_stockout = self.get_plan(individual)
        tran_cost = self.compute_tran_cost(supplier_to_distribution, distribution_to_demand) * days
        construction_cost = self.compute_construction_cost(individual)
        storage_cost = sum(db_storage) * per_storage_cost * days
        stockout_cost = sum(dd_stockout) * per_stockout_cost * days
        all_cost = sum([tran_cost, construction_cost, storage_cost, stockout_cost])
        print("运输成本: ", tran_cost)
        print("建造成本: ", construction_cost)
        print("存储成本: ", storage_cost)
        print("缺货成本: ", stockout_cost)
        print("总成本: ", all_cost)
        return all_cost

    def fitness_by_scene(self, individual):
        """
        根据中断场景的加权，返回适应值
        individual 长度为 16， 前 8 个是选择什么配送中心，后面 8 个是根据对应的供应中心是否需要冗余备份

        Args:
            individual (_type_): _description_
        """
        cost = 0
        service_rate = 0
        for scene in self.scene_list:
            self.set_scene(scene)
            probability = scene["probability"]
            cost1, service_rate1 = self.fitness_with_backup(individual)
            cost += cost1 * probability
            service_rate += service_rate1 * probability
        return cost / 100, service_rate / 100

    def fitness_by_scene8(self, individual):
        """
        根据中断场景的加权，返回适应值
        individual 长度为 8， 前 8 个是选择什么配送中心，后面 8 个是根据对应的供应中心是否需要冗余备份

        Args:
            individual (_type_): _description_
        """
        cost = 0
        service_rate = 0
        for scene in self.scene_list:
            self.set_scene(scene)
            probability = scene["probability"]
            cost1, service_rate1 = self.fitness_2(individual)
            cost += cost1 * probability
            service_rate += service_rate1 * probability
        return cost / 100, service_rate / 100

    def fitness_with_backup(self, individual):
        """
        计算有冗余情况下的物流方案与成本

        Args:
            individual (_type_): _description_
        """
        backup = individual[8:]
        self.distribution_containers.add_caps_backup(backup)
        individual_distribution = individual[:8]
        cost, server_rate = self.fitness_2(individual_distribution)
        cost += self.distribution_containers.compute_backup_cost(backup, individual_distribution)
        return cost, server_rate

    def fitness_2(self, individual):
        """
        返回成本和服务率的评估值

        Args:
            individual (_type_): _description_
        """
        supplier_to_distribution, distribution_to_demand, db_storage, dd_stockout = self.get_plan(individual)
        tran_cost = self.compute_tran_cost(supplier_to_distribution, distribution_to_demand) * days
        construction_cost = self.compute_construction_cost(individual)
        storage_cost = sum(db_storage) * per_storage_cost * days
        stockout_cost = sum(dd_stockout) * per_stockout_cost * days
        all_cost = sum([tran_cost, construction_cost, storage_cost, stockout_cost])

        # 服务率
        service_rate = 1 - sum(dd_stockout) / sum(self.demands_container.caps)
        return all_cost, service_rate

    def compute_tran_cost(self, supplier_to_distribution, distribution_to_demand):
        """
        计算运输成本

        Args:
            supplier_to_distribution (_type_): _description_
            distribution_to_demand (_type_): _description_
        """
        cost = 0
        for (s_i, d_i), num in supplier_to_distribution.items():
            cost += num * self.distribution_supplier_matrix[d_i][s_i] * per_cost

        for (d_i, dd_i), num in distribution_to_demand.items():
            cost += num * self.demand_distribution_matrix[dd_i][d_i] * per_cost
        return cost

    def compute_construction_cost(self, individual):
        """
        计算建造成本

        Args:
            individual (_type_): _description_
        """
        cost = 0
        for d, c in zip(individual, self.distribution_containers.construction_cost):
            cost += d * c
        return cost

    def get_plan(self, individual):
        """
        计算这个计划

        Args:
            individual (_type_): _description_
        return:
            supplier_to_distribution (_type_): 供应商向配送中心的计划
            distribution_to_demand (_type_): 配送中心向零售商的计划
            db_storage: 配送中心的使用容量
            dd_caps: 零售商的缺货量
        """
        supplier_to_distribution = {} # 供应商向配送中心的计划
        distribution_to_demand = {} # 配送中心向零售商的计划
        # 供应商容量
        s_caps = self.supplier_containers.caps.copy()
        # 配送中心容量1
        db_caps1 = self.distribution_containers.caps.copy()
        # 配送中心容量2
        db_caps2 = self.distribution_containers.caps.copy()
        # 零售商容量
        dd_caps = self.demands_container.caps.copy()

        for i, d in enumerate(individual):
            if d == 0:
                continue
            # 供应商向配送中心的计划
            for s_i in self.distribution_containers.distributions[i].supplier_sort:
                if db_caps1[i] <= 0:
                    # 配送中心已经被填满了
                    break
                if s_caps[s_i] > 0:
                    goods = min(s_caps[s_i], db_caps1[i])
                    # 供应商发货
                    s_caps[s_i] -= goods
                    # 配送中心收货
                    db_caps1[i] -= goods
                    supplier_to_distribution[(s_i, i)] = goods

        for index, num in enumerate(db_caps1):
            if num > 0:
                db_caps2[index] -= num

        db_storage = copy.deepcopy(db_caps2)    # 配送中心的存储容量

        # 配送中心向零售商的计划, 需要填满零售商
        for dd_i, demand in enumerate(self.demands_container.demands):
            # 从最近的零售商那边取货
            for d_i in demand.distribution_sort:
                if individual[d_i] == 0:
                    # 该配送中心未被选中
                    continue
                if db_caps2[d_i] >= dd_caps[dd_i]:
                    goods = min(dd_caps[dd_i], db_caps2[d_i])
                    distribution_to_demand[(d_i, demand.id)] = goods
                    # 配送中心发货
                    db_caps2[d_i] -= goods
                    # 零售商收货
                    dd_caps[dd_i] -= goods
                    distribution_to_demand[(d_i, demand.id)] = goods
                    #零售商只能够接受一次货物
                    break
        
        # 配送中心向零售商的计划，不需要填满，考虑最近的贪婪原则
        for dd_i, demand in enumerate(self.demands_container.demands):
            # 从最近的零售商那边取货
            for d_i in demand.distribution_sort:
                if individual[d_i] == 0:
                    # 该配送中心未被选中
                    continue
                if db_caps2[d_i] > 0 and dd_caps[dd_i] > 0:
                    goods = min(dd_caps[dd_i], db_caps2[d_i])
                    distribution_to_demand[(d_i, demand.id)] = goods
                    # 配送中心发货
                    db_caps2[d_i] -= goods
                    # 零售商收货
                    dd_caps[dd_i] -= goods
                    distribution_to_demand[(d_i, demand.id)] = goods
                    #零售商只能够接受一次货物
                    break
        
        return supplier_to_distribution, distribution_to_demand, db_storage, dd_caps



def print_cost8(individual):
    """
    打印长度为 16 的个体计划

    Args:
        individual (_type_): _description_
    """
    ga = GA()
    ga.fitness(individual)

def print_individual_plan8(individual):
    """
    打印长度为 16 的个体计划

    Args:
        individual (_type_): _description_
    """
    ga = GA()
    ga.fitness(individual)

    for i, c in enumerate(individual):
        if c == 1:
            print("第{}个配送中心被选中".format(i+1))

    print("物流计划如下")

    p1, p2, c1, c2 = ga.get_plan(individual)
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
    print_cost8([1, 0, 1, 0, 0, 0, 1, 1])
    print_cost8([0, 0, 0, 0, 1, 1, 1, 0])
