import numpy as np
import copy
import math

cos34 = math.cos(34.2 / 180 * math.pi)  # 计算维度距离的系数
per_cost = 0.012            # 运输成本
per_stockout_cost = 0.2      # 缺货成本
per_stock_safe_cost = 0.1   # 安全库存的成本
stock_safe_rate = 0.05      # 安全库存的比例
per_storage_cost = 0.1      # 存储成本
days = 1000                # 系统运行的天数

# 供应商容量
supply_caps = [6000, 3000, 6000, 9000, 3000, 3000]   
# 零售商需求
demand_caps = [500, 200, 200, 1000, 500, 200, 500, 500, 500, 1000, 200, 3000, 3000, 500, 1000, 200, 3000, 500, 200, 3000, 1000, 500, 3000, 1000, 1000, 1000, 500] 
# 物流中心容量
distribution_caps = [5000, 5000, 5000, 5000, 10000, 10000, 10000, 10000]
# 物流中心的建造成本
distribution_construct_cost = [i * 10000 for i in  [150, 150, 150, 150, 300, 300, 300, 300]]
# 物流中心的备用处理能力
distribution_backup = [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]
# 物流中心的备用成本造价
distribution_backup_cost = [i * 10000 for i in [30, 30, 30, 30, 30, 30, 30, 30]]

# 中断场景
scenes = [
    {
        "probability": 95,                  # 概率
        "supplier_ratio": [0] * 6,          # 供应商的中断比例
        "distribution_ratio": [0] * 8,      # 配送中心的中断比例
    },
    {
        "probability": 2,
        "supplier_ratio": [0.5, 0, 0, 0, 0, 0],
        "distribution_ratio": [0, 0, 0, 0, 0, 0, 0, 0]
    },
    {
        "probability": 1,
        "supplier_ratio": [0, 0, 0, 0, 0, 0],
        "distribution_ratio": [0.5, 0, 0, 0, 0.5, 0, 0, 0]
    },
    {
        "probability": 0.5,
        "supplier_ratio": [0.5, 0, 0, 0, 0, 0],
        "distribution_ratio": [0.5, 0, 0, 0, 0.5, 0, 0, 0]
    },
    {
        "probability": 0.5,
        "supplier_ratio": [0.5, 0.5, 0.5, 0, 0, 0],
        "distribution_ratio": [0, 0, 0, 0, 0, 0, 0, 0]
    },
    {
        "probability": 0.5,
        "supplier_ratio": [0, 0, 0, 0, 0, 0],
        "distribution_ratio": [1, 0.5, 0, 0, 1, 0.5, 0, 0]
    },
    {
        "probability": 0.5,
        "supplier_ratio": [0.5, 0.5, 0.5, 0, 0, 0],
        "distribution_ratio": [1, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5]
    },
]


class Addr:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def distance(self, addr):
        distance = abs(self.x - addr.x) * 111 * cos34  + abs(self.y - addr.y) * 111
        return distance

class Supplier:
    # 供应商
    def __init__(self, id, addr, cap) -> None:
        self.id = id
        self.addr = addr
        self.cap = cap

    def __repr__(self) -> str:
        return "(Supplier: {}, {}, {})".format(self.id, self.addr.x, self.addr.y)
    
    def __str__(self) -> str:
        return "(Supplier: {}, {}, {})".format(self.id, self.addr.x, self.addr.y)


class Demands:
    # 零售商
    def __init__(self, id, addr, cap) -> None:
        self.id = id
        self.addr = addr
        self.cap = cap
        # 距离该零售商的配送中心距离的排序
        self.distribution_sort = None

    def compute_distribution_sort(self, distributions):
        """
        计算配送中心的排序

        Args:
            distributions (_type_): _description_
        """
        distributions_copy = copy.deepcopy(distributions)
        distributions_copy.sort(key=lambda x: self.addr.distance(x.addr))
        self.distribution_sort = [i.id for i in distributions_copy]

    def __str__(self) -> str:
        return "(Demands: {}, {}, {})".format(self.id, self.addr.x, self.addr.y)

    def __repr__(self) -> str:
        return "(Demands: {}, {}, {})".format(self.id, self.addr.x, self.addr.y)

class Distribution:
    # 配送中心
    def __init__(self, id, addr, cap) -> None:
        self.id = id
        self.addr = addr
        self.cap = cap
        # 距离该配送中心的供应商距离的排序
        self.safe_stock_rate = 0
        self.supplier_sort = None

    def compute_supplier_sort(self, suppliers):
        """
        计算供应商的排序

        Args:
            supplier_addr (_type_): _description_
        """
        suppliers_copy = copy.deepcopy(suppliers)
        suppliers_copy.sort(key=lambda x: self.addr.distance(x.addr))
        self.supplier_sort = [i.id for i in suppliers_copy]

    def set_safe_stock(self, safe_stock_rate):
        """"
        设计安全成本的比例
        """
        self.safe_stock_rate = safe_stock_rate

    def __str__(self) -> str:
        return "(Distribution: {}, {}, {})".format(self.id, self.addr.x, self.addr.y)

    def __repr__(self) -> str:
        return "(Distribution: {}, {}, {})".format(self.id, self.addr.x, self.addr.y)


class SupplyContainer:
    # 供应商容器
    def __init__(self) -> None:
        self.num = 6
        self.addr = [
            116.576495,	34.25315,
            116.937227,	33.546092,
            116.51433,	34.43965,
            116.372942,	34.438832,
            116.92036,	34.221858,
            116.883632,	34.222028,
        ]
        self.addr = np.array(self.addr).reshape(self.num, 2)
        self.caps = supply_caps
        self.real_caps = copy.deepcopy(self.caps)
        self.suppliers = [Supplier(i, Addr(self.addr[i][0], self.addr[i][1]), self.caps[i]) for i in range(self.num)]

    def set_caps_rate(self, rate):
        """
        设置供应商的中断比例

        Args:
            rate (_type_): _description_
        """
        for i, supplier in enumerate(self.suppliers):
            supplier.cap = self.real_caps[i] * (1 - rate[i])
            self.caps[i] = supplier.cap

    def __str__(self) -> str:
        return "Suppliers: {}".format(self.suppliers)

    def __repr__(self) -> str:
        return "Suppliers: {}".format(self.suppliers)


class DemandsContainer:
    # 零售商容器
    def __init__(self) -> None:
        self.num = 27
        self.addr = [
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
        self.addr = np.array(self.addr).reshape(27, 2)
        self.caps =  demand_caps
        self.demands = [Demands(i, Addr(self.addr[i][0], self.addr[i][1]), self.caps[i]) for i in range(self.num)]

class DistributionContainer:
    # 配送中心容器

    def __init__(self) -> None:
        self.num = 8
        self.addr = [
            116.726581,	34.296618,
            116.724856, 34.29805,
            116.769412, 34.119163,
            117.044222, 34.151195,
            116.921729, 34.194562,
            116.971891, 34.201668,
            116.955973, 34.235041,
            116.910554, 34.313556,
        ]
        self.addr = np.array(self.addr).reshape(8, 2)
        self.caps = distribution_caps
        self.real_caps = copy.deepcopy(self.caps)
        self.distributions = [Distribution(i, Addr(self.addr[i][0], self.addr[i][1]), self.caps[i]) for i in range(self.num)]
        self.construction_cost = distribution_construct_cost
        self.backups = distribution_backup
        self.backups_cost = distribution_backup_cost

    def set_caps_rate(self, rate):
        """
        设置配送中心的中断比例

        Args:
            rate (_type_): _description_
        """
        for i, distribution in enumerate(self.distributions):
            distribution.cap = self.real_caps[i] * (1 - rate[i])
            self.caps[i] = distribution.cap

    def add_caps_backup(self, backup_list):
        """
        添加备用容量， 输入为一个0， 1 值的列表

        Args:
            backups (_type_): _description_
        """
        for i, distribution in enumerate(self.distributions):
            if backup_list[i] == 1:
                distribution.cap += self.backups[i]
                self.caps[i] = distribution.cap

    def compute_backup_cost(self, backup_list, construct_list):
        cost = 0
        for index, num in enumerate(backup_list):
            if num == 1 and construct_list[index] == 1:
                cost += self.backups_cost[index]
        return cost
