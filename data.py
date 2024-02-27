import numpy as np
import copy
import math

cos34 = math.cos(34.2 / 180 * math.pi)  # 计算维度距离的系数
per_cost = 0.012        # 运输成本
per_stockout_cost = 1  # 缺货成本
per_storage_cost = 0.1  # 存储成本
days = 1000             # 系统运行的天数

# 供应商容量
supply_caps = [6000, 3000, 6000, 9000, 3000, 3000]   
# 零售商需求
demand_caps = [500, 200, 200, 1000, 500, 200, 500, 500, 500, 1000, 200, 3000, 3000, 500, 1000, 200, 3000, 500, 200, 3000, 1000, 500, 3000, 1000, 1000, 1000, 500] 
# 物流中心容量
distribution_caps = [5000, 5000, 5000, 5000, 10000, 10000, 10000, 10000]
# 物流中心的建造成本
distribution_construct_cost = [150, 150, 150, 150, 300, 300, 300, 300]


class Addr:
    def __init__(self, x, y) -> None:
        self.x = int(x)
        self.y = int(y)

    def distance(self, addr):
        distance = int((abs(self.x - addr.x) * 111 * cos34  + abs(self.y - addr.y) * 111))
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
        print("self.distribution_sort", self.distribution_sort)

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
        print("self.supplier_sort", self.supplier_sort)
        print("sort supplier: ", suppliers_copy)
        print("sort distance: ", [self.addr.distance(s.addr) for s in suppliers_copy])

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
        self.suppliers = [Supplier(i, Addr(self.addr[i][0], self.addr[i][1]), self.caps[i]) for i in range(self.num)]

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
        self.distributions = [Distribution(i, Addr(self.addr[i][0], self.addr[i][1]), self.caps[i]) for i in range(self.num)]
        self.construction_cost = distribution_construct_cost
