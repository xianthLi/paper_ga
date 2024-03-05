import matplotlib.pyplot as plt
import numpy as np

class Draw:

    def __init__(self, suppliers, distributions, demands, supplier_to_distribution, distribution_to_demand) -> None:
        self.suppliers = suppliers
        self.distributions = distributions
        self.demands = demands
        self.supplier_to_distribution = supplier_to_distribution
        self.distribution_to_demand = distribution_to_demand

    def run(self):
        self.fig, self.ax = plt.subplots()

        self.draw_point()
        self.draw_path()
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Supply Chain Transportation Map')
        plt.grid(True)
        plt.show()

    def draw_point(self):
        for i, (x, y) in enumerate(self.suppliers.addr):
            self.ax.plot(x, y, '^b', markersize=10)
            self.ax.text(x, y, f'S{i+1}', fontsize=9, ha='right')
        
        for i, (x, y) in enumerate(self.distributions.addr):
            self.ax.plot(x, y, 'sg', markersize=8)
            self.ax.text(x, y, f'L{i+1}', fontsize=9, ha='right')
        
        for i, (x, y) in enumerate(self.demands.addr):
            self.ax.plot(x, y, 'or', markersize=6)
            self.ax.text(x, y, f'R{i+1}', fontsize=9, ha='right')

    def draw_path(self):
        """
        绘制相关路径
        """
        for (start, to), num in self.supplier_to_distribution.items():
            s_point = self.suppliers.addr[start]
            to_point = self.distributions.addr[to]
            self.ax.arrow(s_point[0], s_point[1], to_point[0] - s_point[0], to_point[1] - s_point[1], head_width=0.01, head_length=0.01, fc='black', ec='black')
        for (to, end), num in self.distribution_to_demand.items():
            to_point = self.distributions.addr[to]
            end_point = self.demands.addr[end]
            self.ax.arrow(to_point[0], to_point[1], end_point[0] - to_point[0], end_point[1] - to_point[1], head_width=0.01, head_length=0.01, fc='purple', ec='purple')

def draw_point(hof1, hof2, print_word=False):

    front = np.array([ind.fitness.values for ind in hof1])
    plt.scatter(front[:,0], front[:,1], c="r")

    if print_word:
        for x, y in front:
            plt.annotate(f'({x:.4f}, {y:.4f})', # 这里使用了格式化字符串保留四位小数
                        (x, y),                 # 这是标注文本的坐标位置
                        textcoords="offset points", # 使用偏移
                        xytext=(0,10),         # 每个标签上移 10 个单位，避免被点覆盖
                        ha='center')           # 水平居中对齐

    front = np.array([ind.fitness.values for ind in hof2])
    plt.scatter(front[:,0], front[:,1], c="g")

    if print_word:
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

def draw_point3(hof1, hof2, hof3, print_word=False):

    front = np.array([ind.fitness.values for ind in hof1])
    plt.scatter(front[:,0], front[:,1], c="r")
    if print_word:
        for x, y in front:
            plt.annotate(f'({x:.4f}, {y:.4f})', # 这里使用了格式化字符串保留四位小数
                        (x, y),                 # 这是标注文本的坐标位置
                        textcoords="offset points", # 使用偏移
                        xytext=(0,10),         # 每个标签上移 10 个单位，避免被点覆盖
                        ha='center')           # 水平居中对齐

    front = np.array([ind.fitness.values for ind in hof2])
    plt.scatter(front[:,0], front[:,1], c="g")

    if print_word:
        for x, y in front:
            plt.annotate(f'({x:.4f}, {y:.4f})', # 这里使用了格式化字符串保留四位小数
                        (x, y),                 # 这是标注文本的坐标位置
                        textcoords="offset points", # 使用偏移
                        xytext=(0,10),         # 每个标签上移 10 个单位，避免被点覆盖
                        ha='center')           # 水平居中对齐

    front = np.array([ind.fitness.values for ind in hof3])
    plt.scatter(front[:,0], front[:,1], c="y")

    if print_word:
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
