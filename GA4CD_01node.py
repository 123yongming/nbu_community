'''
author: pym
date: 2024.3.31
tips: 编码方式：每个节点给三位二进制编码，则整个网络至多8个社区
'''
from Evaluation import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

'''
class GA4CD:
    __init__()
    fitness()
    select()
    crossover()
    mutataion()
    show()
    
    编码方式：0-1编码，有几个节点就编码几个
'''

class GA4CD:
    def __init__(self, pop_size, crossover_rate, mutation_rate, gen, G, encode_num, tag_name):
        self.A = nx.to_numpy_array(G)
        self.pop_size = pop_size
        self.chromosome_length = len(self.A) * encode_num
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.encode_num = encode_num
        self.population = np.random.randint(2, size=(self.pop_size, self.chromosome_length))    # 记录每一代种群
        self.fitness = np.zeros(self.pop_size)
        self.gen = gen
        self.tag_name = tag_name
        self.G = G      # 整体网络结构
        self.num_edge = sum(sum(self.A))
        self.Q = []     # 每一代最优的模块度
        self.community = []     # 每一代最优的社区结构

    def cal_fitness(self, chromosome):
        index = 0
        dict = {}
        real_chromosome = []
        for i in range(0, self.chromosome_length, self.encode_num):
            real_num = 0
            for j in range(0, self.encode_num):
                if chromosome[i + j] == 0:
                    real_num += math.pow(2, 0)
                else:
                    real_num += math.pow(2, j)
            real_chromosome.append(real_num)
        for node in G.nodes():
            dict[node] = real_chromosome[index]
            index += 1

        return modularity(self.G, dict)
        # return NMI(real_chromosome, self.G, self.tag_name)

    def get_fitness(self):
        '''
        计算种群中每个个体的适应度函数
        :return:
        '''
        for i in range(self.pop_size):
            self.fitness[i] = self.cal_fitness(self.population[i])



    def select(self):
        '''
        计算种群适应度，为每个个体设置概率，进行选择，更新种群
        :return:
        '''
        self.get_fitness()
        fitness_sum = np.sum(self.fitness)
        pop_pro = [fit / fitness_sum for fit in self.fitness]
        # 轮盘赌选择概率高的个体，replace代表可以替换，返回生成index标签
        # index = np.random.choice(np.arange(self.pop_size), size=self.pop_size, p=pop_pro, replace=True)
        # self.population = self.population[index, :]
        min_index = np.argmin(self.fitness)
        max_index = np.argmax(self.fitness)
        self.fitness[min_index] = self.fitness[max_index]
        self.population[min_index] = self.population[max_index]


    def crossover(self):
        '''
        交叉操作
        :return:
        '''
        for i in range(self.pop_size, 2):
            if np.random.uniform(0, 1) < self.crossover_rate:
                index = np.random.randint(0, self.chromosome_length)
                self.population[i][index], self.population[i + 1][index] = self.population[i + 1][index], self.population[i][index]

    def mutation(self):
        for i in range(self.pop_size):
            if np.random.uniform(0, 1) < self.mutation_rate:
                index = np.random.randint(self.chromosome_length)
                self.population[i][index] = 1 - self.population[i][index]

    def showCommunity(self, G, sc_com):

        G_copy = copy.deepcopy(G)  # 复制一个图来进行社区发现

        # 可视化(原图布局)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, node_size=70, width=0.5, node_color=sc_com)
        plt.show()

        # 获取每个社区所包含的节点
        V = [node for node in G.nodes()]
        com_dict = {node: com for node, com in zip(V, sc_com)}
        k = max(com_dict.values()) + 1
        com = [[V[i] for i in range(G.number_of_nodes()) if sc_com[i] == j] for j in range(k)]

        # 构造可视化所需要的图
        G_graph = nx.Graph()
        for each in com:
            G_graph.update(nx.subgraph(G, each))
        color = [com_dict[node] for node in G_graph.nodes()]

        # 可视化(社区布局)
        pos = nx.spring_layout(G_graph, seed=4, k=0.33)
        nx.draw(G, pos, with_labels=False, node_size=1, width=0.1, alpha=0.2)
        nx.draw(G_graph, pos, with_labels=True, node_color=color, node_size=70, width=0.5, font_size=5,
                font_color='#000000')
        plt.show()

    def run(self):
        for i in range(self.gen):
            self.select()
            self.crossover()
            self.mutation()
            index = np.argmax(self.fitness)
            index = int(index)
            self.Q.append(self.fitness[index])
            self.community.append(self.population[index])

            print(f"第{i + 1} 轮迭代开始：")
            print("社区模块度：" + str(self.Q[i]))
            print("社区结构：" + str(self.community[i]))
            print(self.fitness)
        index = np.argmax(self.Q)
        real_chromosome = []
        for i in range(0, self.chromosome_length, self.encode_num):
            real_num = 0
            for j in range(0, self.encode_num):
                if self.community[index][i + j] == 0:
                    real_num += math.pow(2, 0)
                else:
                    real_num += math.pow(2, j)
            real_chromosome.append(int(real_num))
        self.showCommunity(self.G, real_chromosome)


        # print(real_chromosome)




if __name__ == "__main__":

    pop_size = 60
    crossover_rate = 0.
    mutation_rate = 0.2
    gen = 1000
    encode_num = 4
    # filepath = r'./data/karate_club.gml'
    # tag_name = 'club'
    filepath = r'./data/football.gml'
    tag_name = 'value'
    G = nx.read_gml(filepath)
    ga = GA4CD(pop_size, crossover_rate, mutation_rate, gen, G, encode_num, tag_name)
    ga.run()


    plt.plot(ga.Q)
    print(ga.Q)
    plt.xlabel("gens")
    plt.ylabel("modularity")
    plt.title("GA4CD")
    plt.show()












