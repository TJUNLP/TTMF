# -*- coding: utf-8 -*-

from pygraph.classes.digraph import digraph
import os

class PRIterator:
    __doc__ = '''计算一张图中的PR值'''

    def __init__(self, dg, core_node):
        self.damping_factor = 0.85  # 阻尼系数,即α
        self.max_iterations = 500  # 最大迭代次数
        self.min_delta = 0.00001  # 确定迭代是否结束的参数,即ϵ
        self.core_node = core_node
        self.graph = dg

    def page_rank(self):
        print('******')
        cout =0
         # 先将图中没有出链的节点改为对所有节点都有出链
        for node in self.graph.nodes():
            if len(self.graph.neighbors(node)) == 0:
                cout +=1
                # print(cout)
                digraph.add_edge(self.graph, (node, node), wt=0.5)
                digraph.add_edge(self.graph, (node, core_node), wt=0.5)
                # for node2 in self.graph.nodes():
                #     # print('$$$$$$$')
                #     if node !=node2:
                #         digraph.add_edge(self.graph, (node, node2),wt=float(1/len(self.graph.nodes())))

        print(cout)

        nodes = self.graph.nodes()
        graph_size = len(nodes)

        if graph_size == 0:
            return {}

        # page_rank = dict.fromkeys(nodes, 1.0 / graph_size)  # 给每个节点赋予初始的PR值
        page_rank = dict.fromkeys(nodes, 0.0)  # 给每个节点赋予初始的PR值
        page_rank[core_node] = 1.0
        # print(page_rank)
        damping_value = (1.0 - self.damping_factor) / graph_size  # 公式中的(1−α)/N部分
        print('start iterating...')
        flag = False
        for i in range(self.max_iterations):
            change = 0
            for node in nodes:
                rank = 0
                for incident_page in self.graph.incidents(node):  # 遍历所有“入射”的页面
                    # count = 0
                    # for neighboredge in self.graph.neighbors(incident_page):
                    #     count += self.graph.edge_weight((incident_page,neighboredge))
                    # rank += self.damping_factor * (page_rank[incident_page] / count * self.graph.edge_weight((incident_page,node)))
                    # rank += self.damping_factor * (page_rank[incident_page] / len(self.graph.neighbors(incident_page)))

                    rank += self.damping_factor * page_rank[incident_page] * float(self.graph.edge_weight((incident_page,node)))
                rank += damping_value
                change += abs(page_rank[node] - rank)  # 绝对值
                page_rank[node] = rank

            # print("This is NO.%s iteration" % (i + 1))
            # print(page_rank)

            if change < self.min_delta:
                flag = True
                # print("\n\nfinished in %s iterations!" % i)
                break
        if flag == False:
            print("finished out of %s iterations!" % self.max_iterations)
        return page_rank


if __name__ == '__main__':
    # dg = digraph()
    #
    # dg.add_nodes(["A", "B", "C", "D", "E"])
    #
    # dg.add_edge(("A", "B"), wt=199)
    # dg.add_edge(("A", "C"), wt=14)
    # dg.add_edge(("A", "D"), wt=100)
    # dg.add_edge(("B", "D"), wt=124)
    # dg.add_edge(("C", "E"), wt=1000)
    # dg.add_edge(("D", "E"), wt=13)
    # dg.add_edge(("B", "E"), wt=16)
    # dg.add_edge(("E", "A"), wt=18)
    # core_node = 'A'

    file_data = "/Users/shengbinjia/Documents/GitHub/TCdata"
    file_entityRank = file_data + "/entityRank_4/"
    file_subGraphs = file_data + "/subGraphs_4/"

    for files in os.listdir(file_subGraphs):
        file = open(file_subGraphs+files, "r")
        dg = digraph()
        core_node = os.path.splitext(files)[0]
        print('corenode----',core_node)
        for i, line in enumerate(file):
            if i == 0:
                list = line.rstrip('\t\n').rstrip('\t').rstrip('\n').split("\t")
                for n in list:
                    dg.add_node(n.strip("\t").strip("\n"))
            else:

                list = line.split("\t")
                # dg.add_edge((list[0], list[1]), wt=int(list[2].strip("\n")))
                dg.add_edge((list[0], list[1]), wt=list[2].strip("\n"))
        print('dg size...', dg.nodes().__len__())

        pr = PRIterator(dg, core_node)
        page_ranks = pr.page_rank()

        # print("The final page rank is\n", page_ranks)
        fo = open(file_entityRank + core_node + ".txt", "w")
        for key in page_ranks.keys():
            fo.write(key +"\t" + str(page_ranks.get(key)) + "\n")
        fo.close()
        file.close()

    # {'A': 0.29633523435739856, 'B': 0.11396164973459627, 'C': 0.11396164973459627, 'D': 0.16239535087179965,
    #  'E': 0.31333715165264}

    # files = os.listdir(file_entityRank)
    # for f in files:
    #     print(f)
    #     if f == '.DS_Store':
    #         continue
    #     fo = open(os.path.join(file_entityRank, f),'r')
    #     fin = open(os.path.join(file_data + "/entityRank_444/", f),'w')
    #     lines = fo.readlines()
    #     for i, line in enumerate(lines):
    #         nodes = line.rstrip('\n').split('\t')
    #
    #         if nodes[0] == '0':
    #             nodes[0] = '14951'
    #         fin.write(nodes[0] + '\t' + nodes[1] + '\n')
    #
    #     fin.close()
    #     fo.close()

