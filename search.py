#coding = utf-8

import time
from pygraph.classes.digraph import digraph
import os


def ReadAllTriples(files):
    dict = {}

    for f in files:
        file = open(f, "r")
        for line in file:
            list = line.split(" ")

            if list[0] in dict.keys():
                if list[1] in dict.get(list[0]).keys():
                    dict.get(list[0]).get(list[1]).append(list[2].strip('\n'))
                else:
                    dict.get(list[0])[list[1]] = [list[2].strip('\n')]
            else:
                dict[list[0]] = {list[1]:[list[2].strip('\n')]}

        # for key in dict.keys():
        #     print(key+' : ',dict[k])
        file.close()

    return dict


def DFS(dict, dg, node, depth=3):
    depth -= 1

    if depth < 0:
        return dg
    if node not in dict.keys():
        return dg
    sequence = dict[node]
    count = 0
    for key in sequence.keys():
        if not dg.has_node(key):
            dg.add_node(key)
        if not dg.has_edge((node, key)):
            dg.add_edge((node, key), wt=len(sequence[key]))
            count += len(sequence[key])
        else:
            continue
            # print(node, key, dg.edge_weight((node, key)), len(sequence[key]))

        # array[int(node)][int(key)] = len(sequence[key])
        dg = DFS(dict, dg, key, depth)

    for n in dg.neighbors(node):
        dg.set_edge_weight((node, n),wt= float(dg.edge_weight((node, n))/max(count,1)))

    return dg




if __name__ == '__main__':

    file_data = "/Users/shengbinjia/Documents/GitHub/TCdata"
    file_entity = "./data/FB15K/entity2id.txt"
    file_train = "./data/FB15K/train2id.txt"
    file_test = "./data/FB15K/test2id.txt"
    file_valid = "./data/FB15K/valid2id.txt"
    file_subGraphs = file_data + "/subGraphs_4/"

    dict = ReadAllTriples([file_train, file_test, file_valid])
    print("dict size--", dict.__len__())
    print("ReadAllTriples is done!")

    file = open(file_entity, "r")

    for line in file:
        list = line.split("	")
        node0 = list[1].strip('\n')
        print("node0-----", node0)

        dg = digraph()
        dg.add_node(node0)
        t1 = time.clock()
        dg = DFS(dict, dg, node0, depth=4)

        fo = open(file_subGraphs + node0 + ".txt", "w")
        NODE = ""
        for nodei in dg.nodes():
            NODE = NODE +nodei+ "\t"
        fo.write(NODE+'\n')

        for e in dg.edges():
            fo.write(e[0] + "\t" + e[1] + "\t" + str(dg.edge_weight(e))+'\n')
        fo.close()



        t2=time.clock()
        # time.sleep(1)
        print(t2-t1)
        # print(dg.nodes().__len__())
        # for edge in dg.edges():
        #     print('edge----',edge)
    file.close()

    # files = os.listdir(file_subGraphs)
    # for f in files:
    #     print(f)
    #     fo = open(os.path.join(file_subGraphs, f),'r')
    #     fin = open(os.path.join(file_data + "/subGraphs_444/", f),'w')
    #     lines = fo.readlines()
    #     for i, line in enumerate(lines):
    #         nodes = line.rstrip('\t\n').rstrip('\t').rstrip('\n').split('\t')
    #
    #         if i == 0:
    #             for node in nodes:
    #                 if node == '0':
    #                     node = '14951'
    #                 fin.write(node + '\t')
    #             fin.write('\n')
    #         else:
    #             if nodes[0] == '0':
    #                 nodes[0] = '14951'
    #             if nodes[1] == '0':
    #                 nodes[1] = '14951'
    #             fin.write(nodes[0] + '\t' + nodes[1] + '\t' + nodes[2] + '\n')
    #
    #     fin.close()
    #     fo.close()






