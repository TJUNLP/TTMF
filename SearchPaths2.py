# -*- coding: utf-8 -*-

from pygraph.classes.digraph import digraph
import os
from numpy import *
import numpy as np
from search import ReadAllTriples
from PrecessData import load_vec_txt,get_index

def Rank(Paths, Ent2V, Rel2V, h, t, r):
    plist =[]

    for path in Paths:
        SD_r = 0.0
        SD_h = 0.0
        SD_t = 0.0
        for triple in path:
            # print(triple)
            cosV_h = dot(Ent2V[int(h)], Ent2V[int(triple[1])]) / (linalg.norm(Ent2V[int(h)]) * linalg.norm(Ent2V[int(triple[1])]))
            SD_h +=cosV_h
            cosV_t = dot(Ent2V[int(t)], Ent2V[int(triple[0])]) / (linalg.norm(Ent2V[int(t)]) * linalg.norm(Ent2V[int(triple[0])]))
            SD_t +=cosV_t

            cosV_r = dot(Rel2V[int(r)], Rel2V[int(triple[2])]) / (linalg.norm(Rel2V[int(r)]) * linalg.norm(Rel2V[int(triple[2])]))
            SD_r +=cosV_r
        SD = (SD_r + SD_h + SD_t) / (3 * len(path))
        plist.append((SD, path))

    plist = sorted(plist, key=lambda sp: sp[0], reverse=True)


    return plist


def searchpath(core, startnode, dict, taillist, Paths, pathlist, depth=5):
    depth -= 1

    if depth <= 0:
        return Paths

    if startnode not in dict.keys():
        return Paths

    sequence = dict[startnode]
    count = 0
    for key in sequence.keys():

        if key in taillist:
            continue

        for val in sequence.get(key):
            pathlist.append((startnode, key, val))
            taillist.append(key)
            # print('***', pathlist)
            s = tuple(pathlist)
            if (core + '_' + key) not in Paths.keys():
                Paths[core + '_' + key] = [s]
            else:
                Paths[core + '_' + key].append(s)
            # print(Paths)
            pathlist.remove((startnode, key, val))
            taillist.remove(key)



        # array[int(node)][int(key)] = len(sequence[key])
        for val in sequence.get(key):
            taillist.append(key)
            pathlist.append((startnode, key, val))
            Paths = searchpath(core, key, dict, taillist, Paths, pathlist, depth)
            taillist.remove(key)
            pathlist.remove((startnode, key, val))

    return Paths



if __name__ == '__main__':
    file_data = "/Users/shengbinjia/Documents/GitHub/TCdata"
    file_entity = file_data + "/FB15K/entity2id.txt"

    file_train = file_data + "/FB15K/golddataset/train2id.txt"
    file_test = file_data + "/FB15K/golddataset/test2id.txt"
    file_valid = file_data + "/FB15K/golddataset/valid2id.txt"
    file_relation = file_data + "/FB15K/relation2id.txt"
    file_ent2vec = file_data + "/FB15K_PTransE_Entity2Vec_100.txt"
    file_rel2vec = file_data + "/FB15K_PTransE_Relation2Vec_100.txt"

    file_train2_neg = file_data + "/KBE/datasets/FB15k/train2id_neg.txt"
    file_train2_pos = file_data + "/KBE/datasets/FB15k/train2id_pos.txt"
    file_test2 = file_data + "/KBE/datasets/FB15k/test2id.txt"
    file_valid2 = file_data + "/KBE/datasets/FB15k/valid2id.txt"
    file_path = file_data + "/Path_42/"

    file_temptest = file_data + "/tmptest.txt"
    # dict = ReadAllTriples([file_temptest])

    dict = ReadAllTriples([file_train, file_test, file_valid])
    print("dict size--", dict.__len__())
    print("ReadAllTriples is done!")
    rel_vocab, rel_idex_word = get_index(file_relation)
    relvec_k, Rel2V = load_vec_txt(file_rel2vec, rel_vocab, k=100)
    ent_vocab, ent_idex_word = get_index(file_entity)
    entvec_k, Ent2V = load_vec_txt(file_ent2vec, ent_vocab, k=100)

    line_dict = {}
    headlist = []
    # for filep in [file_train2_pos, file_test2, file_valid2]:!!!!!!!!!!!!!!!
    ff = '/Users/shengbinjia/Documents/GitHub/TCdata/FB15K/KBCdataset/100/'
    for filep in [ff +'/conf_train2id.txt', ff +'/conf_test2id.txt']:
        file = open(filep, "r")
        for linet in file:
            list = linet.rstrip('\n').split('\t')

            if list[0]+'_'+list[1] in line_dict.keys():
                if (list[0],list[1],list[2]) not in line_dict[list[0]+'_'+list[1]]:
                    line_dict[list[0] + '_' + list[1]].append((list[0],list[1],list[2]))
            else:
                line_dict[list[0] + '_' + list[1]] = [(list[0],list[1],list[2])]
            if int(list[0]) not in headlist:
                headlist.append(int(list[0]))

        file.close()

    for i in range(2000, 2500):#561, 2500  2750, 5000
    # for i in ['A','B','C','D','E','F','G','H','I','J','K']:
        if i in headlist:
            startnode = str(i)
            Paths = {}
            pathlist = []
            taillist = [startnode]
            Paths = searchpath(startnode, startnode, dict, taillist, Paths, pathlist, 4)

            for head in Paths.keys():
                if head in line_dict.keys():
                    for tri in line_dict[head]:
                        print('------------------'+str(i)+'--------------', str(tri))

                        if os.path.exists(file_path + tri[0] + '_' + tri[1] + '_' + tri[2] + '.txt') is True:
                            continue

                        # print(Paths[head])
                        # for i, p in enumerate(Paths[head]):
                        #     print(i, p)

                        Pranklist = Rank(Paths[head], Ent2V, Rel2V, tri[0], tri[1], tri[2])

                        fin = open(file_path + tri[0] + '_' + tri[1] + '_' + tri[2] +'.txt','w')
                        for num, ps in enumerate(Pranklist):
                            if num > 50:
                                break
                            if ps[1] == ((tri[0], tri[1], tri[2]),):
                                continue
                            for tri in ps[1]:
                                fin.write('('+tri[0]+', '+tri[1]+', '+tri[2]+')'+'\t')
                            fin.write(str(ps[0]) + '\n')
                        fin.close()

    for li in line_dict.keys():
        for tri in line_dict[li]:
            if os.path.exists(file_path + tri[0] + '_' + tri[1] + '_' + tri[2] + '.txt') is False:
                if tri[0] == tri[1]:
                    fin = open(file_path + tri[0] + '_' + tri[1] + '_' + tri[2] + '.txt', 'w')
                    fin.close()
                else:
                    print('!!!!!!!!!!!!!!!!!')



    # for list in line_tri:
    #     print('--------------------------------'+str(list))
    #     Paths = {}
    #     startnode = list[0]
    #     endnode = list[1]
    #
    #
    #     if os.path.exists(file_path + startnode+'_'+endnode+'_'+list[2]+'.txt') is True:
    #         continue
    #
    #     # startnode!=endnode
    #     if startnode == endnode:
    #         Paths.append(((startnode, endnode, list[2]),))
    #         # print([startnode, endnode])
    #     else:
    #         pathlist = []
    #         taillist = [startnode]
    #         Paths = searchpath(startnode, dict, taillist, Paths, pathlist, 4)
        # print(Paths)
        # for i, p in enumerate(Paths):
        #     print(i, p)

        # for list2 in line_dict[startnode+'_'+endnode]:
        #     print(list2)
        #     Pranklist = Rank(Paths, Rel2V, list2)
        #
        #     fin = open(file_path + startnode+'_'+endnode+'_'+list2+'.txt','w')
        #     for ps in Pranklist:
        #         if ps[1] == ((startnode, endnode, list2),):
        #             continue
        #         for tri in ps[1]:
        #             fin.write('('+tri[0]+', '+tri[1]+', '+tri[2]+')'+'\t')
        #         fin.write(str(ps[0]) + '\n')
        #     fin.close()


