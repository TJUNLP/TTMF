# -*- coding: utf-8 -*-
import numpy as np
from pygraph.classes.digraph import digraph
import os
from search import ReadAllTriples
from ResourceRankConfidence import get_RRankConfidence, rrcThreshold
from TransConfidence import get_TransConfidence, tcThreshold
import pickle
import json
import re
import math
import random
def load_vec_txt(fname, vocab, k=300):
    f = open(fname)
    w2v={}
    W = np.zeros(shape=(vocab.__len__() + 2, k))
    unknowtoken = 0
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        w2v[word] = coefs
    f.close()
    w2v["**UNK**"] = np.random.uniform(-0.25, 0.25, k)
    for word in vocab:
        # print(word)

        if not w2v.__contains__(word):
            w2v[word] = w2v["**UNK**"]
            unknowtoken +=1
            W[vocab[word]] = w2v[word]
        else:
            W[vocab[word]] = w2v[word]

    print('!!!!!! UnKnown tokens in w2v', unknowtoken)
    # for sss in W:
    #     print(sss)
    return k, W


def get_index(file):

    source_vob = {}
    sourc_idex_word = {}

    f = open(file,'r')
    fr = f.readlines()
    for line in fr:
        sourc = line.strip('\r\n').rstrip('\n').split('\t')
        if not source_vob.__contains__(sourc[0]):
            source_vob[sourc[0]] = int(sourc[1])
            sourc_idex_word[int(sourc[1])] = sourc[0]
    f.close()

    return source_vob, sourc_idex_word


def make_idx_data_index(file, max_s, source_vob, target_vob):

    data_s_all=[]
    data_t_all=[]
    f = open(file,'r')
    fr = f.readlines()

    count = 0
    data_t = []
    data_s = []
    for line in fr:

        if line.__len__() <= 1:
            num = max_s - count
            # print('num ', num, 'max_s', max_s, 'count', count)
            for inum in range(0, num):
                data_s.append(0)
                data_t.append(0)
            # print(data_s)
            # print(data_t)
            data_s_all.append(data_s)
            data_t_all.append(data_t)
            data_t = []
            data_s = []
            count =0
            continue

        sent = line.strip('\r\n').rstrip('\n').split(' ')
        if not source_vob.__contains__(sent[0]):
            data_s.append(source_vob["**UNK**"])
        else:
            data_s.append(source_vob[sent[0]])

        data_t.append(target_vob[sent[4]])
        count += 1


    f.close()
    return [data_s_all,data_t_all]


# def get_values_transE(train_triple, entity2vec, relation2vec):
#     train_transE = []
#     for tri in train_triple:
#         h = entity2vec[tri[0]]
#         t = entity2vec[tri[1]]
#         r = relation2vec[tri[2]]
#         s = h + r - t
#         sum = 0.0
#         for i in s:
#             sum +=i * i
#         train_transE.append(np.linalg.norm(s, ord=2))
#         # train_transE.append(math.sqrt(sum))
#     return train_transE

def get_values_transE(train_triple, train_transE_file, ent_vocab, rel_vocab):
    trans_dict = {}
    f = open(train_transE_file, 'r')
    lines = f.readlines()

    for line in lines:
        sp = line.rstrip('\n').split('\t')
        key = str(ent_vocab[sp[0]]) + ' ' + str(ent_vocab[sp[2]]) + ' ' + str(rel_vocab[sp[1]])
        # print(key)
        trans_dict[key] = float(sp[4])
    num = 0
    train_transE = []
    for tri in train_triple:
        h = tri[0]
        t = tri[1]
        r = tri[2]
        key = str(h)+ ' ' + str(t) + ' ' + str(r)

        if key in trans_dict.keys():
            train_transE.append(trans_dict[key])
        else:
            train_transE.append(random.random())
            num += 1
            print(num, 'get_values_transE error !!!!', key)
    f.close()
    return train_transE


def get_data_txt(trainfile):
    train_triple = []
    train_confidence = []

    f = open(trainfile, "r")
    lines = f.readlines()
    for line in lines:
        tri = line.rstrip('\r\n').rstrip('\n').rstrip('\r').split('\t')
        train_triple.append((int(tri[0]), int(tri[1]), int(tri[2]), int(tri[3])))
        if tri[3] == '1':
            train_confidence.append([0, 1])
        else:
            train_confidence.append([1, 0])
    f.close()

    return train_triple, train_confidence


def get_path_index(trainfile_path, max_p, train_triple, topk):
    train_path_h = []
    train_path_t = []
    train_path_r = []

    for baset in train_triple:
        ph = []
        pt = []
        pr = []
        length = 0
        fstr = trainfile_path + str(baset[0])+'_'+ str(baset[1])+'_'+ str(baset[2])+'.txt'
        if os.path.exists(fstr) is True:
            # print("$$$$$$$$$$$$")
            f = open(fstr, "r")
            lines = f.readlines()

            if lines.__len__() >= (topk+1):
                tri = lines[topk].rstrip('\t\n').rstrip('\n').split('\t')
                for t in range(0, len(tri)-1):
                    id = tri[t].strip('(').strip(')').split(', ')
                    # print('***', id[0], id[1], id[2])
                    ph.append(int(id[0]))
                    pt.append(int(id[1]))
                    pr.append(int(id[2]))
                length = len(tri)-1
            f.close()
        else:
            print(fstr, 'Not find the path file!!!!!')
        for i in range(0, max_p - length):
            ph.append(0)
            pt.append(0)
            pr.append(0)
        train_path_h.append(ph)
        train_path_t.append(pt)
        train_path_r.append(pr)

    return train_path_h, train_path_t, train_path_r

def get_dict_entityRank(entityRank):
    dict = {}

    files = os.listdir(entityRank)
    for file in files:
        # print(file)
        fo = open(entityRank + file, 'r')
        lines = fo.readlines()
        dict_l = {}
        for line in lines:

            nodes = line.rstrip('\n').split('\t')
            if nodes[0] == '':
                continue
            dict_l[int(nodes[0])] = float(nodes[1])
        dict[int(os.path.splitext(file)[0])] = dict_l
        fo.close()
    return dict

def get_values_rrank(dict, train_triple):

    train_rrank = []
    for triple in train_triple:

        if triple[1] in dict[triple[0]].keys():
            # print(dict[triple[0]][triple[1]])
            train_rrank.append(dict[triple[0]][triple[1]])
        else:
            train_rrank.append(0.0)

    return train_rrank


def get_rrank_features(dict_features, Examples):

    features = []

    for id, example in enumerate(Examples):
        # print(id)
        if example[1] in dict_features[example[0]].keys():
            features.append(dict_features[example[0]][example[1]])
        else:
            features.append([0.0, 0.0, 0.0, 0.0, 0.0, 10000.0])


    return features

def get_dict_features(file_RRfeatures):
    dict = {}

    files = os.listdir(file_RRfeatures)
    id = 1
    for file in files:
        # print(id)
        id += 1
        fo = open(file_RRfeatures + file, 'r')
        lines = fo.readlines()
        dict_l = {}
        for line in lines:

            nodes = line.rstrip('\n').split('\t')

            dict_l[int(nodes[0])] = [float(nodes[1]), float(nodes[2]), float(nodes[3]), float(nodes[4]), float(nodes[5]), float(nodes[6])]
        dict[int(os.path.splitext(file)[0])] = dict_l
        fo.close()
    return dict

def get_data(entity2idfile, relation2idfile,
             entity2vecfile, relation2vecfile, w2v_k,
             trainfile, testfile,
             testfile_KGC_h_t,
             testfile_KGC_hr_,
             testfile_KGC__rt,
             path_file, max_p,
             entityRank,
             datafile):

    ent_vocab, ent_idex_word = get_index(entity2idfile)
    rel_vocab, rel_idex_word = get_index(relation2idfile)
    print("entity vocab size: ", str(len(ent_vocab)), str(len(ent_idex_word)))
    print("relation vocab size: ", str(len(rel_vocab)), str(len(rel_idex_word)))


    entvec_k, entity2vec = load_vec_txt(entity2vecfile, ent_vocab, k=w2v_k)
    print("word2vec loaded!")
    print("entity2vec  size: " + str(len(entity2vec)))

    relvec_k, relation2vec = load_vec_txt(relation2vecfile, rel_vocab, k=w2v_k)
    print("word2vec loaded!")
    print("relation2vec  size: " + str(len(relation2vec)))

    train_triple, train_confidence = get_data_txt(trainfile)
    # dev_triple, dev_confidence = get_data_txt(devfile)
    test_triple, test_confidence = get_data_txt(testfile)
    print('train_triple size: ', len(train_triple), 'train_confidence size: ', len(train_confidence))
    # print('dev_triple size: ', len(dev_triple), 'dev_confidence size: ', len(dev_confidence))
    print('test_triple size: ', len(test_triple), 'test_confidence size: ', len(test_confidence))

    test_triple_KGC_h_t, test_confidence_KGC_h_t = get_data_txt(testfile_KGC_h_t)
    test_triple_KGC_hr_, test_confidence_KGC_hr_ = get_data_txt(testfile_KGC_hr_)
    test_triple_KGC__rt, test_confidence_KGC__rt = get_data_txt(testfile_KGC__rt)
    print('test_triple_KGC_h_t size: ', len(test_triple_KGC_h_t), 'test_confidence_KGC_hr_ size: ', len(test_confidence_KGC_hr_))

    # train_transE = get_values_transE(train_triple, train_transE_file, ent_vocab, rel_vocab)
    # dev_transE = get_values_transE(dev_triple, dev_transE_file, ent_vocab, rel_vocab)
    # test_transE = get_values_transE(test_triple, test_transE_file, ent_vocab, rel_vocab)

    tcthreshold_dict = tcThreshold(train_triple, entity2vec, relation2vec)
    train_transE = get_TransConfidence(tcthreshold_dict, train_triple, entity2vec, relation2vec)
    test_transE = get_TransConfidence(tcthreshold_dict, test_triple, entity2vec, relation2vec)
    print('train_transE size: ',train_transE.__len__())
    # print('dev_transE size: ', dev_transE.__len__())
    print('test_transE size: ', test_transE.__len__())

    test_transE_hr_ = get_TransConfidence(tcthreshold_dict, test_triple_KGC_hr_, entity2vec, relation2vec)
    test_transE_h_t = get_TransConfidence(tcthreshold_dict, test_triple_KGC_h_t, entity2vec, relation2vec)
    test_transE__rt = get_TransConfidence(tcthreshold_dict, test_triple_KGC__rt, entity2vec, relation2vec)
    print('test_transE__rt size: ', test_transE__rt.__len__())

    # dict_entityRank = get_dict_entityRank(entityRank)
    # train_rrank = get_values_rrank(dict_entityRank, train_triple)
    # dev_rrank = get_values_rrank(dict_entityRank, dev_triple)
    # test_rrank = get_values_rrank(dict_entityRank, test_triple)

    # rrkthreshold_dict = rrcThreshold(train_triple, dict_entityRank)
    # train_rrank = get_RRankConfidence(rrkthreshold_dict, train_triple, dict_entityRank)
    # test_rrank = get_RRankConfidence(rrkthreshold_dict, test_triple, dict_entityRank)

    dict_features = get_dict_features(entityRank)
    rrkthreshold_dict = {}
    train_rrank = get_rrank_features(dict_features, train_triple)
    test_rrank = get_rrank_features(dict_features, test_triple)
    print('train_rrank size : ', len(train_rrank))
    # print('dev_rrank size : ', len(dev_rrank))
    print('test_rrank size : ', len(test_rrank))

    test_rrank_KGC_h_t = get_rrank_features(dict_features, test_triple_KGC_h_t)
    test_rrank_KGC_hr_ = get_rrank_features(dict_features, test_triple_KGC_hr_)
    test_rrank_KGC__rt = get_rrank_features(dict_features, test_triple_KGC__rt)
    print('test_rrank_KGC__rt size : ', len(test_rrank_KGC__rt))

    train_path_h, train_path_t, train_path_r = get_path_index(path_file, max_p, train_triple, 0)
    test_path_h, test_path_t, test_path_r = get_path_index(path_file, max_p, test_triple, 0)
    train_path2_h, train_path2_t, train_path2_r = get_path_index(path_file, max_p, train_triple, 1)
    test_path2_h, test_path2_t, test_path2_r = get_path_index(path_file, max_p, test_triple, 1)
    train_path3_h, train_path3_t, train_path3_r = get_path_index(path_file, max_p, train_triple, 2)
    test_path3_h, test_path3_t, test_path3_r = get_path_index(path_file, max_p, test_triple, 2)
    print('train_path size: ', len(train_path_h))
    # print('dev_path_h size: ', len(dev_path_r))
    print('test_path_t size: ', len(test_path_t))

    test_path_h_KGC_h_t, test_path_t_KGC_h_t, test_path_r_KGC_h_t = get_path_index(path_file, max_p, test_triple_KGC_h_t, 0)
    test_path_h_hr_, test_path_t_hr_, test_path_r_hr_ = get_path_index(path_file, max_p, test_triple_KGC_hr_, 0)
    test_path_h__rt, test_path_t__rt, test_path_r__rt = get_path_index(path_file, max_p, test_triple_KGC__rt, 0)

    print ("dataset created!")
    out = open(datafile,'wb')
    pickle.dump([ent_vocab, ent_idex_word, rel_vocab, rel_idex_word,
                 entity2vec, entvec_k,
                 relation2vec, relvec_k,
                 train_triple, train_confidence,
                 test_triple, test_confidence,
                    test_triple_KGC_h_t, test_confidence_KGC_h_t,
                    test_triple_KGC_hr_, test_confidence_KGC_hr_,
                    test_triple_KGC__rt, test_confidence_KGC__rt,
                 tcthreshold_dict, train_transE, test_transE,
                    test_transE_h_t,
                    test_transE_hr_,
                    test_transE__rt,
                 rrkthreshold_dict, train_rrank, test_rrank,
                    test_rrank_KGC_h_t,
                    test_rrank_KGC_hr_,
                    test_rrank_KGC__rt,
                 max_p,
                 train_path_h, train_path_t, train_path_r,
                 test_path_h, test_path_t, test_path_r,
                 train_path2_h, train_path2_t, train_path2_r,
                 test_path2_h, test_path2_t, test_path2_r,
                 train_path3_h, train_path3_t, train_path3_r,
                 test_path3_h, test_path3_t, test_path3_r,
                    test_path_h_KGC_h_t, test_path_t_KGC_h_t, test_path_r_KGC_h_t,
                    test_path_h_hr_, test_path_t_hr_, test_path_r_hr_,
                    test_path_h__rt, test_path_t__rt, test_path_r__rt], out)
    out.close()

    '''
    print ("dataset created!")
    out = open(datafile,'wb')
    pickle.dump([ent_vocab, ent_idex_word, rel_vocab, rel_idex_word,
                 entity2vec, entvec_k,
                 relation2vec, relvec_k,
                 train_triple, train_confidence,
                 test_triple, test_confidence,
                    test_triple_KGC_h_t, test_confidence_KGC_h_t,
                    test_triple_KGC_hr_, test_confidence_KGC_hr_,
                    test_triple_KGC__rt, test_confidence_KGC__rt,
                 tcthreshold_dict, train_transE, test_transE,
                    test_transE_h_t,
                    test_transE_hr_,
                    test_transE__rt,
                 max_p], out)

    out.close()
    '''


if __name__=="__main__":

    file_data = "/Users/shengbinjia/Documents/GitHub/TCdata"

    entity2idfile = file_data + "/FB15K/entity2id.txt"
    relation2idfile = file_data + "/FB15K/relation2id.txt"
    entity2vecfile =file_data + "/FB15K_TransE_Entity2Vec_100.txt"
    relation2vecfile = file_data + "/FB15K_TransE_Relation2Vec_100.txt"
    datafile = "./model/data.pkl"
    trainfile = file_data + "/KBE/datasets/FB15k/valid2id.txt"
    devfile = file_data + "/KBE/datasets/FB15k/test2id.txt"
    testfile = file_data + "/KBE/datasets/FB15k/test2id.txt"
    trainfile_path = file_data + "/Path_4/train/"
    devfile_path = file_data + "/Path_4/test/"
    testfile_path = file_data + "/Path_4/test/"
    entityRank = file_data + "/entityRank_4/"
    modelfile = "./model/model.h5"
    resultdir = "./data/result/"

    ent_vocab, ent_idex_word = get_index(entity2idfile)
    rel_vocab, rel_idex_word = get_index(relation2idfile)
    print("entity vocab size: ", str(len(ent_vocab)), str(len(ent_idex_word)))
    print("relation vocab size: ", str(len(rel_vocab)), str(len(rel_idex_word)))


    entvec_k, entity2vec = load_vec_txt(entity2vecfile, ent_vocab, k=100)
    print("word2vec loaded!")
    print("entity2vec  size: " + str(len(entity2vec)))

    relvec_k, relation2vec = load_vec_txt(relation2vecfile, rel_vocab, k=100)
    print("word2vec loaded!")
    print("relation2vec  size: " + str(len(relation2vec)))

    test_triple, test_confidence = get_data_txt(testfile)
    print('test_triple size: ', len(test_triple), 'test_confidence size: ', len(test_confidence))
    #
    # test_transE = get_values_transE(test_triple, entity2vec, relation2vec)
    # print('test_transE size: ', test_transE.__len__())

    dict_entityRank = get_dict_entityRank(entityRank)
    test_rrank = get_values_rrank(dict_entityRank, test_triple)

    print('test_rrank size : ', len(test_rrank))

    result = {}
    E_max = 0.0
    E_min = 10.0
    core = 0.0

    while core <= 0.001:
        right = 0.0
        right_neg = 0.0
        right_pos = 0.0
        neg = 0.00001
        pos = 0.00001
        for i in range(0, len(test_triple)):
            if E_max < test_rrank[i]:
                E_max = test_rrank[i]
            if E_min > test_rrank[i]:
                E_min = test_rrank[i]

            if test_rrank[i] < core:
                neg += 1.0
                if test_confidence[i] == 0.0:
                    right_neg +=1.0

            elif test_rrank[i] >= core:
                pos += 1.0
                if test_confidence[i] == 1.0:
                    right_pos += 1.0

            else:
                print(test_confidence[i], '***!!!!!!!!!!!!!!!')

        print('right_neg / neg ---------------------------------------------', right_neg / neg)
        print('right_pos / pos------------------------------------------------', right_pos / pos)
        print(core, '-----', right_pos, right_neg,  (right_pos + right_neg)/ len(test_triple))
        core +=0.00005

    print('E_max', E_max, 'E_min', E_min)



