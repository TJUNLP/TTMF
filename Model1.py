import pickle
import os.path
import numpy as np
from PrecessData import get_data
from keras.layers.core import Dropout,Flatten
from keras.layers.merge import concatenate
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, SimpleRNN, RepeatVector,add,subtract,dot
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import regularizers
# from keras.losses import my_cross_entropy_withWeight
import matplotlib.pyplot as plt
from ResourceRankConfidence import get_RRankConfidence
from TransConfidence import get_TransConfidence
from PrecessData import get_dict_entityRank

def creat_Model_BiLSTM_BP(entvocabsize, relvocabsize, ent2vec, rel2vec, input_path_lenth,
                          ent_emd_dim, rel_emd_dim):

    ent_h_input = Input(shape=(1,), dtype='int32')
    ent_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=1,
                                   mask_zero=False, trainable=False, weights=[ent2vec])(ent_h_input)

    ent_t_input = Input(shape=(1,), dtype='int32')
    ent_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=1,
                                   mask_zero=False, trainable=False, weights=[ent2vec])(ent_t_input)

    rel_r_input = Input(shape=(1,), dtype='int32')
    rel_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=1,
                                   mask_zero=False, trainable=False, weights=[rel2vec])(rel_r_input)


    path_h_input = Input(shape=(input_path_lenth,), dtype='int32')
    path_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path_h_input)

    path_t_input = Input(shape=(input_path_lenth,), dtype='int32')
    path_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path_t_input)

    path_r_input = Input(shape=(input_path_lenth,), dtype='int32')
    path_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[rel2vec])(path_r_input)

    path2_h_input = Input(shape=(input_path_lenth,), dtype='int32')
    path2_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path2_h_input)

    path2_t_input = Input(shape=(input_path_lenth,), dtype='int32')
    path2_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path2_t_input)

    path2_r_input = Input(shape=(input_path_lenth,), dtype='int32')
    path2_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[rel2vec])(path2_r_input)

    path3_h_input = Input(shape=(input_path_lenth,), dtype='int32')
    path3_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path3_h_input)

    path3_t_input = Input(shape=(input_path_lenth,), dtype='int32')
    path3_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path3_t_input)

    path3_r_input = Input(shape=(input_path_lenth,), dtype='int32')
    path3_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[rel2vec])(path3_r_input)

    ent_h_embedding = Flatten()(ent_h_embedding)
    ent_t_embedding = Flatten()(ent_t_embedding)
    rel_r_embedding = Flatten()(rel_r_embedding)
    ent_h_embedding = RepeatVector(input_path_lenth)(ent_h_embedding)
    ent_t_embedding = RepeatVector(input_path_lenth)(ent_t_embedding)
    rel_r_embedding = RepeatVector(input_path_lenth)(rel_r_embedding)

    path_embedding = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding,
                                  path_h_embedding,
                                  path_r_embedding,
                                  path_t_embedding], axis=-1)
    path_embedding = Dropout(0.5)(path_embedding)

    path_LSTM = SimpleRNN(100, return_sequences=False)(path_embedding)
    path_LSTM = BatchNormalization()(path_LSTM)
    # path_LSTM = Dropout(0.5)(path_LSTM)

    # path_LSTM = RepeatVector(1)(path_LSTM)

    # ent_h_embedding = Flatten()(ent_h_embedding)
    # ent_t_embedding = Flatten()(ent_t_embedding)
    # rel_r_embedding = Flatten()(rel_r_embedding)
    # path_LSTM = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding, path_LSTM])
    path_LSTM = Dropout(0.5)(path_LSTM)

    path_value = Dense(1, activation='sigmoid')(path_LSTM)

    # -----------------
    path2_embedding = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding,
                                  path2_h_embedding,
                                  path2_r_embedding,
                                  path2_t_embedding], axis=-1)
    path2_embedding = Dropout(0.5)(path2_embedding)

    path2_LSTM = SimpleRNN(100, return_sequences=False)(path2_embedding)
    path2_LSTM = BatchNormalization()(path2_LSTM)
    path2_LSTM = Dropout(0.5)(path2_LSTM)
    path2_value = Dense(1, activation='sigmoid')(path2_LSTM)
    # ------------------
    path3_embedding = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding,
                                  path3_h_embedding,
                                  path3_r_embedding,
                                  path3_t_embedding], axis=-1)
    path3_embedding = Dropout(0.5)(path3_embedding)

    path3_LSTM = SimpleRNN(100, return_sequences=False)(path3_embedding)
    path3_LSTM = BatchNormalization()(path3_LSTM)
    path3_LSTM = Dropout(0.5)(path3_LSTM)
    path3_value = Dense(1, activation='sigmoid')(path3_LSTM)
    # ------------------


    TransE_input = Input(shape=(1,), dtype='float32')
    RRank_input = Input(shape=(6,), dtype='float32')


    RRank_hinden = Dense(100, activation='tanh')(RRank_input)
    RRank_hinden = Dropout(0.5)(RRank_hinden)
    RRank_value = Dense(1, activation='sigmoid')(RRank_hinden)
    # TransE_hidden = Dense(50)(TransE_input)
    # TransE_hidden = Dropout(0.5)(TransE_hidden)

    BP_input = concatenate([
        path_value, path2_value, path3_value,
        TransE_input,
        RRank_value
    ], axis=-1)


    BP_hidden = Dense(50)(BP_input)
    BP_hidden = Dropout(0.5)(BP_hidden)
    model = Dense(2, activation='softmax')(BP_hidden)
    # model = Dense(1, activation='tanh')(BP_hidden)

    Models = Model([
        ent_h_input, ent_t_input, rel_r_input,
        path_h_input, path_t_input, path_r_input,
        path2_h_input, path2_t_input, path2_r_input,
        path3_h_input, path3_t_input, path3_r_input,
        TransE_input,
        RRank_input
    ], model)

    Models.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return Models


def SelectModel(modelname, entvocabsize, relvocabsize, ent2vec, rel2vec,
                input_path_lenth,
                ent_emd_dim, rel_emd_dim):
    nn_model = None
    if modelname is 'creat_Model_BiLSTM_BP':
        nn_model = creat_Model_BiLSTM_BP( entvocabsize = entvocabsize,
                                         relvocabsize = relvocabsize,
                                         ent2vec = ent2vec, rel2vec =rel2vec,
                                         input_path_lenth = input_path_lenth,
                ent_emd_dim = ent_emd_dim, rel_emd_dim =rel_emd_dim)


    return nn_model


def get_training_xy_otherset(train_triple, train_confidence, dev_triple, dev_confidence, train_transE, dev_transE,
                             train_rrank, dev_rrank, train_path_h, train_path_t, train_path_r, dev_path_h, dev_path_t,
                             dev_path_r, max_p, shuffle):

    assert len(train_triple) == len(train_confidence)
    indices = np.arange(len(train_confidence))
    if shuffle:
        np.random.shuffle(indices)

    input_train_h = np.zeros((len(train_triple),1)).astype('int32')
    input_train_t = np.zeros((len(train_triple),1)).astype('int32')
    input_train_r = np.zeros((len(train_triple),1)).astype('int32')

    input_train_path_h = np.zeros((len(train_path_h),max_p)).astype('int32')
    input_train_path_t = np.zeros((len(train_path_h),max_p)).astype('int32')
    input_train_path_r = np.zeros((len(train_path_h),max_p)).astype('int32')
    y = np.zeros((len(train_confidence), 2)).astype('float32')
    input_train_transE = np.zeros((len(train_transE), 1)).astype('float32')
    input_train_rrank = np.zeros((len(train_rrank), 1)).astype('float32')

    for idx, s in enumerate(indices):
        # print(s, len(train_path_h))
        # print(train_path_h[s])
        input_train_h[idx,] = train_triple[s][0]
        input_train_t[idx,] = train_triple[s][1]
        input_train_r[idx,] = train_triple[s][2]
        input_train_path_h[idx,] = train_path_h[s]
        input_train_path_t[idx,] = train_path_t[s]
        input_train_path_r[idx,] = train_path_r[s]
        y[idx,] = train_confidence[s]
        input_train_transE[idx,] = train_transE[s]
        input_train_rrank[idx,] = train_rrank[s]
        # for idx2, word in enumerate(train_confidence[s]):
        #     targetvec = np.zeros(2).astype('int32')
        #     targetvec[word] = 1
        #     y[idx, ] = targetvec



    input_dev_h = np.zeros((len(dev_triple),1)).astype('int32')
    input_dev_t = np.zeros((len(dev_triple),1)).astype('int32')
    input_dev_r = np.zeros((len(dev_triple),1)).astype('int32')
    for idx, tri in enumerate(dev_triple):
        input_dev_h[idx,] = tri[0]
        input_dev_t[idx,] = tri[1]
        input_dev_r[idx,] = tri[2]

    input_dev_path_h = np.array(dev_path_h[:])
    input_dev_path_t = np.array(dev_path_t[:])
    input_dev_path_r =np.array(dev_path_r[:])
    y_dev = np.array(dev_confidence[:])
    input_dev_transE = np.array(dev_transE[:])
    input_dev_rrank = np.array(dev_rrank[:])


    yield input_train_h, input_train_t, input_train_r, \
          input_dev_h, input_dev_t, input_dev_r,\
          input_train_path_h, input_train_path_t, input_train_path_r, y, \
            input_dev_path_h, input_dev_path_t, input_dev_path_r, y_dev, \
            input_train_transE, input_dev_transE,\
            input_train_rrank, input_dev_rrank


# def get_training_xy(train_triple, train_confidence, train_transE, train_rrank, train_path_h, train_path_t, train_path_r,
#                     max_p, shuffle):
#
#     assert len(train_triple) == len(train_confidence)
#     indices = np.arange(len(train_confidence))
#     if shuffle:
#         np.random.shuffle(indices)
#
#     input_train_h = np.zeros((len(train_triple),1)).astype('int32')
#     input_train_t = np.zeros((len(train_triple),1)).astype('int32')
#     input_train_r = np.zeros((len(train_triple),1)).astype('int32')
#
#     input_train_path_h = np.zeros((len(train_path_h),max_p)).astype('int32')
#     input_train_path_t = np.zeros((len(train_path_h),max_p)).astype('int32')
#     input_train_path_r = np.zeros((len(train_path_h),max_p)).astype('int32')
#     y = np.zeros((len(train_confidence), 2)).astype('float32')
#     input_train_transE = np.zeros((len(train_transE), 1)).astype('float32')
#     input_train_rrank = np.zeros((len(train_rrank), 1)).astype('float32')
#
#     for idx, s in enumerate(indices):
#         # print(s, len(train_path_h))
#         # print(train_path_h[s])
#         input_train_h[idx,] = train_triple[s][0]
#         input_train_t[idx,] = train_triple[s][1]
#         input_train_r[idx,] = train_triple[s][2]
#         input_train_path_h[idx,] = train_path_h[s]
#         input_train_path_t[idx,] = train_path_t[s]
#         input_train_path_r[idx,] = train_path_r[s]
#         y[idx,] = train_confidence[s]
#         input_train_transE[idx,] = train_transE[s]
#         input_train_rrank[idx,] = train_rrank[s]
#
#     num_validation_samples = int(0.2 * len(train_triple))
#
#     train_h = input_train_h[:-num_validation_samples]
#     train_t = input_train_t[:-num_validation_samples]
#     train_r = input_train_r[:-num_validation_samples]
#     input_dev_h = input_train_h[-num_validation_samples:]
#     input_dev_t = input_train_t[-num_validation_samples:]
#     input_dev_r = input_train_r[-num_validation_samples:]
#
#     input_train_path_h, input_train_path_t, input_train_path_r, y,
#
#     yield train_h, train_t, train_r, \
#           input_dev_h, input_dev_t, input_dev_r,\
#           input_train_path_h, input_train_path_t, input_train_path_r, y, \
#             input_dev_path_h, input_dev_path_t, input_dev_path_r, y_dev, \
#             input_train_transE, input_dev_transE,\
#             input_train_rrank, input_dev_rrank

def train_model(modelname, datafile, modelfile, resultdir, npochos=100, batch_size=50, retrain=False):
    # load training data and test data

    ent_vocab, ent_idex_word, rel_vocab, rel_idex_word, \
    entity2vec, entity2vec_dim, \
    relation2vec, relation2vec_dim, \
    train_triple, train_confidence, \
    test_triple, test_confidence, \
    test_triple_KGC_h_t, test_confidence_KGC_h_t, \
    test_triple_KGC_hr_, test_confidence_KGC_hr_, \
    test_triple_KGC__rt, test_confidence_KGC__rt, \
    tcthreshold_dict, train_transE, test_transE, \
    test_transE_h_t, \
    test_transE_hr_, \
    test_transE__rt, \
    rrkthreshold_dict, train_rrank, test_rrank, \
    test_rrank_KGC_h_t, \
    test_rrank_KGC_hr_, \
    test_rrank_KGC__rt, \
    max_p, \
    train_path_h, train_path_t, train_path_r, \
    test_path_h, test_path_t, test_path_r, \
    train_path2_h, train_path2_t, train_path2_r,\
    test_path2_h, test_path2_t, test_path2_r,\
    train_path3_h, train_path3_t, train_path3_r,\
    test_path3_h, test_path3_t, test_path3_r,\
    test_path_h_KGC_h_t, test_path_t_KGC_h_t, test_path_r_KGC_h_t, \
    test_path_h_hr_, test_path_t_hr_, test_path_r_hr_, \
    test_path_h__rt, test_path_t__rt, test_path_r__rt = pickle.load(open(datafile, 'rb'))

    input_train_h = np.zeros((len(train_triple),1)).astype('int32')
    input_train_t = np.zeros((len(train_triple),1)).astype('int32')
    input_train_r = np.zeros((len(train_triple),1)).astype('int32')
    for idx, s in enumerate(train_triple):
        input_train_h[idx,] = train_triple[idx][0]
        input_train_t[idx,] = train_triple[idx][1]
        input_train_r[idx,] = train_triple[idx][2]

    input_test_h = np.zeros((len(test_triple),1)).astype('int32')
    input_test_t = np.zeros((len(test_triple),1)).astype('int32')
    input_test_r = np.zeros((len(test_triple),1)).astype('int32')
    for idx, tri in enumerate(test_triple):
        input_test_h[idx,] = tri[0]
        input_test_t[idx,] = tri[1]
        input_test_r[idx,] = tri[2]


    nn_model = SelectModel(modelname, entvocabsize = len(ent_vocab), relvocabsize = len(rel_vocab),
                           ent2vec = entity2vec, rel2vec = relation2vec,
                input_path_lenth = max_p,
                ent_emd_dim = entity2vec_dim, rel_emd_dim = relation2vec_dim)

    if retrain:
        nn_model.load_weights(modelfile)

    nn_model.summary()


    # x_word, y, x_word_val, y_val= get_training_xy_otherset(x_train, y_train,
    #                                                         x_dev, y_dev,
    #                                                         max_s,
    #                                                         len(target_vob), target_idex_word, shuffle=True)
    # # x_word, x_posl, x_entl, y, x_word_val, x_posl_val, x_entl_val, y_val = get_training_xy(x_train, poslabel_train,
    # #                                                                                        entlabel_train, y_train,
    # #                                                                                        max_s, max_s,
    # #                                                                                        len(target_vob),
    # #                                                                                        target_idex_word,
    # #                                                                                        shuffle=True)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=8)
    # checkpointer = ModelCheckpoint(filepath="./data/model/best_model.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)
    # nn_model.fit(x_word, y,
    #              batch_size=batch_size,
    #              epochs=npochos,
    #              verbose=1,
    #              shuffle=True,
    #              # validation_split=0.2,
    #              validation_data=(x_word_val, y_val),
    #              callbacks=[reduce_lr, checkpointer, early_stopping])
    #
    # save_model(nn_model, modelfile)
    # # nn_model.save(modelfile, overwrite=True)
    epoch = 0
    save_inter = 1
    saveepoch = save_inter
    maxF = 0
    earlystopping =0

    while (epoch < npochos):
        epoch = epoch + 1

        # for input_train_h, input_train_t, input_train_r, \
        #     input_dev_h, input_dev_t, input_dev_r,\
        #     input_train_path_h, input_train_path_t, input_train_path_r, y, \
        #     input_dev_path_h, input_dev_path_t, input_dev_path_r, y_dev, \
        #     input_train_transE, input_dev_transE,\
        #     input_train_rrank, input_dev_rrank in get_training_xy(train_triple, train_confidence,
        #                                     train_transE,
        #                                     train_rrank,
        #                                     train_path_h, train_path_t, train_path_r,
        #                                     max_p,
        #                                     shuffle=True):
        history = nn_model.fit([
            np.array(input_train_h), np.array(input_train_t), np.array(input_train_r),
            np.array(train_path_h), np.array(train_path_t), np.array(train_path_r),
            np.array(train_path2_h), np.array(train_path2_t), np.array(train_path2_r),
            np.array(train_path3_h), np.array(train_path3_t), np.array(train_path3_r),
            np.array(train_transE),
            np.array(train_rrank)
            ],
            np.array(train_confidence),
                         batch_size=batch_size,
                         epochs=1,
                         validation_split=0.2,
                         shuffle=True,
                         verbose=1)

            # plt.plot(history.history['acc'])
            # plt.plot(history.history['val_acc'])
            # plt.title('model accuracy')
            # plt.ylabel('accuracy')
            # plt.xlabel('epoch')
            # plt.legend(['train', 'test'], loc='upper left')
            # plt.show()
            # # summarize history for loss
            # plt.plot(history.history['loss'])
            # plt.plot(history.history['val_loss'])
            # plt.title('model loss')
            # plt.ylabel('loss')
            # plt.xlabel('epoch')
            # plt.legend(['train', 'test'], loc='upper left')
            # plt.show()

        if epoch >= saveepoch:
            # if epoch >=0:
            saveepoch += save_inter
            resultfile = resultdir+"result-"+str(saveepoch)

            print('the test result-----------------------')
            acc = test_model(nn_model,
                             input_test_h, input_test_t, input_test_r,
                             test_path_h, test_path_t, test_path_r,
                             test_path2_h, test_path2_t, test_path2_r,
                             test_path3_h, test_path3_t, test_path3_r,
                             test_transE, test_rrank, test_confidence, resultfile)

            if acc > maxF:
                earlystopping = 0
                maxF=acc
                save_model(nn_model, modelfile)
            else:
                earlystopping += 1

            print(epoch, acc, '  maxF=', maxF)

        if earlystopping >= 20:
            break

    return nn_model


def save_model(nn_model, NN_MODEL_PATH):
    nn_model.save_weights(NN_MODEL_PATH, overwrite=True)

def test_model(model,
               input_test_h, input_test_t, input_test_r,
               test_path_h, test_path_t, test_path_r,
               test_path2_h, test_path2_t, test_path2_r,
               test_path3_h, test_path3_t, test_path3_r,
               test_transE, test_rrank, test_confidence, resultfile):

    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.
    results = model.predict([
        np.array(input_test_h), np.array(input_test_t), np.array(input_test_r),
        np.array(test_path_h), np.array(test_path_t), np.array(test_path_r),
        np.array(test_path2_h), np.array(test_path2_t), np.array(test_path2_r),
        np.array(test_path3_h), np.array(test_path3_t), np.array(test_path3_r),

        np.array(test_transE),
        np.array(test_rrank)
        ], batch_size=40)

    fin0 = open(resultfile + 'train_conf0.txt','w')
    fin1 = open(resultfile + 'train_conf1.txt', 'w')
    # print(results)
    for i, res in enumerate(results):
        tag = np.argmax(res)
        # print(res)
        # tag = 0
        # if res >=0.5:
        #     tag = 1
        if test_confidence[i][1] == 1:
            fin1.write(str(res[1]) + '\n')
            if tag == 1:
                total_predict_right += 1.0
        else:
            fin0.write(str(res[1]) + '\n')
            if tag == 0:
                total_predict_right += 1.0
    fin0.close()
    fin1.close()
    print('total_predict_right', total_predict_right, 'len(test_confidence)', float(len(test_confidence)))
    acc = total_predict_right / float(len(test_confidence))
    return acc


def test_model_PR(model,
                     input_test_h, input_test_t, input_test_r,
                     test_path_h, test_path_t, test_path_r,
                     test_path2_h, test_path2_t, test_path2_r,
                     test_path3_h, test_path3_t, test_path3_r,
                     test_transE, test_rrank, test_confidence,resultfile):

    results = model.predict([
        np.array(input_test_h), np.array(input_test_t), np.array(input_test_r),
        np.array(test_path_h), np.array(test_path_t), np.array(test_path_r),
        np.array(test_path2_h), np.array(test_path2_t), np.array(test_path2_r),
        np.array(test_path3_h), np.array(test_path3_t), np.array(test_path3_r),
        np.array(test_transE),
        np.array(test_rrank)
        ], batch_size=40)

    # fw = open(resultfile + 'RP.txt', 'w')
    # total_predict_right = 0.
    # Plist0 = []
    # Rlist0 = []
    # PRlist = []
    # maxF = 0.0
    # for i, tri in enumerate(test_confidence):
    #     PRlist.append((results[i][1], test_confidence[i][1]))
    # PRlist = sorted(PRlist, key=lambda sp: sp[0], reverse=True)
    #
    # for i, res in enumerate(PRlist):
    #     if res[1] == 1:
    #         total_predict_right += 1.0
    #     if (i + 1) % (3470) == 0:
    #         P = total_predict_right / (i + 1)
    #         R = total_predict_right / (len(PRlist) * 0.5)
    #         F = 2 * P * R / (P + R)
    #         if maxF < F:
    #             maxF = F
    #         Rlist0.append(R)
    #         Plist0.append(P)
    #         print(total_predict_right / (i + 1), '\t', total_predict_right / (len(PRlist) * 0.5), 'maxF = ',maxF)
    #         fw.write(str(R) + '\t' + str(P) + '\n')
    # fw.close()


    Thresholdlist = []
    Plist1 = []
    Rlist1 = []
    Plist0 = []
    Rlist0 = []
    maxF = 0.0
    th = 0.01
    # fw = open(resultfile + 'RP.txt', 'w')
    while th <= 1.0:
        total_predict_right = 0.0
        total_predict_right0 = 0.0
        total_predict = 0.00001
        total_right = 0.00001
        for i, res in enumerate(results):
            tag = 0
            if res[1] >= th:
                tag = 1
                total_predict += 1.0

            if test_confidence[i][1] == 1:
                if tag == 1:
                    total_predict_right += 1.0
                total_right += 1.0
            else:
                if tag == 0:
                    total_predict_right0 += 1.0

        P0 = total_predict_right0 / (len(results) - total_predict)
        R0 = total_predict_right0 / (len(results) - total_right)
        Plist0.append(P0)
        Rlist0.append(R0)
        P = total_predict_right / total_predict
        R = total_predict_right / total_right
        F = 2 * P * R / (P + R +0.00001)
        if maxF < F:
            maxF = F
        print('threshold = ', th, R, P, 'maxF = ', maxF)
        Thresholdlist.append(th)
        Rlist1.append(R)
        Plist1.append(P)
        # fw.write(str(R) + '\t' + str(P) + '\n')

        th += 0.02

    # fw.close()

    a = plt.subplot(1, 1, 1)

    # 这里b表示blue，g表示green，r表示red，-表示连接线，--表示虚线链接
    a1 = a.plot(Thresholdlist, Plist1, 'b-', label='Precision')
    a2 = a.plot(Thresholdlist, Rlist1, 'r-', label='Reall')
    # a3 = a.plot(Thresholdlist, Plist0, 'b--', label='P_negitive')
    # a4 = a.plot(Thresholdlist, Rlist0, 'r--', label='R_negitive')
    # a3 = a.plot(Thresholdlist, Flist, 'y-', label='F')
    # a4 = a.plot(Rlist, Plist, 'ro-', label='R-P')

    # 标记图的题目，x和y轴
    # plt.title("The Precision, Recall, F-values during different triple confidence threshold")
    plt.xlabel("Triple Trustworthiness")
    plt.ylabel("Values")

    # 显示图例
    handles, labels = a.get_legend_handles_labels()
    a.legend(handles[::-1], labels[::-1])
    plt.show()

    # b = plt.subplot(1, 1, 1)
    # # 这里b表示blue，g表示green，r表示red，-表示连接线，--表示虚线链接
    # # b1 = b.plot(Rlist, Plist, 'bx-', label='R-P')
    # b2 = b.plot(Rlist0, Plist0, 'go-', label='R-F')
    # # 标记图的题目，x和y轴
    # # plt.title("The Precision, Recall, F-values during different triple confidence threshold")
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # # 显示图例
    # handles, labels = b.get_legend_handles_labels()
    # b.legend(handles[::-1], labels[::-1])
    # plt.show()


def test_model_linkPrediction(model, datafile, entityRank):

    ent_vocab, rel_vocab, \
    entity2vec, entity2vec_dim, \
    relation2vec, relation2vec_dim, \
    train_triple, train_confidence, \
    test_triple, test_confidence, \
    tcthreshold_dict, train_transE, test_transE,\
    rrkthreshold_dict, train_rrank, test_rrank,\
    max_p, \
    train_path_h, train_path_t, train_path_r,\
    test_path_h, test_path_t, test_path_r = pickle.load(open(datafile, 'rb'))
    dict_entityRank = get_dict_entityRank(entityRank)
    goldtriples = get_goldtriples()

    totalRawHeadRank = 0.
    totalRawTailRank = 0.
    totalFilterHeadRank = 0.
    totalFilterTailRank = 0.

    totalRawHeadHit10 = 0.
    totalRawTailHit10 = 0.
    totalRawHeadHit1 = 0.
    totalRawTailHit1 = 0.

    totalFilterHeadHit10 = 0.
    totalFilterTailHit10 = 0.
    totalFilterHeadHit1 = 0.
    totalFilterTailHit1 = 0.

    rawTailList = []
    rawHeadList = []
    filterTailList = []
    filterHeadList = []

    for i in range(len(test_triple)):
        print(i)
        rawTailList.clear()
        filterTailList.clear()
        changetriples = []
        for corruptedTailEntity in ent_vocab.values():
            changetriples.append((test_triple[i][0], corruptedTailEntity, test_triple[i][2], 1))

        transE = get_TransConfidence(tcthreshold_dict, changetriples, entity2vec, relation2vec)
        rrank = get_RRankConfidence(rrkthreshold_dict, changetriples, dict_entityRank)

        results = model.predict([np.array(transE), np.array(rrank)])

        for r in range(len(results)):
            rawTailList.append((changetriples[r][1], results[r][1]))
            if (changetriples[r][0], changetriples[r][1], changetriples[r][2], 1) not in goldtriples:
                filterTailList.append((changetriples[r][1], results[r][1]))

        rawTailList = sorted(rawTailList, key=lambda sp: sp[1], reverse=True)
        filterTailList = sorted(filterTailList, key=lambda sp: sp[1], reverse=True)
        for j, tri in enumerate(rawTailList):
            j = j+1
            if tri[0] == test_triple[1]:
                totalRawTailRank += j
                if j <= 10:
                    totalRawTailHit10 +=1.0
                if j == 1:
                    totalRawTailHit1 +=1.0
                break
        for j, tri in enumerate(filterTailList):
            j = j+1
            if tri[0] == test_triple[1]:
                totalFilterTailRank += j
                if j <= 10:
                    totalFilterTailHit10 += 1.0
                if j == 1:
                    totalFilterTailHit1 += 1.0
                break

    for i in range(len(test_triple)):
        print(i)
        rawHeadList.clear()
        filterHeadList.clear()
        changetriples = []
        for corruptedHeadEntity in ent_vocab.values():
            changetriples.append((corruptedHeadEntity, test_triple[i][1], test_triple[i][2], 1))

        transE = get_TransConfidence(tcthreshold_dict, changetriples, entity2vec, relation2vec)
        rrank = get_RRankConfidence(rrkthreshold_dict, changetriples, dict_entityRank)

        results = model.predict([np.array(transE), np.array(rrank)])

        for r in range(len(results)):
            rawHeadList.append((changetriples[r][1], results[r][1]))
            if (changetriples[r][0], changetriples[r][1], changetriples[r][2], 1) not in goldtriples:
                filterHeadList.append((changetriples[r][1], results[r][1]))

        rawHeadList = sorted(rawHeadList, key=lambda sp: sp[1], reverse=True)
        filterHeadList = sorted(filterHeadList, key=lambda sp: sp[1], reverse=True)
        for j, tri in enumerate(rawHeadList):
            j = j + 1
            if tri[0] == test_triple[1]:
                totalRawHeadRank += j
                if j <= 10:
                    totalRawHeadHit10 += 1.0
                if j == 1:
                    totalRawHeadHit1 += 1.0
                break
        for j, tri in enumerate(filterHeadList):
            j = j + 1
            if tri[0] == test_triple[1]:
                totalFilterHeadRank += j
                if j <= 10:
                    totalFilterHeadHit10 += 1.0
                if j == 1:
                    totalFilterHeadHit1 += 1.0
                break

    print("RAW_RANK: ", (totalRawTailRank + totalRawHeadRank) / float(2. * len(test_triple)))
    print("FILTER_RANK: ", (totalFilterHeadRank + totalFilterTailRank) / float(2. * len(test_triple)))
    print("RAW_HIT@10: ", (totalRawTailHit10 + totalRawHeadHit10) / float(2. * len(test_triple)))
    print("FILTER_HIT@10: ", (totalFilterHeadHit10 + totalFilterTailHit10) / float(2. * len(test_triple)))
    print("RAW_HIT@1: ", (totalRawHeadHit1 + totalRawTailHit1) / float(2. * len(test_triple)))
    print("FILTER_HIT@1: ", (totalFilterHeadHit1 + totalFilterTailHit1) / float(2. * len(test_triple)))


def test_model_load(model, ent_idex_word, rel_idex_word, test_triple,
                    test_path_h, test_path_t, test_path_r,
                    test_transE, test_rrank,
                    resultfile):


    results = model.predict([
        np.array(test_path_h), np.array(test_path_t), np.array(test_path_r),
                            np.array(test_transE),
                            np.array(test_rrank)
    ], batch_size=40)


    # print(results)
    All_conf = 0.0
    fr = open(resultfile,'w')
    for i, res in enumerate(results):
        conf = res[1]
        # print(conf)
        All_conf += conf
        strs = ent_idex_word[test_triple[i][0]] + '\t' + rel_idex_word[test_triple[i][2]] +\
              '\t' + ent_idex_word[test_triple[i][1]] + '\t' + str(conf) + '\n'
        fr.write(strs)
    fr.close()

    avg_conf = All_conf / float(len(results))
    print('avg_conf is ... ', avg_conf)




def infer_model(modelname, entityRank, datafile, modelfile, resultfile, batch_size=50):

    ent_vocab, ent_idex_word, rel_vocab, rel_idex_word, \
    entity2vec, entity2vec_dim, \
    relation2vec, relation2vec_dim, \
    train_triple, train_confidence, \
    test_triple, test_confidence, \
    test_triple_KGC_h_t, test_confidence_KGC_h_t, \
    test_triple_KGC_hr_, test_confidence_KGC_hr_, \
    test_triple_KGC__rt, test_confidence_KGC__rt, \
    tcthreshold_dict, train_transE, test_transE, \
    test_transE_h_t, \
    test_transE_hr_, \
    test_transE__rt, \
    rrkthreshold_dict, train_rrank, test_rrank, \
    test_rrank_KGC_h_t, \
    test_rrank_KGC_hr_, \
    test_rrank_KGC__rt, \
    max_p, \
    train_path_h, train_path_t, train_path_r, \
    test_path_h, test_path_t, test_path_r, \
    train_path2_h, train_path2_t, train_path2_r,\
    test_path2_h, test_path2_t, test_path2_r,\
    train_path3_h, train_path3_t, train_path3_r,\
    test_path3_h, test_path3_t, test_path3_r,\
    test_path_h_KGC_h_t, test_path_t_KGC_h_t, test_path_r_KGC_h_t, \
    test_path_h_KGC_hr_, test_path_t_KGC_hr_, test_path_r_KGC_hr_, \
    test_path_h_KGC__rt, test_path_t_KGC__rt, test_path_r_KGC__rt = pickle.load(open(datafile, 'rb'))

    input_train_h = np.zeros((len(train_triple),1)).astype('int32')
    input_train_t = np.zeros((len(train_triple),1)).astype('int32')
    input_train_r = np.zeros((len(train_triple),1)).astype('int32')
    for idx, s in enumerate(train_triple):
        input_train_h[idx,] = train_triple[idx][0]
        input_train_t[idx,] = train_triple[idx][1]
        input_train_r[idx,] = train_triple[idx][2]

    input_test_h = np.zeros((len(test_triple),1)).astype('int32')
    input_test_t = np.zeros((len(test_triple),1)).astype('int32')
    input_test_r = np.zeros((len(test_triple),1)).astype('int32')
    for idx, tri in enumerate(test_triple):
        input_test_h[idx,] = tri[0]
        input_test_t[idx,] = tri[1]
        input_test_r[idx,] = tri[2]

    model = SelectModel(modelname, entvocabsize = len(ent_vocab), relvocabsize = len(rel_vocab),
                           ent2vec = entity2vec, rel2vec = relation2vec,
                input_path_lenth = max_p,
                ent_emd_dim = entity2vec_dim, rel_emd_dim = relation2vec_dim)

    model.load_weights(modelfile)
    # nnmodel = load_model(lstm_modelfile)

    # acc = test_model(model,
    #                  input_test_h, input_test_t, input_test_r,
    #                  test_path_h, test_path_t, test_path_r,
    #                  test_path2_h, test_path2_t, test_path2_r,
    #                  test_path3_h, test_path3_t, test_path3_r,
    #                  test_transE, test_rrank, test_confidence, resultfile)
    # print(acc)

    acc = test_model(model,
                     input_train_h, input_train_t, input_train_r,
                  train_path_h, train_path_t, train_path_r,
                  train_path2_h, train_path2_t, train_path2_r,
                  train_path3_h, train_path3_t, train_path3_r,
                  train_transE, train_rrank, train_confidence, resultfile)
    print(acc)

    # test_model_PR(model,
    #                  input_test_h, input_test_t, input_test_r,
    #                  test_path_h, test_path_t, test_path_r,
    #                  test_path2_h, test_path2_t, test_path2_r,
    #                  test_path3_h, test_path3_t, test_path3_r,
    #                  test_transE, test_rrank, test_confidence, resultfile)

    # test_model_PR(model,
    #                  input_train_h, input_train_t, input_train_r,
    #               train_path_h, train_path_t, train_path_r,
    #               train_path2_h, train_path2_t, train_path2_r,
    #               train_path3_h, train_path3_t, train_path3_r,
    #               train_transE, train_rrank, train_confidence, resultfile)

    # test_model_linkPrediction(model, datafile, entityRank)

    # test_model_load(model, test_triple, ent_idex_word, rel_idex_word, test_path_h, test_path_t, test_path_r, test_transE, test_rrank, test_confidence, resultfile)
    ''''
    acc = test_model(model,
                     test_path_h_KGC_hr_, test_path_t_KGC_hr_, test_path_r_KGC_hr_,
                     test_transE_hr_, test_rrank_KGC_hr_,
                     test_confidence_KGC_hr_, resultfile)
    print('hr_ acc ... ', acc)
    test_model_load(model, ent_idex_word, rel_idex_word,
                    test_triple_KGC_hr_,
                    test_path_h_KGC_hr_, test_path_t_KGC_hr_, test_path_r_KGC_hr_,
                    test_transE_hr_, test_rrank_KGC_hr_,
                    resultfile+'_hr__conf.txt')
    
    acc = test_model(model,
                     test_path_h_KGC_h_t, test_path_t_KGC_h_t, test_path_r_KGC_h_t,
                     test_transE_h_t, test_rrank_KGC_h_t,
                     test_confidence_KGC_h_t, resultfile)
    print('h_t acc ... ', acc)
    test_model_load(model, ent_idex_word, rel_idex_word,
                    test_triple_KGC_h_t,
                    test_path_h_KGC_h_t, test_path_t_KGC_h_t, test_path_r_KGC_h_t,
                    test_transE_h_t, test_rrank_KGC_h_t,
                    resultfile+'_h_t_conf.txt')

    acc = test_model(model,
                     test_path_h_KGC__rt, test_path_t_KGC__rt, test_path_r_KGC__rt,
                     test_transE__rt, test_rrank_KGC__rt,
                     test_confidence_KGC__rt, resultfile)
    print('_rt acc ... ', acc)
    test_model_load(model, ent_idex_word, rel_idex_word,
                    test_triple_KGC__rt,
                    test_path_h_KGC__rt, test_path_t_KGC__rt, test_path_r_KGC__rt,
                    test_transE__rt, test_rrank_KGC__rt,
                    resultfile+'__rt_conf.txt')
    '''


def get_goldtriples():
    path = "/Users/shengbinjia/Documents/GitHub/TCdata/FB15K/golddataset/"
    goldtriples = []

    files = os.listdir(path)
    for file in files:
        # print(file)
        fo = open(path + file, 'r')
        lines = fo.readlines()
        for line in lines:

            nodes = line.rstrip('\n').split(' ')
            goldtriples.append((int(nodes[0]), int(nodes[1]), int(nodes[2])))
        fo.close()
    return goldtriples

if __name__ == "__main__":

    modelname = 'creat_Model_BiLSTM_BP'

    print(modelname)
    file_data = "/Users/shengbinjia/Documents/GitHub/TCdata"

    entity2idfile = file_data + "/FB15K/entity2id.txt"
    relation2idfile = file_data + "/FB15K/relation2id.txt"

    entity2vecfile =file_data + "/FB15K_TransE_Entity2Vec_100.txt"
    relation2vecfile = file_data + "/FB15K_TransE_Relation2Vec_100.txt"

    # entity2vecfile =file_data + "/FB15K_PTransE_Entity2Vec_100.txt"
    # relation2vecfile = file_data + "/FB15K_PTransE_Relation2Vec_100.txt"

    # entity2vecfile =file_data + "/FB15K_TransH_Entity2Vec_100.txt"
    # relation2vecfile = file_data + "/FB15K_TransH_Relation2Vec_100.txt"

    trainfile = file_data + "/KBE/datasets/FB15k/conf_train2id.txt"
    # devfile = file_data + "/KBE/datasets/FB15k/test2id.txt"
    testfile = file_data + "/KBE/datasets/FB15k/conf_test2id.txt"
    testfile_KGC_h_t = file_data + "/FB15K/KBCdataset/h_t.txt"
    testfile_KGC_hr_ = file_data + "/FB15K/KBCdataset/hr_.txt"
    testfile_KGC__rt = file_data + "/FB15K/KBCdataset/_rt.txt"

    path_file = file_data + "/Path_4/"
    entityRank = file_data + "/ResourceRank_4/"

    datafile = "./model/data2_TransE.pkl"
    modelfile = "./model/model2_TransE.h5"
    resultdir = "./data/result/"
    resultdir = "./data/result/Model1_model_TransE_---"

    batch_size = 64
    retrain = False
    Test = True
    valid = False
    Label = False
    if not os.path.exists(datafile):
        print("Precess data....")
        get_data(entity2idfile=entity2idfile, relation2idfile=relation2idfile,
             entity2vecfile=entity2vecfile, relation2vecfile=relation2vecfile, w2v_k=100,
             trainfile=trainfile, testfile=testfile,
             testfile_KGC_h_t=testfile_KGC_h_t,
             testfile_KGC_hr_=testfile_KGC_hr_,
             testfile_KGC__rt=testfile_KGC__rt,
             path_file=path_file, max_p=3,
             entityRank=entityRank,
             datafile=datafile)
    if not os.path.exists(modelfile):
        print("data has extisted: " + datafile)
        print("Training model....")
        print(modelfile)
        train_model(modelname, datafile, modelfile, resultdir,
                        npochos=200, batch_size=batch_size, retrain=False)
    else:
        if retrain:
            print("ReTraining EE model....")
            train_model(modelname, datafile, modelfile, resultdir,
                            npochos=200, batch_size=batch_size, retrain=retrain)

    if Test:
        print("test EE model....")
        print(modelfile)
        infer_model(modelname, entityRank, datafile, modelfile, resultdir, batch_size=batch_size)
