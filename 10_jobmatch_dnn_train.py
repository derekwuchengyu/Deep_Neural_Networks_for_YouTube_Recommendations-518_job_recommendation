import sys,os
# import time
import datetime
import numpy as np 
import pandas as pd
import shutil
import pickle
sys.path.append('/home/htdocs/apiworker/')
from utils.module import *
from utils.airmodule import *
from p518.config import *


import math
import tensorflow as tf
from gensim.models import word2vec
from sklearn.model_selection import train_test_split

from clickhouse_driver import connect,Client
conn = connect('clickhouse://192.168.1.42')

MODEL_PATH = BASE_DIR + "/model/"
NN_MODEL_CONFIG = MODEL_PATH + '10_jobmatch_youtube_nn.pkl'
NN_MODEL_WEIGHT = MODEL_PATH + '10_jobmatch_youtube_nn.npy'
JS_MODEL_FILE = MODEL_PATH + '10_w2v_js_id.model'
JC_MODEL_FILE = MODEL_PATH + '10_w2v_job_class.model'
JL_MODEL_FILE = MODEL_PATH + '10_w2v_job_place.model'

jsW2v_model = word2vec.Word2Vec.load(JS_MODEL_FILE)
jlW2v_model = word2vec.Word2Vec.load(JL_MODEL_FILE)
jcW2v_model = word2vec.Word2Vec.load(JC_MODEL_FILE)

def getWeight(model):
    # 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
    embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
    embeddings_matrix[1:] = model.wv.syn0
    return embeddings_matrix

js_embeddings_matrix = getWeight(jsW2v_model)
jl_embeddings_matrix = getWeight(jlW2v_model)
jc_embeddings_matrix = getWeight(jcW2v_model)


# 自定義function
class W2VConvert(ClassW2VConvert):
    def __init__(self,model):
        super().__init__(model)




def getTrainingData():
    # Select the user_images with apply_hist & expect_class
    train_data = pd.read_sql(f'SELECT * FROM recomm.518_user_image \
                               WHERE apply_hist != [\'0\'] \
                               AND apply_hist != [] \
                               AND expect_class != [\'0\']\
                               AND i_time >= (SELECT max(i_time) FROM recomm.518_user_image LIMIT 1) ORDER BY i_time DESC LIMIT 30000',conn)
    print("Traning data amount: %s" % len(train_data))
    # print(train_data.head(2))
    
    # 轉成token (word2vec的index)
    train_data['apply_hist']   = train_data['apply_hist'].apply(W2VConvert(model=jsW2v_model).token)
    train_data['apply_class']  = train_data['apply_class'].apply(W2VConvert(model=jcW2v_model).token)
    train_data['apply_locat']  = train_data['apply_locat'].apply(W2VConvert(model=jlW2v_model).token)
    train_data['expect_class'] = train_data['expect_class'].apply(W2VConvert(model=jcW2v_model).token)
    train_data['expect_locat'] = train_data['expect_locat'].apply(W2VConvert(model=jlW2v_model).token)
    train_data['exp_job']      = train_data['exp_job'].apply(W2VConvert(model=jcW2v_model).token)
    
    # 移除最後一個apply物件當Label
    train_data['label'] = train_data['apply_hist'].apply(lambda x: x[0])
    train_data['apply_hist'] = train_data['apply_hist'].apply(lambda x: x[1:])
    train_data['apply_hist_len'] = train_data['apply_hist'].apply(len)
    train_data['expect_class_len'] = train_data['expect_class'].apply(len)
    train_data = train_data[train_data['label']>0]
    train_data = train_data[train_data['apply_hist_len']>0]
    train_data = train_data[train_data['expect_class_len']>0]
    
    # 避免數據洩漏(js_id token 被過濾掉，但apply_class仍保留的情況)
    train_data['apply_class'] = train_data.apply(lambda x: x['apply_class'][-x['apply_hist_len']:],axis=1)
    train_data['apply_locat'] = train_data.apply(lambda x: x['apply_locat'][-x['apply_hist_len']:],axis=1)
    # print(train_data['label'])
    
    # train_data = train_data.query('apply_hist!=\[\]') #[len(train_data['apply_hist'])!=0]

    train_data = train_data[['m_id','apply_hist',
                                'apply_class',
                                'apply_locat',
                                'expect_class',
                                'expect_locat','exp_job',
                                'gender','exp_year',
                                'label']]
    print(train_data.head(2))
    return train_data
  
    
def trainModel():

    #--- 參數
    EMBEDDING_DIMS = 32
    DENSE_UNITS = 64
    DROPOUT_PCT = 0.0
    ALPHA = 0.0
    NUM_CLASSES=len(js_embeddings_matrix) + 1
    LEARNING_RATE = 0.003

    #--- Feature Used
    FEATURE  = ['apply_hist','apply_class','apply_locat','expect_class','expect_locat','gender','exp_year','exp_job']
    FEATURE_USED = ['apply_hist',
                    'apply_class',
                    # 'apply_locat',
                    'expect_class',
                    # 'expect_locat',
                    'exp_job',
                    'gender',
                    'exp_year'
                   ]

    #--- Embedding Featrues
    PAD_FEATURES = ['apply_hist','apply_class','apply_locat','expect_class','expect_locat','exp_job']
    NUM_FEATURES = ['gender','exp_year']
    FEATURES = {}
    for i in FEATURE:
        FEATURES[i] = {}



    #--- input
    FEATURES['apply_hist']['input']   = tf.keras.layers.Input(shape=(10,), name='apply_hist')
    FEATURES['apply_class']['input']  = tf.keras.layers.Input(shape=(10,), name='apply_class')
    FEATURES['apply_locat']['input']  = tf.keras.layers.Input(shape=(10,), name='apply_locat')
    FEATURES['expect_class']['input'] = tf.keras.layers.Input(shape=(10,), name='expect_class')
    FEATURES['expect_locat']['input'] = tf.keras.layers.Input(shape=(10,), name='expect_locat')
    FEATURES['exp_job']['input']      = tf.keras.layers.Input(shape=(10,), name='exp_job')
    FEATURES['exp_year']['input']     = tf.keras.layers.Input(shape=(1,), name='exp_year')
    FEATURES['gender']['input']       = tf.keras.layers.Input(shape=(1,), name='gender')


    #--- Embedding layers
    embedding = lambda embedding_matrix,name: (
            tf.keras.layers.Embedding(input_dim =embedding_matrix.shape[0],
                                      output_dim =embedding_matrix.shape[1], 
                                      weights =[embedding_matrix],
                                      mask_zero=True, 
                                      name=name)
    )

    # 因為 要處理0 向量，暫無使用word2vec.wv.get_keras_embedding()
    js_embedding_layer = embedding(js_embeddings_matrix,'js_embeddings')
    jc_embedding_layer = embedding(jc_embeddings_matrix,'jc_embeddings')
    jl_embedding_layer = embedding(jl_embeddings_matrix,'jl_embeddings')


    #--- Avg Embedding layers
    avg_embeddings = MaskedEmbeddingsAggregatorLayer(agg_mode='mean', name='aggregate_embeddings')

    #--- Dense layers
    dense_0 = tf.keras.layers.Dense(units=DENSE_UNITS*4, name='dense_0')
    dense_1 = tf.keras.layers.Dense(units=DENSE_UNITS*2, name='dense_1')
    dense_2 = tf.keras.layers.Dense(units=DENSE_UNITS*2, name='dense_2')
    dense_3 = tf.keras.layers.Dense(units=DENSE_UNITS*1, name='dense_3')
    l2_norm_1 = L2NormLayer(name='l2_norm_1')

    # l2_norm_apply_hist = l2_norm_1(input_apply_hist)
    # l2_norm_want_locat = l2_norm_1(input_want_locat)

    #--- 輸出層
    dense_output = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.nn.softmax, name='dense_output')


    #--- Features
    l2_avg_embedding = lambda eb_layer,input_layer: (
            avg_embeddings(l2_norm_1(eb_layer(input_layer)))
    )

    FEATURES['apply_hist']['concat_input']   = l2_avg_embedding(js_embedding_layer,FEATURES['apply_hist']['input'])
    FEATURES['apply_class']['concat_input']  = l2_avg_embedding(jc_embedding_layer,FEATURES['apply_class']['input'])
    FEATURES['apply_locat']['concat_input']  = l2_avg_embedding(jl_embedding_layer,FEATURES['apply_locat']['input'])
    FEATURES['expect_class']['concat_input'] = l2_avg_embedding(jc_embedding_layer,FEATURES['expect_class']['input'])
    FEATURES['expect_locat']['concat_input'] = l2_avg_embedding(jl_embedding_layer,FEATURES['expect_locat']['input'])
    FEATURES['exp_job']['concat_input']      = l2_avg_embedding(jc_embedding_layer,FEATURES['exp_job']['input'])
    FEATURES['exp_year']['concat_input']     = FEATURES['exp_year']['input']
    FEATURES['gender']['concat_input']       = FEATURES['gender']['input']


    # print(avg_apply_hist)
    # print(avg_jl)


    # 全連接層
    concat_inputs = tf.keras.layers.Concatenate(axis=1)([FEATURES[f]['concat_input'] for f in FEATURE_USED])


    # Dense Layers
    dense_0_features = dense_0(concat_inputs)
    dense_0_relu = tf.keras.layers.ReLU(name='dense_0_relu')(dense_0_features)

    dense_1_features = dense_1(dense_0_relu)
    dense_1_relu = tf.keras.layers.ReLU(name='dense_1_relu')(dense_1_features)
    # dense_1_batch_norm = tf.keras.layers.BatchNormalization(name='dense_1_batch_norm')(dense_1_relu)

    dense_2_features = dense_2(dense_1_relu)
    dense_2_relu = tf.keras.layers.ReLU(name='dense_2_relu')(dense_2_features)
    # dense_2_batch_norm = tf.keras.layers.BatchNormalization(name='dense_2_batch_norm')(dense_2_relu)

    dense_3_features = dense_3(dense_2_relu)
    dense_3_relu = tf.keras.layers.ReLU(name='dense_3_relu')(dense_3_features)
    dense_3_batch_norm = tf.keras.layers.BatchNormalization(name='dense_3_batch_norm')(dense_3_relu)
    outputs = dense_output(dense_3_batch_norm)

    #Optimizer
    optimiser = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    #--- prep model
    model = tf.keras.models.Model(
        inputs=[[FEATURES[f]['input'] for f in FEATURE_USED]],
        outputs=[outputs]
    )
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['acc'],run_eagerly=False)

    model.summary()
    # 模型圖
    # print(tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True,to_file='518_NN.png')))

    #--- 測試訓練資料
    
    train_data = getTrainingData()
    train, test = train_test_split(train_data, test_size=0.1)
    x_train = train.drop(columns=['label'])
    y_train = train['label']
    x_test = test.drop(columns=['label'])
    y_test = test['label']

    pad = lambda array,maxlen=10: (
            tf.keras.preprocessing.sequence.pad_sequences(
              array, maxlen=maxlen,padding="post"
          )+ 1e-12
    )

    for f in FEATURE:
        if f in PAD_FEATURES:
            FEATURES[f]['train'] = pad(x_train[f])
            FEATURES[f]['test'] = pad(x_test[f])
        else:
            FEATURES[f]['train'] = x_train[f].to_numpy()
            FEATURES[f]['test'] = x_test[f].to_numpy()
            

    history = model.fit([FEATURES[f]['train'] for f in FEATURE_USED],y_train,
                        steps_per_epoch=1, epochs=120)

    # Save Model
    nn_config = model.get_config()
    with open(NN_MODEL_CONFIG, 'wb') as handle:
        pickle.dump(nn_config, handle)

    ww = model.get_weights()
    with open(NN_MODEL_WEIGHT, 'wb') as f:
        np.save(f, np.array(ww))

    # Result
    predict =model.predict([FEATURES[f]['train'] for f in FEATURE_USED])
    predictions = np.array([np.argmax(a) for a in predict])
    x_train['predicted_label'] = predictions
    r = model.evaluate([FEATURES[f]['test'] for f in FEATURE_USED],y_test)
    print("TrainData正確率: %.3f %%" % (np.where(x_train['predicted_label']==y_train)[0].size/len(y_train) *100 ))
    print("TestData正確率: %.3f %%" % (r[1] *100 ))
      


if __name__ == "__main__":
    print("開始訓練")
    start_time = time.time()
    trainModel()
    print("518 Youtube NN training spent (%.2f seconds)" % (time.time() - start_time))