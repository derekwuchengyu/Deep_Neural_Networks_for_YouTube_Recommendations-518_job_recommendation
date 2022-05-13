'''
    對應@app.route('/')
    function define
'''

from math import exp
from utils.module import *
from utils.airmodule import *
from p518.config import *
from p518.common import *
import jieba, json , sys,time,pickle
from gensim.models import word2vec
from clickhouse_driver import connect,Client
conn = connect('clickhouse://192.168.1.42')

TOPN = 10


MODEL_PATH = BASE_DIR + "/model/"
NN_MODEL_CONFIG = MODEL_PATH + '10_jobmatch_youtube_nn.pkl'
NN_MODEL_WEIGHT = MODEL_PATH + '10_jobmatch_youtube_nn.npy'
JS_MODEL_FILE = MODEL_PATH + '10_js_id.model'
JC_MODEL_FILE = MODEL_PATH + '10_job_class.model'
JL_MODEL_FILE = MODEL_PATH + '10_job_place.model'
jsW2v_model = word2vec.Word2Vec.load(JS_MODEL_FILE)
jlW2v_model = word2vec.Word2Vec.load(JL_MODEL_FILE)
jcW2v_model = word2vec.Word2Vec.load(JC_MODEL_FILE)

# Load DNN model
with open(NN_MODEL_CONFIG, 'rb') as handle:
    nn_config = pickle.load(handle)
with open(NN_MODEL_WEIGHT, 'rb') as handle:
    nn_weight = np.load(handle, allow_pickle=True)
# ww = np.load(NN_MODEL_WEIGHT,allow_pickle=True)

model = tf.keras.Model.from_config(nn_config,custom_objects={'L2NormLayer': L2NormLayer,'MaskedEmbeddingsAggregatorLayer':MaskedEmbeddingsAggregatorLayer})
model.set_weights(nn_weight)


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

# 自定義function
class W2VC(ClassW2VConvert):
    def __init__(self,model):
        super().__init__(model)


def routeMain(request):
    resultCode = {'code':-1001,'sysmsg':'param error or no data','result':[]}
    try:
        if 1:
            if request.args.get('m_id'):
                m_ids = request.args.get('m_id')
                print("FROM GET")
            else:
                m_ids = json.loads(request.values['m_id']);

            m_ids = [m_ids] if isinstance(m_ids, str) else m_ids #一律轉成List型態
            m_ids_query = '\',\''.join(m_ids)
            print(m_ids_query)

            query_df =  pd.read_sql('SELECT * FROM recomm.518_user_image \
                                     WHERE m_id IN (\''+m_ids_query+'\') \
                                     ORDER BY i_time DESC LIMIT %s' % len(m_ids),conn)
            if len(query_df)==0:
                resultCode['code'] = -1002
                resultCode['sysmsg'] = "find no m_ids"
                resultCode['result'] = {}
                return resultCode

            # print(query_df)
            query_df['apply_hist']   = query_df['apply_hist'].apply(W2VC(model=jsW2v_model).token)
            query_df['apply_class']  = query_df['apply_class'].apply(W2VC(model=jcW2v_model).token)
            query_df['apply_locat']  = query_df['apply_locat'].apply(W2VC(model=jlW2v_model).token)
            query_df['expect_class'] = query_df['expect_class'].apply(W2VC(model=jcW2v_model).token)
            query_df['expect_locat'] = query_df['expect_locat'].apply(W2VC(model=jlW2v_model).token)
            query_df['exp_job']      = query_df['exp_job'].apply(W2VC(model=jcW2v_model).token)
            r_mids = query_df.m_id.tolist()


            pad = lambda array,maxlen=10: (
                    tf.keras.preprocessing.sequence.pad_sequences(
                      array, maxlen=maxlen,padding="post"
                  )+ 1e-12
            )

            FEATURES_2 = []
            for f in FEATURE_USED:
                if f in PAD_FEATURES:
                    FEATURES_2.append(pad(query_df[f]))
                else:
                    FEATURES_2.append(query_df[f].to_numpy())


            predict = model.predict(FEATURES_2)
            scores = [a[np.argpartition(a, -TOPN)[-TOPN:]] for a in predict]
            ranks = [np.argpartition(a, -TOPN)[-TOPN:] for a in predict]
            ranks = [W2VC(model=jsW2v_model).token_decode(k) for k in ranks]
            # print(scores)
            # print(ranks)

            segs = dict()
            for i,q_str in enumerate(r_mids):
                segs[str(q_str)] = dict()
                rank = 0
                res = dict(zip(ranks[i],scores[i]))
                res = sorted(res.items(),key=lambda x:x[1],reverse=True)
                for r in res:
                    jsId = r[0]
                    score = r[1]
                    segs[str(q_str)].update({rank:{"js_id":int(jsId),"score":float(score)}})
                    rank += 1

            for q_str in m_ids:
                if str(q_str) not in segs:
                    segs[str(q_str)] = {}

        resultCode['code'] = 0
        resultCode['sysmsg'] = "ok"
        resultCode['result'] = segs
        return resultCode
    except:
        resultCode['code'] = -2
        resultCode['sysmsg'] = echoError()
        return resultCode