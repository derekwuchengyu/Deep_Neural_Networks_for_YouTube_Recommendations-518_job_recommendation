import sys,os
import numpy as np
import pandas as pd
import datetime
import math
import re
from gensim.models import word2vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.callbacks import PerplexityMetric
from gensim.models.word2vec import train_sg_pair
from pprint import pprint 
from random import choice
import time
import memcache,redis
from scipy.special import expit,softmax
from collections import defaultdict
from google.cloud import bigquery as bq
from utils.module import *
from p591.config import *
from clickhouse_driver import Client
from clickhouse_driver import connect
import multiprocessing as mp
import tensorflow as tf
from dotenv import load_dotenv
load_dotenv()

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self,corpus,settings=None):
        self.tt = time.time()
        self.corpus = corpus
        self.epochs = 1 
        if settings:
            self.setting(**settings)
        # self.load_data()
    def pre_train(self,model):
        self.model = model
        self.w1 = self.model.wv.vectors 
        self.w2 = self.model.trainables.syn1neg 
        self.model.running_training_loss = 0
        self.size = min(self.w1.shape)
        if self.epochs == 1 :
            print("Epochs start")
            self.cum_table = self.model.vocabulary.cum_table
            self.word_counts = word_counts = {word:vocab.count for word,vocab in self.model.wv.vocab.items()}
            self.v_count = len(word_counts.keys()) # Unique總字數
            self.word_index = {word:vocab.index for word,vocab in self.model.wv.vocab.items()} # index-詞
            self.index_word = {vocab.index:word for word,vocab in self.model.wv.vocab.items()} # 詞-index 
            self.words_list = list(self.word_index.keys())
            # if self.negative>0:
            #     self.make_cum_table()
            # if self.local_negative>0:
            self.make_local_cum_table()
        # self.timerecord()      


    # local 負採樣映射表
    def make_local_cum_table(self, escape_list=[], domain=2**31 - 1):
        cum_table = {}
        cum_table_map = {}
        train_words_pow = {}
        cumulative = {}

        for word_index in range(self.v_count):
            word = self.index_word[word_index]
            group = word[:2]
            if group not in cum_table.keys():
                cum_table[group] = np.zeros(self.v_count)
                train_words_pow[group] = 0.0
                cumulative[group] = 0.0
                cum_table_map[group] = []
            if word[2:] not in escape_list:
                train_words_pow[group] += self.word_counts[word]**self.ns_exponent
                cum_table_map[group].append(self.word_index[word])
        for word_index in range(self.v_count):
            word = self.index_word[word_index]
            group = word[:2]
            if word[2:] not in escape_list:
                cumulative[group] += self.word_counts[word]**self.ns_exponent
                cum_table[group][word_index] = round(cumulative[group] / train_words_pow[group] * domain)  
        for group,value in cum_table.items():
            cum_table[group] = np.array([num for num in value.tolist() if num > 0])    
        self.local_cum_table = cum_table
        self.local_cum_table_map = cum_table_map
    def timerecord(self, string=""):
        print(str(string)+str(round(time.time()-self.tt,4)))
        self.tt = time.time()
 
    def getloss(self,center,context):
        w1=self.w1
        w2=self.w2
        pre = expit(np.dot(w2,w1[center].T))
        word_vec = [0 for i in range(0, self.v_count)]
        word_vec = np.array(word_vec)
        word_vec[context] = 1
        if math.isnan(np.sum(np.subtract(pre,word_vec))):
            print(center,context)
            print(w1)
            print(w2)
        return np.sum(np.subtract(pre,word_vec))
    
    def progesss_rate(self,index):
        total = len(self.corpus)
        if index % round(total/10) ==0 and index!=0 :
            print(round(index*100/total),"%")

    def setting(self,window_size=4,size=25,negative=5,learning_rate=0.005,ns_exponent=0.75,
                local_negative=0,neg_mtpl=1,glb_mtpl=1,local_neg_min=200):
        self.n = size
        self.lr = learning_rate
        self.window = window_size
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.local_negative = local_negative
        self.neg_mtpl = neg_mtpl
        self.glb_mtpl = glb_mtpl 
        self.local_neg_min = local_neg_min
    def weight_update(self, center, word_indices, tj, lr=-1):
        if lr == -1:
            lr = self.lr
        if len(word_indices)==0:
            pass
        h = self.w1[center]
        w2_tmp = self.w2[word_indices]
        u = np.dot(w2_tmp, h.T)
        y_pred = expit(u)
        EI = np.subtract(y_pred,tj)

        #  W1 更新 
        dlt_w1 = np.dot(EI,w2_tmp)
        self.w1[center] -= (lr * dlt_w1)
        
        # W2 更新 
        dlt_w2 = np.outer(EI,h.T)
        self.w2[word_indices] -= (lr * dlt_w2)
        
        # if np.sum(dlt_w1)>10000 or np.sum(dlt_w1)<-10000: # debug用
        #     print(center,"-",(dlt_w1))
        #     print(center,"-",EI)
        
        # 計算loss
        if np.random.randint(30000) < 1:
            lss = self.getloss(center,word_indices)
            # print("loss",lss)
            
    def weight_update2(self, context, word_indices, lr=-1,compute_loss=True):
        if lr == -1:
            lr = self.lr
        if len(word_indices)==0:
            pass

        l1 = self.w1[context] #(h)
        l2b = self.w2[word_indices] #(w2_tmp)
        neu1e = np.zeros(l1.shape)
        tj  = np.zeros(len(word_indices))
        tj[0] = 1
        
        # forward
        prod_term = np.dot(l1, l2b.T)
        EI = expit(prod_term)
        gb = (tj - EI) * lr
        
        #  W1 更新 
        self.w1[context] += np.dot(gb, l2b)
        
        # W2 更新 
        self.w2[word_indices] += np.outer(gb, l1) 
        if compute_loss:
            if expit(-1 * prod_term[1:])==0 or expit(prod_term[0])==0:
                pass
            self.model.running_training_loss -= np.sum(np.log(expit(-1 * prod_term[1:])))  # for the sampled words
            self.model.running_training_loss -= np.log(expit(prod_term[0]))  # for the output word
            
    def local_neg_update(self,center_word,context):
        if center_word not in self.word_index.keys():
            return False
        group = center_word[:2]
        t_index = self.word_index[center_word]
        
        if len(self.local_cum_table[group]) <self.local_neg_min: # 群太小不採樣，避免推錯
            return False

        tj = []                            
        local_neg = []
        j = 0
        while len(local_neg) < self.local_negative:
            if j>100:
                print("group num:",self.local_cum_table_map[group])
                break
            # print("in local neg")
            w = self.local_cum_table[group].searchsorted(np.random.randint(self.local_cum_table[group][-1]))
            w = self.local_cum_table_map[group][w]
            if w != t_index and w not in context:
                local_neg.append(w)
                tj.append(0)
            j += 1
        # print("End local neg sampling")
        if len(local_neg)>0:
            self.weight_update(t_index,local_neg,tj,self.lr*self.neg_mtpl / self.size)
        
    def global_update(self,center,context):
        if center not in self.word_index.keys():
            return False
        t_index = self.word_index[center]
        for c in context: #排除重複瀏覽
            if c not in self.word_index.keys():
                return False
            context_index = self.word_index[c]

            # c_pos
            word_indices = [t_index]

            # c_neg 負採樣 
    #        while len(word_indices) < self.negative + 1:
            while len(word_indices) < 2:
                w = self.cum_table.searchsorted(np.random.randint(self.cum_table[-1]))
                if w != t_index and w != context_index:
                    word_indices.append(w)

            

            self.weight_update2(context_index, word_indices,self.lr * self.glb_mtpl / self.size)


            
    def on_epoch_begin(self, model):
        self.timerecord("on_epoch_begin start:")
        self.pre_train(model)   
        print("Epochs",str(self.epochs))
        for j,sentence in enumerate(self.corpus):  #每行資料
            self.progesss_rate(j) # 輸出進度
            sent_len = len(sentence) 
            global_list = []
            if sentence.count('-1')>0:
                g_index_start = sentence.index('-1')
                global_list = sentence[g_index_start+1:]
                browse_list = sentence[:g_index_start]
                sent_len = len(sentence[:g_index_start])

            

            # local負採樣
            if self.local_negative>0:
                for i, word in enumerate(sentence):
                    if word == '-1' or word not in self.word_index.keys():
                        continue

                    group = word[:2]
                    if len(self.local_cum_table[group]) >self.local_neg_min: # 不採樣，避免推離正相關的    
                        context = []
                        for j in range(i - self.window, i + self.window+1):
                            if j != i and j <= sent_len-1 and j >= 0:
                                if sentence[j] not in self.word_index.keys():
                                    continue
                                context.append(self.word_index[sentence[j]])

                        self.local_neg_update(word,context)
                 

            # Global context 主動提案
            if len(global_list)>0:
                browse_list = list(set(browse_list))
                for b in browse_list: 
                    self.global_update(b,global_list)
        print(self.model.running_training_loss)

                    
        self.timerecord("時間:")  
        self.epochs += 1
        

class ClassGenerateTrain(object):
    def __init__(self,dir_path,settings={}):
        self.settings = dict()  
        self.settings['USER_ID']='USER_ID'
        self.settings['ITEM_ID']='ITEM_ID'
        self.settings['ITEM']='ITEM'
        self.settings['TIMESTAMP']='TIMESTAMP'
        self.settings['STAY_TIME']='stay'
        self.settings['KIND']='KIND'
        self.settings['GROUP']='REGION_ID'
        self.settings['EVENT_TYPE']='EVENT_TYPE'
        self.settings['EVENT_VALUE']='EVENT_VALUE'
        self.settings['TYPE']='TYPE'
        self.settings['TYPE_ASSIGN']='R'
        self.settings['AIM']='dial'
        self.settings['SESSION_GAP']=600
        self.settings['MIN_STAY']=3500
        self.settings['DB']='591_xgb'
        self.settings['TABLE']='user_act'
        self.settings.update(settings) #有新的settings,採用新的
        self.dir_path =dir_path
        self.data_path=dir_path + '/data/'
        self.model_path=dir_path + '/model/'
        self.client = Client(os.getenv("DB_HOST"))
        self.conn = connect('clickhouse://' + os.getenv("DB_HOST"))
        self.cursor = self.conn.cursor()


    #讀取原始資料
    def loadCsvFile(self,data_path,fileName):
        csv = []
        if os.path.isfile(data_path+fileName):
            csv = pd.read_csv(data_path+fileName) 
            #csv = csv.head(2000)
        return csv
    #存取原始資料
    def saveCsvFile(self,csv,data_path,fileName):
        csv.to_csv(data_path+fileName)
    #clickhouse版產生txt
    def pool_sql(self,sql):
        users = pd.read_sql(sql,self.conn)
        checker = 0
        click = []
        click_active = []
        click_row = []
        act_row = []
        last_click_time = 0
        totalCount = 0
        for i,val in users.iterrows():

            if checker != val['USER_ID']:
                click_row,act_row,click,click_active = self.seperate_newline(click_row,act_row,click,click_active)
                click_row = []
                act_row = []
                last_click_time = 0
                checker = val['USER_ID']

            totalCount+=1

            t = int(val[self.settings['TIMESTAMP']])
            c_id = val[self.settings['ITEM_ID']]

            try:
                act_time = val[self.settings['AIM']]
            except:
                act_time = 0
            if last_click_time==0:
                last_click_time = t

            # 前後點擊時間 < 600秒 視為同一列資料
            if last_click_time-t<self.settings['SESSION_GAP']:
                click_row,act_row = self.connect_click(c_id,act_time,click_row,act_row)
                last_click_time = t
            else:
                click_row,act_row,click,click_active = self.seperate_newline(click_row,act_row,click,click_active)
                click_row,act_row = self.connect_click(c_id,act_time,click_row,act_row)
                last_click_time = t

        click_row,act_row,click,click_active = self.seperate_newline(click_row,act_row,click,click_active)

        
        output = open(self.data_path+self.fileName, 'a+', encoding='utf-8')
        for row in click+click_active*5: #目標行為放大5倍
            l = " ".join([str(i) for i in row])
            output.write(l)
            output.write('\n')
        output.close()
    #clickhouse版產生txt
    def pool_sql2(self,dfp):
        dft = dfp.copy()
        dft = dft.drop_duplicates(['USER_ID','ITEM_ID','TIMESTAMP'])

        #目標行為 'dial','message','question' 要重複
        dft['AIM'] = dft[['dial','message','question']].sum(axis=1)
        dft['repeat'] = dft['AIM'].apply(lambda x: np.where(x>0,'#',''))

        #user_id改變時換行
        dft['user_new_line'] = dft['USER_ID'].ne(dft['USER_ID'].shift().bfill()).astype(int)

        #時間間隔太長時換行
        dft['gap'] = dft.TIMESTAMP.diff().abs()
        dft['time_new_line'] = dft['gap'].apply(lambda x: np.where(x>self.settings['SESSION_GAP'],1,0))

        #插入換行符號
        dft['new_line'] = dft[['time_new_line','user_new_line']].sum(axis=1)
        dft['new_line_mark'] = dft['new_line'].apply(lambda x: np.where(x>0,'@',''))
        dft['combine'] = dft['new_line_mark']+dft['repeat']+dft[self.settings['ITEM_ID']]

        #把文字串在一起
        txt = dft['combine'].str.cat(sep=" ")
        lines = txt.split('@')

        output = open(self.data_path+self.fileNameTmp, 'a+', encoding='utf-8')
        for line in lines:
            lineList = []

            # 一列小於5個item時跳過
            lineToList = line.split()
            if len(lineToList)<4: 
                continue

            # 一列大於20個item時，每10個一切
            elif len(lineToList)>20:
                for i in range(0,len(lineToList),10):
                    smallerLine = " ".join(lineToList[i:(i+1)*10])
                    lineList.append(smallerLine)
            else:
                lineList = [line]

            for sliceLine in lineList:
                l = sliceLine+"\n"
                # 目標行為放大 5倍
                if '#' in l:
                    l = l.replace("#","")
                    l = l*5
                output.write(l)
        output.close()

    def generateTxtClickHouse(self,fileName,count_sql=False,pool_sql=False):
        self.fileName = fileName 
        self.fileNameTmp = 'tmp_'+fileName #寫入暫存檔案 ，避免失敗時舊資料被清掉
        output = open(self.data_path+self.fileNameTmp, 'w', encoding='utf-8')
        output.close()
        if count_sql:
            sql = count_sql
        else:
            sql = "SELECT count(*) \
                   FROM   %s.%s \
                   WHERE  EVENT_TYPE='stay' \
                   AND    EVENT_VALUE>%s\
                   AND    USER_ID !='0' \
                   LIMIT 1"
            sql = sql % (self.settings['DB'],self.settings['TABLE'],self.settings['MIN_STAY'])
        users   = pd.read_sql(sql,self.conn)
        count_data = users.values[0][0]
        Processor = int(mp.cpu_count())
        pagelimit = math.ceil(count_data/Processor) 
        
        p=mp.Pool(processes = Processor)
       
        res = []
        if pool_sql:
            sql = pool_sql
        else:
            sql =  "SELECT * \
                    FROM %s.%s as A  \
                    LEFT JOIN ( SELECT   USER_ID,\
                                         ITEM_ID,\
                                         max(DIAL) dial,\
                                         max(MESSAGE) as message,\
                                         max(QUESTION) as question \
                                FROM     591_xgb.auto_view\
                                GROUP BY USER_ID,ITEM_ID ) as B\
                    USING     ( USER_ID,ITEM_ID )\
                    WHERE     EVENT_TYPE='stay' \
                    AND       EVENT_VALUE>%s\
                    AND       USER_ID !='0'\
                    ORDER BY  USER_ID DESC,TIMESTAMP" % (self.settings['DB'],self.settings['TABLE'],self.settings['MIN_STAY'])

        df = pd.read_sql(sql,self.conn)

        for i in range(0,Processor):
            s = i*pagelimit
            e = i*pagelimit+pagelimit
            e = e if len(df) > e else len(df)

            res.append(p.apply_async(self.pool_sql2,(df.iloc[s:e],)))
        
        while res:
            for ret in res:
                if ret.ready():
                    res.remove(ret)
            time.sleep(0.005)

        p.close()#　關閉進程池,不再接受請求
        p.join() # 等待進程池中的事件執行完畢，回收進程池

        if os.stat(self.data_path+self.fileNameTmp).st_size > 0:
            os.replace(self.data_path+self.fileNameTmp, self.data_path+self.fileName)
        else:
            printt(self.data_path+self.fileNameTmp+" is empty, keep old file.")

        
    #產生txt
    def generateTxt(self,table_name,fileName):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = BASE_DIR +"/key/addcnair-9ae79e02bb80.json"
        clientBq = bq.Client()
        sql = "SELECT *  FROM "+table_name+" ORDER BY USER_ID,TIMESTAMP"
        users   = clientBq.query(sql)
        checker = 0
        click = []
        click_active = []
        click_row = []
        act_row = []
        last_click_time = 0
        totalCount = 0
        for val in users:
            
            if checker != val['USER_ID']:
                click_row,act_row,click,click_active = self.seperate_newline(click_row,act_row,click,click_active)
                click_row = []
                act_row = []
                last_click_time = 0
                checker = val['USER_ID']

            totalCount+=1
           
            t = int(val[self.settings['TIMESTAMP']])
            c_id = val[self.settings['ITEM_ID']]

            # act_time = int(val[self.settings['AIM']])
            act_time = 0 #先統一不計Book
            if last_click_time==0:
                last_click_time = t

            # 前後點擊時間 < 900秒 視為同一列資料
            if last_click_time-t<self.settings['SESSION_GAP']:
                click_row,act_row = self.connect_click(c_id,act_time,click_row,act_row)
                last_click_time = t
            else:
                click_row,act_row,click,click_active = self.seperate_newline(click_row,act_row,click,click_active)
                click_row,act_row = self.connect_click(c_id,act_time,click_row,act_row)
                last_click_time = t


        print("僅瀏覽: ",len(click))
        print("有應徵: ",len(click_active))
        print("共:",totalCount)
        output = open(self.data_path+fileName, 'w', encoding='utf-8')
        for row in click+click_active:
            l = " ".join([str(i) for i in row])
            output.write(l)
            output.write('\n')
    def loadTxtFile(self,data_path,fileName):
        text = []
        if os.path.isfile(data_path+fileName):
            fp = open(data_path+fileName, "r")
            for i,line in enumerate(iter(fp)):
                l = line.rstrip('\n')
                text.append(l.split(" "))
            fp.close()
        return text
   
    def train_model(self,model,fileName,gensim_config):
        text =self.loadTxtFile(self.data_path,fileName)
        if len(text) == 0:
            print("no search file")
            return
        """第一階段資料:去除-1"""
        text_orig = []
        for item in text:
            if '-1' in item:
                item.index('-1')
                e = item[:item.index('-1')]
                text_orig.append(e)
            else:
                text_orig.append(item)
        #選擇訓練方式
        if model == "basic":
            printt("use basic model")
            kmodel = word2vec.Word2Vec(text_orig,size=gensim_config['size'],sg=1, min_count=gensim_config['min_count'],iter=gensim_config['epochs'],seed=1,workers=4,alpha=0.025,window=gensim_config['window_size'])
        elif model == "deep": #加強訓練
            printt("use deep model")
            logger_config = gensim_config
            size = gensim_config['size']
            epochs=gensim_config['epochs']
            min_count=gensim_config['min_count']
            del gensim_config['size']
            del gensim_config['epochs']
            del gensim_config['min_count']
            
            epoch_logger = EpochLogger(text,gensim_config)
            kmodel = word2vec.Word2Vec(text_orig,size=size, min_count=min_count,sg=1,iter=epochs,callbacks=[epoch_logger],workers=4,seed=1,alpha=0.01,window=gensim_config['window_size'])  

        return kmodel
    def save_model(self,model,modelName):
        printt("save model")
        model.save(self.model_path+modelName) #all  
    def connect_click(self,c_id,act_time,click_row,act_row):
        if len(click_row)>0:
            if click_row[-1] != c_id: # 排除連續
                click_row.append(c_id)
        else:
            click_row.append(c_id)

        # 有無主動應徵 
        if act_time>0:
            act_row.append(c_id)

        return click_row,act_row
    def seperate_newline(self,click_row,act_row,click,click_active):
        if len(click_row)>4:
            # 主動應徵的case加入列的後面，並用-1隔開
            if len(act_row)>4:
                #click_row += ['-1']+list(set(act_row))
                click_active.append(click_row)

            else:
                click.append(click_row)    
        act_row = []
        click_row = []       
        return click_row,act_row,click,click_active


class ClassBuildWord2vecTrainData:
    def __init__(self):
        """
        df: user、item交互資料
        settings: 必填USER_ID,ITEM_ID,TIMESTAMP 選填SESSION_GAP
        使用方式: ==以下內容複製貼上==
            class W2V(ClassBuildWord2vecTrainData):
                def __init__(self,df,settings={}):
                    super().__init__()
                    self.settings.update(settings)
                    self.df = df

                def writeFile(self):
                    #自定義內容
                    pass

            newW2v = W2V(dft,settings)
            newW2v.go()
        """        
        self.settings = dict()  
        self.settings['USER_ID']='USER_ID'
        self.settings['ITEM_ID']='ITEM_ID'
        self.settings['TIMESTAMP']='TIMESTAMP'
        self.settings['GROUP']='REGION_ID'
        # self.settings['AIM']='dial'
        self.settings['SESSION_GAP']=600
        self.settings['DIR_PATH']='./'
        self.settings['FILE_NAME']='w2v_train.txt'
        self.settings['MODEL_NAME']='w2v_train.model'
        
        self.settings['w2v_size'] = 32
        self.settings['w2v_epochs'] = 5
        self.settings['w2v_windows'] = 10
        self.settings['w2v_min_count'] = 5
        self.settings['w2v_max_vocab_size'] = None
        self.settings['w2v_sample'] = 1e-3
        # self.settings.update(settings) #有新的settings,採用新的
        self.trainData = []

        
    def go(self):
        """ 產生流程 """

        self.echoStart()
        
        self.definePath()
        self.dropDuplicates()
        self.sortDF()
        # self.markAIMtoRepeat()
        self.markNewlineWhenUserChange()
        self.markNewlineWhenTimeIntervalTooLarge()
        self.insertNewlineMark()
        self.concatString()
        self.sepByNewlineMark()
        self.buildData()
        self.train()
        self.saveModel()
        
        self.echoDone()

    def echoStart(self):
        self.time = time.time()
        print("Go...  ",end=" ")

    def echoDone(self):
        cost_time = time.time() - self.time
        print("done. (Cost %.2fs)" % cost_time,end=" ")

    def definePath(self):
        self.dir_path = self.settings['DIR_PATH']
        self.data_path = self.dir_path + '/data/'
        self.model_path = self.dir_path + '/model/'
        self.data_file = self.data_path + self.settings['FILE_NAME']
        self.model_file = self.model_path + self.settings['MODEL_NAME']

    def dropDuplicates(self):
        self.df = self.df.drop_duplicates([self.settings['USER_ID'],self.settings['ITEM_ID'],self.settings['TIMESTAMP']])

    def sortDF(self):
        self.df = self.df.sort_values([self.settings['USER_ID'],self.settings['TIMESTAMP']])

    def markAIMtoRepeat(self):
        self.df['AIM'] = self.df[self.settings['AIM']].sum(axis=1)
        self.df['repeat'] = self.df['AIM'].apply(lambda x: np.where(x>0,'#',''))
    
    def markNewlineWhenUserChange(self):
        self.df['user_new_line'] = self.df[self.settings['USER_ID']].ne(self.df[self.settings['USER_ID']].shift().bfill()).astype(int)
    
    def markNewlineWhenTimeIntervalTooLarge(self):
        self.df['gap'] = self.df[self.settings['TIMESTAMP']].diff().abs()
        self.df['time_new_line'] = self.df['gap'].apply(lambda x: np.where(x>self.settings['SESSION_GAP'],1,0))
    
    def insertNewlineMark(self):
        self.df['new_line'] = self.df[['time_new_line','user_new_line']].sum(axis=1)
        self.df['new_line_mark'] = self.df['new_line'].apply(lambda x: np.where(x>0,'@',''))
        self.df['combine'] = self.df['new_line_mark'].astype(str)+self.df[self.settings['ITEM_ID']].astype(str)
    
    def concatString(self):
        self.txt = self.df['combine'].str.cat(sep=" ")
    
    def sepByNewlineMark(self):
        self.lines = self.txt.split('@')

    def createFile(self):
        # if not os.path.isfile(self.data_file):
        open(self.data_file, 'w+', encoding='utf-8')       
    
    def lineProcess(self,line):
        return line

    def filterShortLine(self,line):
        lineToList = line.split()
        if len(lineToList)<2: 
            return True

    def otherProcess(self,line):
        return line+"\n"
    
    def writeFile(self):
        self.createFile()
        
        return True
                
    def buildData(self):
        if self.writeFile():
            # print("BuildData and writeFile...")
            with open(self.data_file, 'a+', encoding='utf-8') as f:
                for line in self.lines:

                    line = self.lineProcess(line)
                    if self.filterShortLine(line):
                        continue

                    line = self.otherProcess(line)
                    self.trainData.append(line.split())
                    f.write(line)
        else:
            # print("BuildData...")
            for line in self.lines:

                line = self.lineProcess(line)
                if self.filterShortLine(line):
                    continue

                line = self.otherProcess(line)
                self.trainData.append(line.split())
                
    def train(self):
        # print("Training Word2vec...")
        self.model = word2vec.Word2Vec(self.trainData,
                                       size=self.settings['w2v_size'],
                                       iter=self.settings['w2v_epochs'],
                                       window=self.settings['w2v_windows'],
                                       min_count=self.settings['w2v_min_count'],
                                       max_vocab_size=self.settings['w2v_max_vocab_size'],
                                       sample=self.settings['w2v_sample'],
                                       sg=1,workers=-1,seed=1,alpha=0.01)
        return self.model
    
    def saveModel(self):
        print(f"Save model to {self.model_file}...  ",end=" ")
        self.model.save(self.model_file)
        
    def getTrainData(self):
        with open(self.data_file) as f:
            text = f.read().splitlines()

        return [i.split() for i in text]


class ClassCkip:
    def __init__(self):
        from ckiptagger import construct_dictionary, WS, POS, NER

        dic_path = "/home/htdocs/apiworker/public_data/ckip_data"

        self.ws  = WS(dic_path)
        self.pos = POS(dic_path)
        self.ner = NER(dic_path)
        self.settings = dict()  

        self.r='[’!"$&\'()*+,-./:;<=>?@[\\]^_`{|}~？！、，。「」［］（）｜：０１２３４５６７８９【】／]+'
        self.stop = ['Tasker','維修但幾勒','Tasker找師傅',
                     '夠','怎麼樣','怎麼辦','小','大','新','快速','怎麼說','告訴','想','哪些','殺人','好',
                     '1','2','3','4','5','6','7','8','9','10']
        self.stop_pos = ['Nh','Nep','Ncd','V_2','VL']
        self.word_to_weight = {
            "110v": 1,
            "220v": 1,
            "DIY":1,
            "過碳酸鈉":1,
            "矽酸鈣":1,
            "漂白劑":1,
            "補土":1,
            "批土":1,
            "防蟲":1,
            "接案":1,
            "省水":1,
            "斷捨離":1,
            "達人":1,
            "耗電":1,
        }

        self.dictionary = construct_dictionary(self.word_to_weight)

        
    def cut(self,sentence_list):

        sentence_list = [ re.sub(self.r,'',s) for s in sentence_list]
            
        word_sentence_list = self.ws(sentence_list, coerce_dictionary = self.dictionary)
        pos_sentence_list = self.pos(word_sentence_list)

        tmp = []
        for word_list, poss_list in zip(word_sentence_list, pos_sentence_list):
            for word,poss in zip(word_list,poss_list):
                if word in self.stop:
                    continue
                if poss[0]=='N' or poss[0]=='V' or poss=='FW':
                    if poss in self.stop_pos:
                        continue
                else:
                    continue

                tmp.append(word)
        return " ".join(tmp)


class ClassW2VConvert:
    """ word2vec word <=> id轉換 """
    def __init__(self,word2vec_model):
        self.model_vocab = word2vec_model.wv.vocab
        del word2vec_model

    def token(self,element,dtype="list"):
        if dtype == "list":
            return [self.model_vocab[str(e)].index+1 if str(e) in self.model_vocab else 0 for e in element]
        else:
            return self.model_vocab[str(element)].index+1 if str(element) in self.model_vocab else 0

    def token_decode(self,element,dtype="list"):
        reverse = {v.index+1:k for k,v in self.model_vocab.items()}
        if dtype == "list":
            return [reverse[e] for e in element if e in reverse]
        else:
            if element in reverse:
                return reverse[element]

"""
Youtube DNN Layers
"""
class L2NormLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L2NormLayer, self).__init__(**kwargs)
    
    @tf.function
    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = tf.ragged.boolean_mask(inputs, mask).to_tensor()
        return tf.math.l2_normalize(inputs, axis=-1)

    def compute_mask(self, inputs, mask):
        return mask

    
class MaskedEmbeddingsAggregatorLayer(tf.keras.layers.Layer):
    def __init__(self, agg_mode='mean', **kwargs):
        super(MaskedEmbeddingsAggregatorLayer, self).__init__(**kwargs)

        if agg_mode not in ['sum', 'mean']:
            raise NotImplementedError('mode {} not implemented!'.format(agg_mode))
        self.agg_mode = agg_mode
    
    @tf.function
    def call(self, inputs, mask=None):
        masked_embeddings = tf.ragged.boolean_mask(inputs, mask)
        if self.agg_mode == 'sum':
            aggregated =  tf.reduce_sum(masked_embeddings, axis=1)
        elif self.agg_mode == 'mean':
            aggregated = tf.reduce_mean(masked_embeddings, axis=1)
        return aggregated
    
    def get_config(self):
        # this is used when loading a saved model that uses a custom layer
        return {'agg_mode': self.agg_mode}


#宣告物件   


#obj=ClassGenerateTrain("/home/jupyer/derek/591")

#產生訓練txt
#sys.path.append('/home/jupyer/derek/591/')
#from data.config import city
#settings={}
#settings['city'] = city
#x = datetime.datetime.now()
#fileName='591Behavior_'+x.strftime("%Y%m%d")+'.txt'
#csvFileName='591Behavior_'+x.strftime("%Y%m%d")+'.csv'
#obj.generateTxt(pretrain_collection,fileName,otherInfo=settings)

##訓練模型
#gensim_config = {
#     'window_size': 5,    # context window +- center word
#     'size': 32,            # dimensions of word embeddings, also refer to size of hidden layer
#     'epochs': 20,       # number of training epochs
#     'learning_rate': 0.01,   # learning rate
#     'glb_mtpl': 0.05,   # 主動應徵learning rate 倍數(加權)
#     'neg_mtpl': 3,   # 負採樣learning rate 倍數(加權)
#     'local_negative': 5, #local負採樣數
#     'local_neg_min': 1000 #local負採樣 group數量門檻    
#}
#
#model = obj.train_model('basic',fileName,gensim_config)
#儲存模型
#obj.save_model(model,'demo.model')