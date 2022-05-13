import sys,os
# import time
import numpy as np 
import pandas as pd
sys.path.append('/home/htdocs/apiworker/')
from utils.module import *
from utils.airmodule import *
from p518.config import *

import datetime
from datetime import date, timedelta 
from clickhouse_driver import connect,Client




DAYS = 90
TIME_RESTICT = int(time.time()) - 86400*DAYS #90天


class W2V(ClassBuildWord2vecTrainData):
    def __init__(self,df,settings={}):
        super().__init__()
        self.settings.update(settings)
        self.df = df
        print("  ")
    def writeFile(self):
        pass
    def filterShortLine(self,w):
        pass

def update_word2vec_models(applys_data):
  print("Updating word2vec models...")
  # js_id word2vec
  settings = {}
  settings['USER_ID'] = 'm_id'
  settings['ITEM_ID'] = 'js_id'
  settings['TIMESTAMP'] = 'send_time'
  settings['DIR_PATH'] = BASE_DIR
  settings['MODEL_NAME'] = '10_w2v_js_id.model'
  settings['w2v_min_count'] = 0
  jsW2v = W2V(applys_data[['m_id','js_id','send_time']],settings)
  jsW2v.go()
  print('\n  "{}" 數量: {}個。'.format(settings['ITEM_ID'],len(jsW2v.model.wv.index2entity)))

  # job_place word2vec
  settings = {}
  settings['USER_ID'] = 'm_id'
  settings['ITEM_ID'] = 'job_place'
  settings['TIMESTAMP'] = 'send_time'
  settings['DIR_PATH'] = BASE_DIR
  settings['MODEL_NAME'] = '10_w2v_job_place.model'
  settings['w2v_min_count'] = 0
  jlW2v = W2V(applys_data[['m_id','job_place','send_time']],settings)
  jlW2v.go()
  print('\n"{}" 數量: {}個。'.format(settings['ITEM_ID'],len(jlW2v.model.wv.index2entity)))

  # job_class word2vec
  settings = {}
  settings['USER_ID'] = 'm_id'
  settings['ITEM_ID'] = 'job_class'
  settings['TIMESTAMP'] = 'send_time'
  settings['DIR_PATH'] = BASE_DIR
  settings['MODEL_NAME'] = '10_w2v_job_class.model'
  settings['w2v_min_count'] = 0
  jcW2v = W2V(applys_data[['m_id','job_class','send_time']],settings)
  jcW2v.go()
  print('\n"{}" 數量: {}個。'.format(settings['ITEM_ID'],len(jcW2v.model.wv.index2entity)))




class UserImageClass():
  def __init__(self):
    self.settings = dict()  
    self.conn = connect('clickhouse://192.168.1.42')
    self.client = Client('192.168.1.42')
    self.prefix = 'recomm.518paper_' #518db or clickhouse db
    self.user_job_list = False
    self.TABLE_USER_IMAGE = pd.read_sql("SELECT * FROM system.columns where database='recomm' AND table='518_user_image'",self.conn)

  def image_default_value(self):
      for i,row in self.TABLE_USER_IMAGE.iterrows():
          # 未使用的欄位給預設值
          if row['name'] not in self.user_job_list.columns:
              if row['type']=='Array(String)':
                  default = [['0']]*len(self.user_job_list)
              elif row['type']=='Float32' or row['type']=='Int32':
                  default = 0
              else:
                  default = ''
              self.user_job_list[row['name']] = default

  def embedding_list_process(self,s):
      try:
          if len(s)==0 or s==['']:
              return ['0']
          else:
              return s
      except:
          return s


  def dataPrecess(self,cv_data,apply_data=[],apply_hist=[],apply_class=[],apply_locat=[]):


    # # 求職者背景
    cv_data = cv_data[['m_id','gender','location_chooseStr','job_chooseStr','cum_work_exp','cum_job_one','cum_job_two','cum_job_thr']].copy()
    cv_data = cv_data[~cv_data['job_chooseStr'].fillna('0').str.contains('[\u4e00-\u9fa5]', regex = True, na=False)] # 排除異常資料
    cv_data['exp_year'] = cv_data['cum_work_exp']
    cv_data['exp_job'] = cv_data[['cum_job_one','cum_job_two','cum_job_thr']].replace('','0').fillna('0').values.tolist()
    # print(cv_data)


    if len(apply_data)>0:
      user_job_list = apply_data
      if len(cv_data)>0:
        user_job_list = pd.merge(user_job_list,cv_data, how= 'left',on='m_id')      
      if len(apply_hist)>0:
        user_job_list = pd.merge(user_job_list,apply_hist, how= 'left',on='m_id')      
      if len(apply_class)>0:
        user_job_list = pd.merge(user_job_list,apply_class, how= 'left',on='m_id')
      if len(apply_locat)>0:
        user_job_list = pd.merge(user_job_list,apply_locat, how= 'left',on='m_id')
    else:
      user_job_list = cv_data

    user_job_list['gender']   = user_job_list['gender'].fillna(0).replace('',0).astype(int)
    user_job_list['exp_year'] = user_job_list['exp_year'].fillna(0).astype(int)
    user_job_list['expect_locat'] = user_job_list['location_chooseStr'].replace('','0').fillna('0')
    user_job_list['expect_locat'] = user_job_list['expect_locat'].apply(lambda x: str(x).split(','))
    user_job_list['expect_locat'] = user_job_list['expect_locat'].apply(self.embedding_list_process)
    user_job_list['expect_class'] = user_job_list['job_chooseStr'].replace('','0').fillna('0')
    user_job_list['expect_class'] = user_job_list['expect_class'].apply(lambda x: str(x).split(','))
    user_job_list['expect_class'] = user_job_list['expect_class'].apply(self.embedding_list_process)
    user_job_list['exp_job'] = user_job_list['exp_job'].fillna('0').map(list)
    user_job_list['exp_job'] = user_job_list['exp_job'].apply(self.embedding_list_process)

    self.user_job_list = user_job_list
    # print("user_job_list : ")
    # print(user_job_list.head(2))


  def insertData(self):
    self.image_default_value()
    self.user_job_list['i_time'] = int(date.today().strftime("%s")) #今日date  
    self.user_job_list = self.user_job_list[self.TABLE_USER_IMAGE.name] #排順序

    self.client.execute("INSERT INTO recomm.518_user_image VALUES", self.user_job_list.to_dict('records'),types_check=True)
    print('  Inserted {} rows of data'.format(len(self.user_job_list)))

  def dropDuplicate(self):
    print('Droping duplicate...')
    self.client.execute('optimize table recomm.518_user_image DEDUPLICATE')

  def build_cold_user_image(self):
    """
    新開通知，但沒有互動的User
    """

    print('Building cold user image...')
    # 履歷背景
    # 履歷期望值缺
    cv_new = pd.read_sql(f'SELECT A.match_status,B.*,C.*,D.*,E.* FROM (\
                             SELECT PM.* FROM {self.prefix}profile_match AS PM \
                             LEFT JOIN recomm.518_user_image AS UI ON PM.m_id=UI.m_id\
                             WHERE PM.match_status IN (\'1\', \'3\', \'4\', \'5\') AND UI.m_id is NULL\
                           ) as A\
                           INNER JOIN {self.prefix}profile_condition   AS C ON A.m_id=C.m_id\
                           LEFT  JOIN {self.prefix}profile_basic       AS B ON C.ps_id=B.ps_id\
                           LEFT  JOIN {self.prefix}profile_description AS D ON C.ps_id=D.ps_id\
                           LEFT  JOIN {self.prefix}profile_experience  AS E ON C.ps_id=E.ps_id\
                           ORDER BY A.m_id,C.work_start_time DESC',self.conn)
    if len(self.prefix)>0:
        cv_new.columns = [x.split('.')[1] for x in cv_new.columns]
    cv_new = cv_new.loc[:,~cv_new.columns.duplicated()]
    
    self.dataPrecess(cv_new)
    self.insertData()



  def build_active_user_image(self):
    """
    過去90天有互動的user
    """

    print('Building active user image...')
    # 互動紀錄 & 工作背景
    applys = pd.read_sql(f'SELECT A.*,B.job_location,B.job_one,B.job_two FROM {self.prefix}resume_submit_active AS A\
                           LEFT JOIN {self.prefix}job_basic AS B ON A.js_id=B.js_id \
                           WHERE send_time > {TIME_RESTICT} \
                           ORDER BY send_time DESC',self.conn)

    applys['job_place'] = applys['job_location'].str.slice(stop=7) + ' ' + applys['job_location'] 
    applys['job_class'] = applys['job_one'].astype(str).str.slice(stop=7) + ' ' + applys['job_one'].astype(str)

    # 配合 job_chooseStr 全部轉成string
    applys['js_id'] = applys['js_id'].apply(str)
    applys['job_location'] = applys['job_location'].apply(str)
    applys['job_one'] = applys['job_one'].apply(str)
    # print(applys.head(1))

    # 履歷背景
    # 履歷期望值缺
    cv = pd.read_sql(f'SELECT * FROM {self.prefix}profile_basic AS B\
                       LEFT JOIN {self.prefix}profile_condition   AS C ON B.ps_id=C.ps_id\
                       LEFT JOIN {self.prefix}profile_description AS D ON B.ps_id=D.ps_id\
                       LEFT JOIN {self.prefix}profile_experience  AS E ON B.ps_id=E.ps_id\
                       WHERE B.ps_id IN (\
                                         SELECT DISTINCT ps_id FROM {self.prefix}resume_submit_active\
                                         WHERE e_time > {TIME_RESTICT}\
                                        )',self.conn)
    if len(self.prefix)>0:
        cv.columns = [x.split('.')[1] for x in cv.columns]
    cv = cv.loc[:,~cv.columns.duplicated()]
    # print(cv.head(1))

    sample_data = applys.drop_duplicates(subset = "m_id").copy()
    sample_data = sample_data.drop(columns='js_id')
    apply_hist = applys.groupby(['m_id'])['js_id'].apply(list)
    apply_hist.name = 'apply_hist'
    apply_locat = applys.groupby(['m_id'])['job_location'].apply(list)
    apply_locat.name = 'apply_locat'
    apply_class = applys.groupby(['m_id'])['job_one'].apply(list)
    apply_class.name = 'apply_class'
    

    self.dataPrecess(cv,sample_data,apply_hist,apply_locat,apply_class)
    self.insertData()


    self.applys_data = applys # word2vec可以接去用



if __name__ == "__main__":
    print("開始訓練")
    start_time = time.time()

    ImageClass = UserImageClass()
    ImageClass.build_cold_user_image()
    ImageClass.build_active_user_image()
    ImageClass.dropDuplicate()

    update_word2vec_models(ImageClass.applys_data)

    print("Build user image and update w2v spent (%.2f seconds)." % (time.time() - start_time))