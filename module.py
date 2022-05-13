import sys,traceback
import json
import pymysql
import csv
import numpy as np
import pandas as pd
from math import log
sys.path.append('/utils')
from utils.config import *
from utils.jiebamodule import *
import memcache
import hashlib
import datetime


jiebaObj = MyJiebaDection()


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in self.dirname:
            for line in open(fname):
                yield line.split()

#md5編碼               
def md5(sql):
    hl = hashlib.md5()
    hl.update(sql.encode(encoding='utf-8'))
    return hl.hexdigest()
#sql撈資料或是生成csv
def getData(sql,filename=False,csv_file_path='/home/htdocs/apiworker/work/data/',db_opts = {
        'user': 'bi',
        'password': '!qaz2wsx',
        'host': '192.168.1.31',
        'database':'t_518_bi'
    },dftype=False,cachetime=600):
        
   
    rowsKey = md5(sql)
    colsKey = md5(sql+"cols")
    memc = memcache.Client(['192.168.1.42:11211']);
    rows = memc.get(rowsKey)
    description = memc.get(colsKey)
    if not rows:
        db = pymysql.connect(**db_opts)
        cur = db.cursor()
        rows = {}
        try:
            cur.execute(sql)
            rows = cur.fetchall()
            description=cur.description
            memc.set(rowsKey,rows,cachetime)
            memc.set(colsKey,cur.description,cachetime)
        except Exception as e:
            print(e)
            pass
        db.close()
    
   
    
    

   

    if rows:
        result = list()
        column_names = list()
        for i in description:
            column_names.append(i[0])

        result.append(column_names)
        for row in rows:
            result.append(row)
#             print(row)

        if filename:
            # Write result to file.
            csv_file_path = csv_file_path + filename
            with open(csv_file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in result:
                    csvwriter.writerow(row)
        elif dftype:
            return asdf(result)
        else:
            return result
    else:
        return True
#         print("No rows found for query: {}".format(sql))
#         sys.exit("No rows found for query: {}".format(sql))



def getStopwords(filepath):
    stopword_set = set()
    with open(filepath,'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))
    return stopword_set

def getFileContent(filepath):
    dicts = dict()
    with open(filepath,'r', encoding='utf-8') as text:
        r = list(text)
        for i in r:
            i = i.rstrip()
            i = i.lower()
            i = i.split(' ')
            dicts.update( {i[0]:i[1]} )   
    return dicts


def loadFileContent(filepath):
    dicts = dict()
    with open(filepath, 'r') as f:
        dicts = json.load(f)
    return dicts


#asDataframe
def asdf(lists,header=True):
    df = pd.DataFrame(lists)
    if header:
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
    df = df.fillna(0)
    df = df.replace('',0)
    return df

# 錯誤輸出
def echoError():
    ex_type, ex_val, ex_stack = sys.exc_info()
    lastCallStack = traceback.extract_tb(ex_stack)[-1] #取得Call Stack的最後一筆資料
    fileName = lastCallStack[0] #取得發生的檔案名稱
    lineNum = lastCallStack[1] #取得發生的行數
    funcName = lastCallStack[2] #取得發生的函數名稱
    errMsg = "[{}] {}. In file \"{}\", line {}, in {}: ".format(ex_type.__name__, ex_val, fileName, lineNum, funcName)
    errMsg = traceback.format_exc() #取得完整錯誤
    
    write_error(errMsg)

    return (errMsg)
# 錯誤紀錄
def write_error(errMsg):
    DATE = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    output = open('/home/htdocs/apiworker/logs/api_error.log', 'a', encoding='utf-8')
    output.write(DATE+errMsg+"\n")
    
def packJsonReturn(retCode=0, retMsg='ok', retData=None):
    return json.dumps({'retCode':retCode, 'retMsg':retMsg, 'retData':retData},
                      ensure_ascii=False).encode()    

def printt(msg,end='\n'):
    DATE = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    print(DATE+msg,end=end)

def write_log(Msg,file):
    DATE = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    output = open(file, 'a', encoding='utf-8')
    output.write(DATE+str(Msg)+"\n")

def read_sql(sql,db_opts = {
        'user': 'bi',
        'password': '!qaz2wsx',
        'host': '192.168.1.31',
        'database':'t_518_bi'
    }):

    conn = pymysql.connect(**db_opts)
    df = pd.read_sql(sql,conn)
    conn.close()

    return df