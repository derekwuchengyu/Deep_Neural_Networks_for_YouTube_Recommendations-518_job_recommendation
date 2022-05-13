import sys,time,os
from flask import Flask,request
from dotenv import load_dotenv
load_dotenv()
sys.path.append('utils')

start_time = time.time()
print("Loading models...")
from p518.routes import indexApp
from p518.routes.indexApp import *
from p518.routes import keywordApp
from p518.routes.keywordApp import *
from p518.routes import catgoryApp
from p518.routes.catgoryApp import *
from p518.routes import resumeApp
from p518.routes.resumeApp import *
from p518.routes import checkfaceApp
from p518.routes.checkfaceApp import *  
from p518.routes import articleApp
from p518.routes.articleApp import *  
from p518.routes import chickptApp
from p518.routes.chickptApp import *  
from p518.routes import subjectApp
from p518.routes.subjectApp import *  
from p518.routes import invitepsApp
from p518.routes.invitepsApp import *  
from p518.routes import jobmatchApp
from p518.routes.jobmatchApp import *  
  
app = Flask(__name__)
app.config.from_object('p518.config')   # 載入配置檔案


print("start up spend (%.6f seconds)" % (time.time() - start_time))

#正式機設定
PATH=""
PORT=5000

try:
    if sys.argv[1]  == 'debug':        
        PATH= "/demo"
        PORT=5005
        app.debug = True

    if sys.argv[1]  == 'debug2':        
        PATH= "/demo2"
        PORT=5010  
        app.debug = True   
except:
    pass


RANK_LOG = '/home/htdocs/apiworker/logs/api_rank.log'    


@app.route(PATH + "/", methods=['POST', 'GET'])
def index():
    if not request.values.get('text'):
        return packJsonReturn(-1001, 'no seg text')
    
    result = indexApp.routeMain(request)
    
    return packJsonReturn(result['code'], result['sysmsg'],result['result'])
    

@app.route(PATH+"/keyword", methods=['POST', 'GET'])
def keyword():    
    if not request.values.get('text'):
        return packJsonReturn(-1001, 'no seg text')
        
        
    result = keywordApp.routeMain(request)
    return packJsonReturn(result['code'], result['sysmsg'],result['result'])
    
         
@app.route(PATH+"/catgory", methods=['POST', 'GET'])
def catgory():
    if not request.values.get('text'):
        return packJsonReturn(-1001, 'no seg text')

    result = catgoryApp.routeMain(request)
    
    return packJsonReturn(result['code'], result['sysmsg'],result['result'])


@app.route(PATH+"/catgory_resume", methods=['POST', 'GET'])
def catgory_resume():
    if not request.values.get('text'):
        return packJsonReturn(-1001, 'no seg text')

    result = catgoryApp.routeMainResume(request)
    
    return packJsonReturn(result['code'], result['sysmsg'],result['result'])

@app.route(PATH+"/resume", methods=['POST', 'GET'])
def resume():
    result = resumeApp.routeMain(request)
    return packJsonReturn(result['code'], result['sysmsg'],result['result'])


#頭貼辨識
@app.route(PATH+"/checkface", methods=['POST'])
def checkface():
    result = checkfaceApp.routeMain(request)
    return packJsonReturn(result['code'], result['sysmsg'],result['result'])

#相似文章搜尋
@app.route(PATH+"/article", methods=['POST'])
def simarticle():
    result = articleApp.routeMain(request)
    return packJsonReturn(result['code'], result['sysmsg'],result['result'])


#小雞推薦搜尋
@app.route(PATH+"/chickpt", methods=['POST'])
def chickpt():
    result = chickptApp.routeMain(request)
    return packJsonReturn(result['code'], result['sysmsg'],result['result'])


#科系類別
@app.route(PATH+"/department", methods=['POST', 'GET'])
def department():
    if not request.values.get('text'):
        return packJsonReturn(-1001, 'no seg text')
        
    result = subjectApp.routeMain(request)
    return packJsonReturn(result['code'], result['sysmsg'],result['result'])
 

#邀請人才推薦
@app.route(PATH+"/inviteps", methods=['POST', 'GET'])
def inviteps():
    if not request.values.get('m_id'):
        return packJsonReturn(-1001, 'no seg m_id')
    if not request.values.get('area'):
        return packJsonReturn(-1001, 'no seg area')
    if not request.values.get('job_one'):
        return packJsonReturn(-1001, 'no seg job_one')

    result = invitepsApp.routeMain(request)


    m_id = request.values.get('m_id')
    area = request.values.get('area')
    job_one = request.values.get('job_one')

    write_log("inviteps|"+m_id+"|"+area+"|"+job_one+"|"+str(result['result']),RANK_LOG)

    return packJsonReturn(result['code'], result['sysmsg'],result['result'])

#配對信推薦
@app.route(PATH+"/jobmatch", methods=['POST', 'GET'])
def jobmatch():
    if not request.values.get('m_id'):
        return packJsonReturn(-1001, 'no seg text')
    
    result = jobmatchApp.routeMain(request)
    
    return packJsonReturn(result['code'], result['sysmsg'],result['result'])


if __name__ == "__main__":
    #jieba.initialize()
    app.run(os.getenv("APP_IP"), PORT,threaded=True)