from contextlib import asynccontextmanager
from pathlib import Path
import io,json,sqlite3
from typing import Literal
import pandas as pd
from fastapi import FastAPI,HTTPException,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from joblib import load
from .pipeline import bootstrap,DB,MODEL,META,CLUSTERS,CSV,FEATURES,AREAS,ROADS,clean,build_db,train

@asynccontextmanager
async def lifespan(app):bootstrap();yield
app=FastAPI(title='TrafficIQ Analytics API',version='1.0',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['*'],allow_headers=['*'])

def query(sql,args=()):
    with sqlite3.connect(DB) as c:
        c.row_factory=sqlite3.Row
        return [dict(x) for x in c.execute(sql,args).fetchall()]
BASE=' FROM fact_traffic f JOIN dim_location l ON l.location_id=f.location_id JOIN dim_time t ON t.time_id=f.time_id '

@app.get('/api/health')
def health():return {'ok':True,'source':'synthetic demonstration dataset','warehouse':'SQLite star schema'}
@app.get('/api/meta')
def meta():return {'areas':AREAS,'roads':ROADS,'features':FEATURES,'source':'SYNTHETIC DEMO — NOT LIVE TRAFFIC'}
@app.get('/api/overview')
def overview(area:str='All'):
    filt=' WHERE l.area=?' if area!='All' else '';p=(area,) if filt else ()
    summary=query('SELECT COUNT(*) observations, ROUND(AVG(f.vehicle_count),1) avg_vehicles, ROUND(AVG(f.avg_speed_kmh),1) avg_speed, SUM(f.incident_flag) incidents,ROUND(100.0*SUM(CASE WHEN f.congestion_level="High" THEN 1 ELSE 0 END)/COUNT(*),1) high_pct'+BASE+filt,p)[0]
    levels=query('SELECT f.congestion_level name,COUNT(*) value'+BASE+filt+' GROUP BY f.congestion_level',p)
    hourly=query('SELECT t.hour hour,ROUND(AVG(f.vehicle_count),1) vehicles,ROUND(AVG(f.avg_speed_kmh),1) speed'+BASE+filt+' GROUP BY t.hour ORDER BY t.hour',p)
    areas=query('SELECT l.area area, COUNT(*) observations,ROUND(AVG(f.vehicle_count),1) vehicles,ROUND(AVG(f.avg_speed_kmh),1) speed,ROUND(100.0*AVG(CASE WHEN f.congestion_level="High" THEN 1 ELSE 0 END),1) risk'+BASE+' GROUP BY l.area ORDER BY risk DESC')
    monthly=query('SELECT t.month month,ROUND(AVG(f.vehicle_count),1) vehicles,COUNT(*) samples'+BASE+filt+' GROUP BY t.month ORDER BY t.month',p)
    return {'summary':summary,'levels':levels,'hourly':hourly,'areas':areas,'monthly':monthly,'data_source':'synthetic'}
@app.get('/api/olap')
def olap(group_by:Literal['area','road','month','hour','congestion_level']='area',area:str='All',level:str='All'):
    columns={'area':'l.area','road':'l.road','month':'t.month','hour':'t.hour','congestion_level':'f.congestion_level'}
    wh=[];p=[]
    if area!='All':wh.append('l.area=?');p.append(area)
    if level!='All':wh.append('f.congestion_level=?');p.append(level)
    where=' WHERE '+' AND '.join(wh) if wh else ''
    col=columns[group_by]
    return {'group_by':group_by,'rows':query(f'SELECT {col} dimension,COUNT(*) observations,ROUND(AVG(f.vehicle_count),1) avg_vehicles,ROUND(AVG(f.avg_speed_kmh),1) avg_speed, SUM(f.incident_flag) incidents'+BASE+where+f' GROUP BY {col} ORDER BY observations DESC',p),'operation':'roll-up / slice / dice (SQL GROUP BY + filters)'}
@app.get('/api/models')
def models():return {'metrics':json.loads(META.read_text()),'segmentation':json.loads(CLUSTERS.read_text())}
class Prediction(BaseModel):
    area:str='Andheri';road:str='Western Express Highway';hour:int=Field(18,ge=0,le=23);day_of_week:int=Field(0,ge=0,le=6);rainfall_mm:float=Field(4,ge=0,le=500);temperature_c:float=Field(30,ge=-20,le=60);vehicle_count:int=Field(180,ge=0,le=10000);avg_speed_kmh:float=Field(20,ge=0,le=180);incident_flag:int=Field(0,ge=0,le=1)
@app.post('/api/predict')
def predict(payload:Prediction):
    model=load(MODEL);frame=pd.DataFrame([payload.model_dump()]);pred=model.predict(frame)[0];prob=model.predict_proba(frame)[0]
    return {'prediction':pred,'confidence':round(float(max(prob))*100,1),'probabilities':{c:round(float(p)*100,1) for c,p in zip(model.classes_,prob)},'note':'Estimate from synthetic-data-trained model, not a real traffic alert'}
@app.get('/api/hotspots')
def hotspots():return {'areas':query('SELECT l.area,COUNT(*) samples,ROUND(100.0*AVG(CASE WHEN f.congestion_level="High" THEN 1 ELSE 0 END),1) high_congestion_pct,ROUND(AVG(f.avg_speed_kmh),1) speed'+BASE+' GROUP BY l.area ORDER BY high_congestion_pct DESC'),'note':'Statistical areas, not actual geocoded incidents'}
@app.get('/api/export')
def export_summary():return {'rows':query('SELECT l.area,t.month,COUNT(*) observations,ROUND(AVG(f.avg_speed_kmh),2) avg_speed'+BASE+' GROUP BY l.area,t.month ORDER BY t.month,l.area')}
@app.post('/api/retrain')
def retrain():
    df=clean(pd.read_csv(CSV));build_db(df);result=train(df);return {'ok':True,'metrics':result}
@app.post('/api/upload')
async def upload(file:UploadFile=File(...)):
    if not file.filename.lower().endswith('.csv'):raise HTTPException(400,'Upload a .csv file')
    raw=await file.read()
    if len(raw)>8_000_000:raise HTTPException(413,'Maximum CSV size is 8 MB')
    try:
        df=clean(pd.read_csv(io.BytesIO(raw)))
        if len(df)<120 or df.congestion_level.nunique()<2:raise ValueError('Need at least 120 valid records across 2 congestion classes')
        if min(df.congestion_level.value_counts())<2: raise ValueError('Need at least 2 examples per class')
        df.to_csv(CSV,index=False);build_db(df);m=train(df)
        return {'ok':True,'rows':len(df),'accuracy':m['accuracy']}
    except Exception as e: raise HTTPException(400,str(e))
