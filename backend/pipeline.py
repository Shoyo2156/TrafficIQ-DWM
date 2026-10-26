"""Reproducible synthetic demo traffic data, ETL/star-schema and real fitted estimators."""
from pathlib import Path
import sqlite3, json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
DB=DATA/'traffic_warehouse.db'; MODEL=DATA/'congestion_model.joblib'; META=DATA/'model_metrics.json'; CLUSTERS=DATA/'cluster_summary.json'; CSV=DATA/'traffic_demo.csv'
AREAS=['Andheri','Bandra','Dadar','Kurla','Powai','Sion','Thane','Worli']
ROADS=['Western Express Highway','Eastern Express Highway','LBS Marg','SV Road','Link Road','JVLR']
FEATURES=['area','road','hour','day_of_week','rainfall_mm','temperature_c','vehicle_count','avg_speed_kmh','incident_flag']
CATEGORICAL=['area','road']; NUMERIC=[x for x in FEATURES if x not in CATEGORICAL]

def generate_demo(n=4200, seed=42):
    rng=np.random.default_rng(seed)
    start=pd.Timestamp('2026-01-01')
    timestamps=start+pd.to_timedelta(rng.integers(0,180*24,n),unit='h')
    hours=np.asarray(timestamps.hour); day=np.asarray(timestamps.dayofweek)
    a=rng.choice(AREAS,n); r=rng.choice(ROADS,n)
    rain=np.maximum(0,rng.normal(2,6,n)); rain[rng.random(n)<.68]=0
    temp=np.clip(rng.normal(29,4,n),19,42)
    rush=(((hours>=8)&(hours<=11))|((hours>=17)&(hours<=21))).astype(int)
    weekend=(day>=5).astype(int)
    load=np.clip(85+70*rush-24*weekend+rain*2.2+np.isin(a,['Andheri','Kurla','Dadar'])*28+rng.normal(0,24,n),20,320)
    speed=np.clip(65-load*.20-rain*.33+rng.normal(0,7,n),6,85)
    incident=rng.binomial(1,np.clip(.045+rain*.007+(speed<25)*.085,0,.35))
    score=load*.16+(55-speed)*1.35+rain*.6+incident*17+rush*8
    level=np.where(score>=102,'High',np.where(score>=66,'Moderate','Low'))
    df=pd.DataFrame({'timestamp':timestamps,'area':a,'road':r,'hour':hours,'day_of_week':day,'rainfall_mm':np.round(rain,2), 'temperature_c':np.round(temp,1),'vehicle_count':np.round(load).astype(int),'avg_speed_kmh':np.round(speed,1),'incident_flag':incident,'congestion_level':level})
    return df.sort_values('timestamp').reset_index(drop=True)

def clean(df):
    df=df.copy(); required=['timestamp']+FEATURES+['congestion_level']; missing=set(required)-set(df.columns)
    if missing: raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    df=df[required].drop_duplicates().copy();df['timestamp']=pd.to_datetime(df['timestamp'],errors='coerce')
    for c in NUMERIC: df[c]=pd.to_numeric(df[c],errors='coerce')
    for c in CATEGORICAL+['congestion_level']: df[c]=df[c].astype('string').str.strip()
    df=df.dropna(subset=['timestamp','area','road','congestion_level'])
    for c in NUMERIC:
        df[c]=df[c].fillna(df[c].median() if df[c].notna().any() else 0)
    df=df[df.congestion_level.isin(['Low','Moderate','High'])]
    df['hour']=df.hour.clip(0,23).astype(int);df['day_of_week']=df.day_of_week.clip(0,6).astype(int)
    df['incident_flag']=df.incident_flag.clip(0,1).astype(int)
    df['vehicle_count']=df.vehicle_count.clip(lower=0).astype(int)
    df['avg_speed_kmh']=df.avg_speed_kmh.clip(0,180)
    df['rainfall_mm']=df.rainfall_mm.clip(0,500)
    return df.reset_index(drop=True)

def build_db(df):
    with sqlite3.connect(DB) as con:
        con.executescript('''DROP TABLE IF EXISTS fact_traffic; DROP TABLE IF EXISTS dim_location; DROP TABLE IF EXISTS dim_time;
        CREATE TABLE dim_location(location_id INTEGER PRIMARY KEY, area TEXT, road TEXT, UNIQUE(area,road));
        CREATE TABLE dim_time(time_id INTEGER PRIMARY KEY, timestamp TEXT UNIQUE, date TEXT, month TEXT, hour INTEGER, day_of_week INTEGER);
        CREATE TABLE fact_traffic(observation_id INTEGER PRIMARY KEY,location_id INTEGER,time_id INTEGER,vehicle_count INTEGER,avg_speed_kmh REAL,rainfall_mm REAL,temperature_c REAL,incident_flag INTEGER,congestion_level TEXT,FOREIGN KEY(location_id) REFERENCES dim_location(location_id),FOREIGN KEY(time_id) REFERENCES dim_time(time_id));
        CREATE INDEX idx_fact_location ON fact_traffic(location_id); CREATE INDEX idx_fact_time ON fact_traffic(time_id);''')
        loc=df[['area','road']].drop_duplicates().reset_index(drop=True);loc.insert(0,'location_id',range(1,len(loc)+1));loc.to_sql('dim_location',con,if_exists='append',index=False)
        times=df[['timestamp','hour','day_of_week']].drop_duplicates(subset='timestamp').reset_index(drop=True);times.insert(0,'time_id',range(1,len(times)+1));times['date']=times.timestamp.dt.strftime('%Y-%m-%d');times['month']=times.timestamp.dt.strftime('%Y-%m');times['timestamp']=times.timestamp.dt.strftime('%Y-%m-%d %H:%M:%S');times[['time_id','timestamp','date','month','hour','day_of_week']].to_sql('dim_time',con,if_exists='append',index=False)
        rows=df.copy();rows['timestamp']=rows.timestamp.dt.strftime('%Y-%m-%d %H:%M:%S');rows=rows.merge(loc,on=['area','road']).merge(times[['time_id','timestamp']],on='timestamp');rows[['location_id','time_id','vehicle_count','avg_speed_kmh','rainfall_mm','temperature_c','incident_flag','congestion_level']].to_sql('fact_traffic',con,if_exists='append',index=False)
    return len(rows)

def train(df):
    X=df[FEATURES];y=df['congestion_level'];xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=.22,random_state=42,stratify=y)
    prep=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),CATEGORICAL),('num',StandardScaler(),NUMERIC)])
    model=Pipeline([('preprocess',prep),('classifier',RandomForestClassifier(n_estimators=120,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=-1))])
    model.fit(xtrain,ytrain);pred=model.predict(xtest)
    classes=['Low','Moderate','High']; conf=confusion_matrix(ytest,pred,labels=classes).tolist()
    metric={'accuracy':round(float(accuracy_score(ytest,pred)),4),'f1_macro':round(float(f1_score(ytest,pred,average='macro')),4),'rows':len(df),'train_rows':len(xtrain),'test_rows':len(xtest),'classes':classes,'confusion_matrix':conf,'report':classification_report(ytest,pred,output_dict=True,zero_division=0),'data_source':'SYNTHETIC DEMONSTRATION DATA — NOT REAL-TIME TRAFFIC'}
    dump(model,MODEL);META.write_text(json.dumps(metric,indent=2))
    numeric=df[['hour','vehicle_count','avg_speed_kmh','rainfall_mm']]; scaler=StandardScaler();z=scaler.fit_transform(numeric)
    km=KMeans(n_clusters=4,random_state=42,n_init=10); labels=km.fit_predict(z)
    cs=df.assign(cluster=labels).groupby('cluster').agg(records=('hour','size'),vehicles=('vehicle_count','mean'),speed=('avg_speed_kmh','mean'),rainfall=('rainfall_mm','mean')).round(1).reset_index().to_dict('records')
    CLUSTERS.write_text(json.dumps({'clusters':cs,'silhouette_score':round(float(silhouette_score(z[::5],km.predict(z[::5]))),3)},indent=2))
    return metric

def bootstrap(force=False):
    if not force and DB.exists() and MODEL.exists() and META.exists() and CLUSTERS.exists():return
    if not CSV.exists() or force: generate_demo().to_csv(CSV,index=False)
    df=clean(pd.read_csv(CSV));df.to_csv(CSV,index=False);build_db(df);train(df)

if __name__=='__main__':
    bootstrap(force=True);print('Prepared',DB,MODEL)
