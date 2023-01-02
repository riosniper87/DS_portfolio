#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
import mysql.connector
#import dotenv
import os
import re
from datetime import timedelta
from datetime import datetime
import shutil
import subprocess
import enum
from sqlalchemy import create_engine
import pymysql
from datetime import date
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import joblib


# In[3]:


#쿼리 불러오는 function
def model_import_table_fn(query):
    con = mysql.connector.connect(user=db_user, password=db_passwd, host =db_host, database  = db_db, port = int(db_port))
    df = pd.read_sql(query, con)
    con.cursor().close()
    con.close()
    del con
    return df


# In[12]:


#dotenv.load_dotenv('C:/Users/teamsnowball/models/mid_kill_5/d2.env', verbose=True, override=True)
db_host = 'teamsnowball.co'
db_user = 'tsb'
db_passwd = 'tmshdnqhf0731'
db_port = 3500
db_db = 'tipster'


# In[5]:


def md_connect(user, password, db, host, port=db_port):
    url = 'mysql+mysqldb://{}:{}@{}:{}/{}'.format(user, password, host, port, db)
    engine = create_engine(url)
    return engine


# In[6]:


#위의 md_connect를 쓰는데 MySQLdb 모듈을 못찾는 에러가 나는 경우 실행하는 명령어 
#pymysql을 MySQLdb로 인스톨 하는 기능으로 알고 있음.
pymysql.install_as_MySQLdb()


# In[10]:


md_engine = md_connect(user = db_user, password = db_passwd, db = db_db, host = db_host, port = 3500)


# In[14]:


raw_data = model_import_table_fn("SELECT * FROM h2h_stats")
# raw_data = raw_data[(raw_data.camp == 'blue')].reset_index()
schedule = model_import_table_fn("SELECT * FROM league_schedule where NOT team1 = 'TBD'")
db_db = 'teamsnowball'
ai_prob = model_import_table_fn("SELECT * FROM league_schedule")


# In[ ]:


#이걸 위에 데이터 프레임에 맞도록 설계 
#홀짝으로 데이터 거르고 돌리면 될듯. range kill death, assist 정도만 활용 
#이걸로 타임라인 만들고, 적당한 경기수가 있는 지점부터 월즈 전까지로 학습. 
# condition == "wc" 시 world22 데이터로만 featrue 생성 
# days num 은 경기 시작일 기준 몇일 전까지 데이터를 feature로 사용하는가
# 넉넉하게 값을 줄 경우 통계량이 잘 나오지만, 최근 팀의 결과가 적게 반영되는 이슈가 있고, 
# 타이트하게 값을 줄 경우 데이터가 0 이라 데이터가 안나오는 경우가 생김 
def timeline_team_fn(data, condition = False, daysnum = 15) :     
    
    stat_df = pd.DataFrame()
    
    #행별 전처리 진행
    for ij in range(0,len(data)):
#         print(ij)
        dat = data.loc[[ij, ]] 
        team = dat.team.tolist()[0]
        opp_team = dat.oppTeam.tolist()[0]
        league = dat.league.tolist()[0]
        max_date = dat.date.tolist()[0]
        min_date = max_date - pd.DateOffset(days = days_num)
        if condition == "wc" : 
            filter_dat = raw_data[(raw_data.league == "wc") & (raw_data.date >= datetime(2022,10,8,0,0,0))]
        else : 
            std_dat = raw_data
            filter_dat = std_dat[(std_dat.date <= max_date) & (std_dat.date >= min_date)]
            
        
        filter_dat_blue = filter_dat[(filter_dat.team == team)]
        filter_dat_red  = filter_dat[(filter_dat.oppTeam == team)]
        opp_filter_dat_blue = filter_dat[(filter_dat.team == opp_team)]
        opp_filter_dat_red  = filter_dat[(filter_dat.oppTeam == opp_team)]
        length = len(filter_dat_blue) + len(filter_dat_red)
        opp_length = len(opp_filter_dat_blue) + len(opp_filter_dat_red)
        tot_data = {'team' : [team], 
                    'oppTeam' : [opp_team],
                    'date' : [max_date],
                    'league' : [league],
                    'len' : [length],
                    'avg_gamelength' : [np.mean([filter_dat_blue.gameLength.tolist() + filter_dat_red.gameLength.tolist()])],
                    'avg_kills' : [np.mean([filter_dat_blue.kills.tolist() + filter_dat_red.oppKills.tolist()])],
                    'avg_deaths' : [np.mean([filter_dat_blue.deaths.tolist() + filter_dat_red.oppDeaths.tolist()])],
                    'avg_assists' : [np.mean([filter_dat_blue.assists.tolist() + filter_dat_red.oppAssists.tolist()])],
                    'max_kills' : [np.amax([filter_dat_blue.kills.tolist() + filter_dat_red.oppKills.tolist()])],
                    'max_deaths' : [np.amax([filter_dat_blue.deaths.tolist() + filter_dat_red.oppDeaths.tolist()])],
                    'max_assists' : [np.amax([filter_dat_blue.assists.tolist() + filter_dat_red.oppAssists.tolist()])],
                    'min_kills' : [np.amin([filter_dat_blue.kills.tolist() + filter_dat_red.oppKills.tolist()])],
                    'min_deaths' : [np.amin([filter_dat_blue.deaths.tolist() + filter_dat_red.oppDeaths.tolist()])],
                    'min_assists' : [np.amin([filter_dat_blue.assists.tolist() + filter_dat_red.oppAssists.tolist()])],
                    'opp_len' : [opp_length],
                    'opp_avg_gamelength' : [np.mean([opp_filter_dat_blue.gameLength.tolist() + opp_filter_dat_red.gameLength.tolist()])],
                    'opp_avg_kills' : [np.mean([opp_filter_dat_blue.oppKills.tolist() + opp_filter_dat_red.kills.tolist()])],
                    'opp_avg_deaths': [np.mean([opp_filter_dat_blue.oppDeaths.tolist() + opp_filter_dat_red.deaths.tolist()])],
                    'opp_avg_assists': [np.mean([opp_filter_dat_blue.oppAssists.tolist() + opp_filter_dat_red.assists.tolist()])],
                    'opp_max_kills' : [np.amax([opp_filter_dat_blue.oppKills.tolist() + opp_filter_dat_red.kills.tolist()])],
                    'opp_max_deaths' : [np.amax([opp_filter_dat_blue.oppDeaths.tolist() + opp_filter_dat_red.deaths.tolist()])],
                    'opp_max_assists' : [np.amax([opp_filter_dat_blue.oppAssists.tolist() + opp_filter_dat_red.assists.tolist()])],
                    'opp_min_kills' : [np.amin([opp_filter_dat_blue.oppKills.tolist() + opp_filter_dat_red.kills.tolist()])],
                    'opp_min_deaths' : [np.amin([opp_filter_dat_blue.oppDeaths.tolist() + opp_filter_dat_red.deaths.tolist()])],
                    'opp_min_assists' : [np.amin([opp_filter_dat_blue.oppAssists.tolist() + opp_filter_dat_red.assists.tolist()])]}
                   



        tot_dat = pd.DataFrame(tot_data)
        stat_df = pd.concat([stat_df, tot_dat], join= 'outer',ignore_index = True)

    return stat_df


# In[1]:


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


# In[ ]:


#raw data 기반, featrue data (stat_df) 생성
raw_data = raw_data[(raw_data.camp == 'blue')].reset_index()
stat_df = timeline_team_fn(raw_data)
stat_df = timeline_team_fn(raw_data)
stat_df['kills'] = raw_data.kills
stat_df['oppKills'] = raw_data.oppKills
stat_df['result'] = raw_data.result
#stat_df = stat_df[(stat_df.league != 'vcs')].reset_index()
#stat_df = stat_df[(stat_df.league != 'lpl')].reset_index()
stat_df['team1Prob'] = 0


# In[ ]:


# ai prob 를 stat_df 에 붙이는 과정 
for ijk in range(0,len(stat_df)) : 
    #print(ijk)
    stat_add = stat_df.iloc[ijk,]
    team = stat_add.team
    oppTeam = stat_add.oppTeam
    date = stat_add.date
    ai_candi = ai_prob[(ai_prob.team1 == team)&(ai_prob.team2== oppTeam)]
    if len(ai_candi) >= 1 : 
        ai_date = nearest(ai_candi.date,date)
        ai_sel = ai_candi[(ai_prob.date == ai_date)]
        prob =ai_sel.team1Prob.tolist()[0]
        stat_df.at[ijk,'team1Prob']=prob
        del prob


# In[ ]:


#ai prob가 없는 매치들은 쓸수 없으므로 제외 
stat_df = stat_df[(stat_df.team1Prob != 0)]
stat_df = stat_df.dropna()
stat_df = stat_df.reset_index()


# In[ ]:


stat_df['team2Prob'] = 1-stat_df.team1Prob


# In[ ]:


#feature list 생성, 나중에 feature selection 같은걸 하려고 하였음. 
features = ['team1Prob','team2Prob','avg_gamelength','opp_avg_gamelength','avg_kills','avg_deaths','avg_assists','max_kills',
            'max_deaths','max_assists','min_kills','min_deaths','min_assists','opp_avg_kills','opp_avg_deaths',
            'opp_avg_assists','opp_max_kills','opp_max_deaths','opp_max_assists','opp_min_kills','opp_min_deaths',
            'opp_min_assists']
#킬 토탈 핸디 범위 설정 (전체)
total_handi = np.arange(15.5,35.5,1).tolist()

#킬 토탈 핸디 모델링할 핸디 설정 (핸디 가장 큰 값 / 작은 값/ 중간을 재고 나머지 핸디 값들은 저 세 값들을 사용한 
#등차 수열로 만듬)
std_total_handi = [total_handi[0],total_handi[round(len(total_handi)/2)],total_handi[len(total_handi)-1]]
std_total_loc_0 = total_handi.index(std_total_handi[0])
std_total_loc_1 = total_handi.index(std_total_handi[1])
std_total_loc_2 = total_handi.index(std_total_handi[2])

#등차 수열 레인지 
total_range_0 = std_total_loc_1-std_total_loc_0
total_range_1 = std_total_loc_2-std_total_loc_1

#모델 리스트 설정 
total_model_list = []
for x in total_handi : 
    total_model_list.append('Lol-Pre-Kill-Total-Handicapped-' + str(x) + '-Team')

#킬 차이 모델 범위 설정 (이후는 킬 토탈과 비슷하므로 생략)
diff_handi = np.arange(-22.5,23.5,1).tolist()
#킬 토탈과 다른 점은 no model cnt 값인데, 킬 차이 모델은 핸디캡 범위가 기존 생각보다 넓어서 
#양 끝값 보다는 앞에서 no model cnt 번째 값, 뒤에서도 같은 no model cnt 번째 값만 가지고 모델링 하는 과정
handi_nomodel_cnt = 7
std_diff_handi = [diff_handi[handi_nomodel_cnt],diff_handi[round(len(diff_handi)/2)],
                  diff_handi[len(diff_handi)-(handi_nomodel_cnt+1)]]

std_diff_loc_0 = diff_handi.index(std_diff_handi[0])
std_diff_loc_1 = diff_handi.index(std_diff_handi[1])
std_diff_loc_2 = diff_handi.index(std_diff_handi[2])

diff_range_0 = std_diff_loc_1-std_diff_loc_0
diff_range_1 = std_diff_loc_2-std_diff_loc_1

diff_model_list = []
for x in diff_handi : 
    diff_model_list.append('Lol-Pre-MoreKillsHandicapped-' + str(x) + '-Team')


# In[ ]:


#킬 토탈 모델링, std_total_handi 안에 있는 값들만 local에 저장하는 형식으로 진행 
#10월 8일 이전의 게임들로 학습하여, 그 이후의 게임들을 맞추는 형식으로 성능을 확인
#성능 확인 값 (logloss, acc, auc)들을 각 모델별로 보여준다. 
#randomforest를 사용하도록 되어있으며 , 주석 처리된 부분은 logregression 사용이 필요할 경우 활용할 수 있다. 
total_summary_df = pd.DataFrame()
for i in range(0,len(std_total_handi)) : 
    handi = std_total_handi[i]
    model = 'Lol-Pre-totalKills-' + str(handi) + '-UnderOver'
    stat_df['y'] = np.where(stat_df.kills + stat_df.oppKills <= handi,0,1)

    #handi = diff_handi[0]
    #model = 'Lol-Pre-Kill-Diff-Handicapped-' + str(handi) + '-Team'
    #stat_df['y'] = np.where(stat_df.kills + handi <= stat_df.opp_kills ,0,1)

    train_df = stat_df[(stat_df.date < datetime(2022,10,8,0,0,0))]
    X_train  = train_df[features]
    y_train  = train_df['y']

    test_df  = stat_df[(stat_df.date >= datetime(2022,10,8,0,0,0))]
    X_test   = test_df[features]
    y_test   = test_df['y']

    rf_reg=RandomForestClassifier(n_estimators=500, max_features = 3)
    rf_reg.fit(X_train,y_train)

    y_pred=rf_reg.predict_proba(X_test)

#     clf = LogisticRegression(random_state=0).fit(X_train, y_train)
#     y_pred = clf.predict_proba(X_test)


#     summary = {'model' : [model], 
#                'log_loss' : [metrics.log_loss(y_test, y_pred)], 
#                'acc' : [metrics.accuracy_score(y_test, clf.predict(X_test))], 
#                'auc' : [metrics.auc(fpr, tpr)]}
#     summary_df = pd.DataFrame(summary)

#     fit_df = stat_df
#     X_save = fit_df[features]
#     y_save = fit_df['y']
#     clf.fit(X_save, y_save)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred[:,1], pos_label=1)
    
    summary = {'model' : [model], 
               'log_loss' : [metrics.log_loss(y_test, y_pred)], 
               'acc' : [metrics.accuracy_score(y_test, rf_reg.predict(X_test))], 
               'auc' : [metrics.auc(fpr, tpr)]}
    summary_df = pd.DataFrame(summary)
    
    fit_df = stat_df
    X_save = fit_df[features]
    y_save = fit_df['y']
    rf_reg.fit(X_save, y_save)

    nm = "/Users/davidkim/Documents/kill_total_diff_models/" + model + '.joblib'
    total_summary_df = pd.concat([summary_df, total_summary_df], join= 'outer',ignore_index = True)

    joblib.dump(rf_reg,nm)
#     joblib.dump(clf,nm)


# In[ ]:


#킬 차이 모델, 킬 토탈과 같은 컨셉으로 이루어져 있다.
diff_summary_df = pd.DataFrame()
for i in range(0,len(std_diff_handi)) : 

    handi = std_diff_handi[i]
    model = 'Lol-Pre-MoreKillsHandicapped-' + str(handi) + '-Team'
    stat_df['y'] = np.where(stat_df.kills + handi <= stat_df.oppKills ,0,1)

    #train_df = stat_df[(stat_df.league != 'wc')]
    train_df = stat_df[(stat_df.date < datetime(2022,10,8,0,0,0))]
    X_train  = train_df[features]
    y_train  = train_df['y']

    test_df  = stat_df[(stat_df.date >= datetime(2022,10,8,0,0,0))]
    X_test   = test_df[features]
    y_test   = test_df['y']

#     clf = LogisticRegression(random_state=0).fit(X_train, y_train)
#     y_pred = clf.predict_proba(X_test)

#     fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred[:,1], pos_label=1)

#     summary = {'model' : [model], 
#                'log_loss' : [metrics.log_loss(y_test, y_pred)], 
#                'acc' : [metrics.accuracy_score(y_test, clf.predict(X_test))], 
#                'auc' : [metrics.auc(fpr, tpr)]}
#     summary_df = pd.DataFrame(summary)

#     fit_df = stat_df
#     X_save = fit_df[features]
#     y_save = fit_df['y']
#     clf.fit(X_save, y_save)
    rf_reg=RandomForestClassifier(n_estimators=500, max_features = 5)
    rf_reg.fit(X_train,y_train)

    y_pred=rf_reg.predict_proba(X_test)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred[:,1], pos_label=1)

    summary = {'model' : [model], 
               'log_loss' : [metrics.log_loss(y_test, y_pred)], 
               'acc' : [metrics.accuracy_score(y_test, rf_reg.predict(X_test))], 
               'auc' : [metrics.auc(fpr, tpr)]}
    summary_df = pd.DataFrame(summary)

    fit_df = stat_df
    X_save = fit_df[features]
    y_save = fit_df['y']
    rf_reg.fit(X_save, y_save)
    
    nm = "/Users/davidkim/Documents/kill_total_diff_models/" + model + '.joblib'
    diff_summary_df = pd.concat([summary_df, diff_summary_df], join= 'outer',ignore_index = True)

    joblib.dump(rf_reg,nm)
#     joblib.dump(clf,nm)


# In[ ]:


#wc 본선 진출 팀 모든 팀의 매치업을 만드는 과정 
team_df = raw_data[(raw_data.league == "wc") & (raw_data.date >= datetime(2022,10,8,0,0,0))]
# team_df = raw_data[(raw_data.league == "wc")]
team_list = team_df.team.unique()

team_comb_df = pd.DataFrame()

for i in range(len(team_list)) : 
    team100 = team_list[i]

    pair_df = pd.DataFrame()
    pair_df['team'] = np.repeat(team100,len(team_list))
    pair_df['oppTeam'] = list(team_list)
    pair_df['date'] = np.repeat(datetime.now(),len(team_list))
    pair_df['league'] = 'wc'
    team_comb_df = pd.concat([team_comb_df, pair_df], join= 'outer',ignore_index = True)

team_comb_df = team_comb_df[(team_comb_df.team != team_comb_df.oppTeam)]
team_comb_df = team_comb_df.reset_index()


# In[ ]:


#상위에서 만든 매치업들에 feature들을 모델링과 동일한 방법으로 수행 
#본선만의 데이터를 활용할 경우 condition을 wc로, 아닌경우 빼고 하는 것이 좋다 
#다시 강조 : General한 적당한 결과를 error없이 처리하려면 daysnum을 길게 가져가고, 
#롤드컵 상위 라운드와 같이, 최근의 데이터가 중요한 경우 시간을 줄이는 것이 대체로 결과가 좋은 것 같음 
pred_stat_df = timeline_team_fn(team_comb_df, condition = "wc")


# In[ ]:


#pred_stat_df에 가능한 ai prob를 추가
for ijk in range(0,len(pred_stat_df)) : 
#     print(ijk)
    stat_add = pred_stat_df.iloc[ijk,]
    team = stat_add.team
    oppTeam = stat_add.oppTeam
    date = stat_add.date
    ai_candi = ai_prob[(ai_prob.team1 == team)&(ai_prob.team2== oppTeam)]
    if len(ai_candi) >= 1 : 
        ai_date = nearest(ai_candi.date,date)
        ai_sel = ai_candi[(ai_prob.date == ai_date)]
        prob = ai_sel.team1Prob.tolist()[0]
#         pred_stat_df.at[ijk,'team1Prob']=np.where(prob >= 0.5 , prob + 0.15, prob-0.15)
        pred_stat_df.at[ijk,'team1Prob']=prob
        pred_stat_df.at[ijk,'team2Prob']=1-prob
        del prob


# In[ ]:


#ai prob이 없는 것들은 예측이 안되므로 제외한다 
pred_stat_df = pred_stat_df.dropna()
pred_stat_df = pred_stat_df.reset_index()


# In[ ]:


model_df = pred_stat_df[['team','oppTeam','date','league','team1Prob','team2Prob']]
#model_df = model_df[,0:6]
model_df.head(10)
x_df = pred_stat_df[features]


# In[ ]:


#킬 차이 모델을 상기 만든 pred_stat_df로 예측하고 결과대로 정리하는 과정 
#모델이 있는 핸디 모델들을 확률을 계산하고, 모델이 없는 핸디캡 확률들은 확률이 있는 값들을 활용하여 산출 (등차수열)
diff_summary_model_df = pd.DataFrame() 
for k in range(1,len(pred_stat_df)+1) : 
#     print(k)
    row = pred_stat_df.iloc[k-1:k]
    match = row[['team','oppTeam','date','league','team1Prob']]
    x = row[features]
    model_nm = "Lol-Pre-MoreKillsHandicapped-"
    model_suffix = "-Team"
    con_pred_df = pd.DataFrame()
    team100Odds = []
    team200Odds = []
    for h in range(0,len(std_diff_handi)) : 
        model = model_nm + str(std_diff_handi[h]) + model_suffix
        modelfile = model + ".joblib"
        pred_reg = joblib.load("/Users/davidkim/Documents/kill_total_diff_models/" + modelfile)
        pred =pred_reg.predict_proba(x)
        pred_0 = [item[1] for item in pred]
        team100Odds.append(pred_0[0])
        models.append(model)
        handis.append(std_diff_handi[h])
    term_0 = np.linspace(team100Odds[0],team100Odds[1],diff_range_0+1).tolist()
    term_1 = np.linspace(team100Odds[1],team100Odds[2],diff_range_1+1).tolist()
    smooth_team100Odds = term_0 + term_1
    del smooth_team100Odds[std_diff_loc_1]
    lower_bound = smooth_team100Odds[0] - smooth_team100Odds[1]
    upper_bound = smooth_team100Odds[len(smooth_team100Odds)-1] - smooth_team100Odds[len(smooth_team100Odds)-2]

    lower_term = []
    upper_term = []
    lower_start = smooth_team100Odds[0]
    upper_start = smooth_team100Odds[len(smooth_team100Odds)-1]
    for m in range(1, handi_nomodel_cnt+1) : 
        lower_start += lower_bound
        upper_start += upper_bound
        lower_term.append(lower_start)
        upper_term.append(upper_start)
    smooth_team100Odds = lower_term + smooth_team100Odds + upper_term
    # smooth_team100Odds
    #     smooth_team100Odds = pd.Series(team100Odds).ewm(span = 5).mean()
    match_df = match.append([match]*(len(diff_handi)-1), ignore_index=True)
    match_df['model'] = diff_model_list
    match_df['handi'] = diff_handi
    match_df['team100Odds'] = smooth_team100Odds
    match_df['team200Odds'] = 1-match_df.team100Odds
#     match_df['team100ratio'] = (1/match_df.team100Odds)
#     match_df['team200ratio'] = (1/match_df.team200Odds)
    diff_summary_model_df = pd.concat([diff_summary_model_df, match_df], join= 'outer',ignore_index = True)
    del match_df
    del match


# In[ ]:


#킬 토탈 모델의 예측값 산출 : 킬 차이와 큰 다른 점 없음 
total_summary_model_df = pd.DataFrame() 
for k in range(1,len(pred_stat_df)+1) : 
#     print(k)
    row = pred_stat_df.iloc[k-1:k]
    match = row[['team','oppTeam','date','league','team1Prob']]
    x = row[features]
    model_nm = "Lol-Pre-totalKills-"
    model_suffix = "-UnderOver"
    con_pred_df = pd.DataFrame()
    team100Odds = []
    team200Odds = []
    for h in range(0,len(std_total_handi)) : 
        model = model_nm + str(std_total_handi[h]) + model_suffix
        modelfile = model + ".joblib"
        pred_reg = joblib.load("/Users/davidkim/Documents/kill_total_diff_models/" + modelfile)
        pred =pred_reg.predict_proba(x)
        pred_0 = [item[1] for item in pred]
        team100Odds.append(pred_0[0])
        models.append(model)
        handis.append(std_total_handi[h])
    term_0 = np.linspace(team100Odds[0],team100Odds[1],total_range_0+1).tolist()
    term_1 = np.linspace(team100Odds[1],team100Odds[2],total_range_1+1).tolist()
    smooth_team100Odds = term_0 + term_1
    del smooth_team100Odds[std_total_loc_1]
    match_df = match.append([match]*(len(total_handi)-1), ignore_index=True)
    match_df['model'] = total_model_list
    match_df['handi'] = total_handi
    match_df['team100Odds'] = smooth_team100Odds
    match_df['team200Odds'] = 1-match_df.team100Odds
#     match_df['team100ratio'] = (1/match_df.team100Odds)
#     match_df['team200ratio'] = (1/match_df.team200Odds)
    total_summary_model_df = pd.concat([total_summary_model_df, match_df], join= 'outer',ignore_index = True)
    del match_df
    del match


# In[ ]:


schedule_df = schedule[['uniqueId','matchId','eventDay','matchStartTime','team1','team2']]
schedule_df = schedule_df.reset_index()


# In[ ]:


#킬 차이 모델을 정리 하여 db 에 올릴 테이블 형태로 정리 
schedule_df = schedule[['uniqueId','matchId','eventDay','matchStartTime','team1','team2']]
schedule_df = schedule_df.reset_index()
schedule_df = schedule_df[(schedule_df.matchStartTime >= datetime(2022,10,8,0,0,0))].reset_index()
db_df = pd.DataFrame()

for p in range(1,len(schedule_df)+1) : 
#     print(p)
    row = schedule_df.iloc[p-1:p]

    team1 = row.team1.tolist()[0]
    team2 = row.team2.tolist()[0]
    team1id = teams[(teams.name.str.lower() == team1.lower())]
    team1id = int(team1id.teamId.tolist()[0])
    team2id = teams[(teams.name.str.lower() == team2.lower())]
    team2id = int(team2id.teamId.tolist()[0])
    row['team1Id'] = team1id
    row['team2Id'] = team2id
    comb1 = diff_summary_model_df[(diff_summary_model_df.team == team1) & (diff_summary_model_df.oppTeam == team2)]
#     comb2 = diff_summary_model_df[(diff_summary_model_df.team == team2) & (diff_summary_model_df.oppTeam == team1)]
    Odds1 = comb1.team100Odds.tolist()
#     Odds2 = comb2.team200Odds.tolist()
#     Odds2 = list(reversed(Odds2))
    handi = comb1.handi.tolist()
    model = comb1.model.tolist()
    nm = ["MoreKillsHandicapped"]*len(model)
#     Odds = [Odds1,Odds2]
#     avg_Odds= np.mean(Odds,axis =0)
    avg_Odds= Odds1
    row = row.append([row]*(len(avg_Odds)-1), ignore_index=True)
    row['marketName'] = ["Lol-Pre-MoreKillsHandicapped-Team"]*len(handi) 
    row['displayName'] = nm
    row['handicap'] = handi
    row['team1Prob'] = avg_Odds
    row['team2Prob'] = 1-row.team1Prob
    row = row[(row.team1Prob >= 0.15) & (row.team1Prob <= 0.85)]
    db_df = pd.concat([db_df, row], join= 'outer',ignore_index = True)
    del row
# row.head(2)


# In[ ]:


killtotal_db_df = db_df 


# In[ ]:


#킬 토탈 모델을 정리 하여 db 에 올릴 테이블 형태로 정리 
schedule_df = schedule[['uniqueId','matchId','eventDay','matchStartTime','team1','team2']]
schedule_df = schedule_df.reset_index()
schedule_df = schedule_df[(schedule_df.matchStartTime >= datetime(2022,10,8,0,0,0))].reset_index()
db_df = pd.DataFrame()

for p in range(1,len(schedule_df)+1) : 
#     print(p)
    row = schedule_df.iloc[p-1:p]

    team1 = row.team1.tolist()[0]
    team2 = row.team2.tolist()[0]
    team1id = teams[(teams.name.str.lower() == team1.lower())]
    team1id = int(team1id.teamId.tolist()[0])
    team2id = teams[(teams.name.str.lower() == team2.lower())]
    team2id = int(team2id.teamId.tolist()[0])
    row['team1Id'] = team1id
    row['team2Id'] = team2id
    comb1 = total_summary_model_df[(total_summary_model_df.team == team1) & (total_summary_model_df.oppTeam == team2)]
#     comb2 = diff_summary_model_df[(diff_summary_model_df.team == team2) & (diff_summary_model_df.oppTeam == team1)]
    Odds1 = comb1.team100Odds.tolist()
#     Odds2 = comb2.team200Odds.tolist()
#     Odds2 = list(reversed(Odds2))
    handi = comb1.handi.tolist()
#     model = comb1.model.tolist()
    model = ["Lol-Pre-totalKills-UnderOver"]*len(handi)
    nm = ["totalKills"]*len(handi)
#     Odds = [Odds1,Odds2]
#     avg_Odds= np.mean(Odds,axis =0)
    avg_Odds= Odds1
    row = row.append([row]*(len(avg_Odds)-1), ignore_index=True)
    row['marketName'] = model 
    row['displayName'] = nm
    row['handicap'] = handi
    row['team1Prob'] = avg_Odds
    row['team2Prob'] = 1-row.team1Prob
    row = row[(row.team1Prob >= 0.15) & (row.team1Prob <= 0.85)]
    db_df = pd.concat([db_df, row], join= 'outer',ignore_index = True)
    del row
# row.head(2)


# In[ ]:


kill_diff_df = db_df


# In[ ]:


total_db_df = pd.concat([kill_diff_df, kill_total_df], join= 'outer',ignore_index = True)
total_db_df = total_db_df.drop(['level_0'], axis = 1)


# In[ ]:


total_db_df.to_sql('ds_kill_total_diff_odds_table', con=md_engine, if_exists='replace')

