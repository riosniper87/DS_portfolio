{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "53a41b32",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd \n",
    "import numpy as np \n",
    "import mysql.connector\n",
    "#import dotenv\n",
    "import os\n",
    "import re\n",
    "from datetime import timedelta\n",
    "from datetime import datetime\n",
    "import shutil\n",
    "import subprocess\n",
    "import enum\n",
    "from sqlalchemy import create_engine\n",
    "import pymysql\n",
    "from datetime import date\n",
    "import math\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn import metrics\n",
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6bad728c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#쿼리 불러오는 function\n",
    "def model_import_table_fn(query):\n",
    "    con = mysql.connector.connect(user=db_user, password=db_passwd, host =db_host, database  = db_db, port = int(db_port))\n",
    "    df = pd.read_sql(query, con)\n",
    "    con.cursor().close()\n",
    "    con.close()\n",
    "    del con\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "5a3b2ad9",
   "metadata": {},
   "outputs": [],
   "source": [
    "#dotenv.load_dotenv('C:/Users/teamsnowball/models/mid_kill_5/d2.env', verbose=True, override=True)\n",
    "db_host = 'teamsnowball.co'\n",
    "db_user = 'tsb'\n",
    "db_passwd = 'tmshdnqhf0731'\n",
    "db_port = 3500\n",
    "db_db = 'tipster'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e5cb8fc7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def md_connect(user, password, db, host, port=db_port):\n",
    "    url = 'mysql+mysqldb://{}:{}@{}:{}/{}'.format(user, password, host, port, db)\n",
    "    engine = create_engine(url)\n",
    "    return engine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "79ebf544",
   "metadata": {},
   "outputs": [],
   "source": [
    "#위의 md_connect를 쓰는데 MySQLdb 모듈을 못찾는 에러가 나는 경우 실행하는 명령어 \n",
    "#pymysql을 MySQLdb로 인스톨 하는 기능으로 알고 있음.\n",
    "pymysql.install_as_MySQLdb()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "d0f378e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "md_engine = md_connect(user = db_user, password = db_passwd, db = db_db, host = db_host, port = 3500)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "f0774e8b",
   "metadata": {},
   "outputs": [],
   "source": [
    "raw_data = model_import_table_fn(\"SELECT * FROM h2h_stats\")\n",
    "# raw_data = raw_data[(raw_data.camp == 'blue')].reset_index()\n",
    "schedule = model_import_table_fn(\"SELECT * FROM league_schedule where NOT team1 = 'TBD'\")\n",
    "db_db = 'teamsnowball'\n",
    "ai_prob = model_import_table_fn(\"SELECT * FROM league_schedule\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "35c06bf5",
   "metadata": {},
   "outputs": [],
   "source": [
    "#이걸 위에 데이터 프레임에 맞도록 설계 \n",
    "#홀짝으로 데이터 거르고 돌리면 될듯. range kill death, assist 정도만 활용 \n",
    "#이걸로 타임라인 만들고, 적당한 경기수가 있는 지점부터 월즈 전까지로 학습. \n",
    "# condition == \"wc\" 시 world22 데이터로만 featrue 생성 \n",
    "# days num 은 경기 시작일 기준 몇일 전까지 데이터를 feature로 사용하는가\n",
    "# 넉넉하게 값을 줄 경우 통계량이 잘 나오지만, 최근 팀의 결과가 적게 반영되는 이슈가 있고, \n",
    "# 타이트하게 값을 줄 경우 데이터가 0 이라 데이터가 안나오는 경우가 생김 \n",
    "def timeline_team_fn(data, condition = False, daysnum = 15) :     \n",
    "    \n",
    "    stat_df = pd.DataFrame()\n",
    "    \n",
    "    #행별 전처리 진행\n",
    "    for ij in range(0,len(data)):\n",
    "#         print(ij)\n",
    "        dat = data.loc[[ij, ]] \n",
    "        team = dat.team.tolist()[0]\n",
    "        opp_team = dat.oppTeam.tolist()[0]\n",
    "        league = dat.league.tolist()[0]\n",
    "        max_date = dat.date.tolist()[0]\n",
    "        min_date = max_date - pd.DateOffset(days = days_num)\n",
    "        if condition == \"wc\" : \n",
    "            filter_dat = raw_data[(raw_data.league == \"wc\") & (raw_data.date >= datetime(2022,10,8,0,0,0))]\n",
    "        else : \n",
    "            std_dat = raw_data\n",
    "            filter_dat = std_dat[(std_dat.date <= max_date) & (std_dat.date >= min_date)]\n",
    "            \n",
    "        \n",
    "        filter_dat_blue = filter_dat[(filter_dat.team == team)]\n",
    "        filter_dat_red  = filter_dat[(filter_dat.oppTeam == team)]\n",
    "        opp_filter_dat_blue = filter_dat[(filter_dat.team == opp_team)]\n",
    "        opp_filter_dat_red  = filter_dat[(filter_dat.oppTeam == opp_team)]\n",
    "        length = len(filter_dat_blue) + len(filter_dat_red)\n",
    "        opp_length = len(opp_filter_dat_blue) + len(opp_filter_dat_red)\n",
    "        tot_data = {'team' : [team], \n",
    "                    'oppTeam' : [opp_team],\n",
    "                    'date' : [max_date],\n",
    "                    'league' : [league],\n",
    "                    'len' : [length],\n",
    "                    'avg_gamelength' : [np.mean([filter_dat_blue.gameLength.tolist() + filter_dat_red.gameLength.tolist()])],\n",
    "                    'avg_kills' : [np.mean([filter_dat_blue.kills.tolist() + filter_dat_red.oppKills.tolist()])],\n",
    "                    'avg_deaths' : [np.mean([filter_dat_blue.deaths.tolist() + filter_dat_red.oppDeaths.tolist()])],\n",
    "                    'avg_assists' : [np.mean([filter_dat_blue.assists.tolist() + filter_dat_red.oppAssists.tolist()])],\n",
    "                    'max_kills' : [np.amax([filter_dat_blue.kills.tolist() + filter_dat_red.oppKills.tolist()])],\n",
    "                    'max_deaths' : [np.amax([filter_dat_blue.deaths.tolist() + filter_dat_red.oppDeaths.tolist()])],\n",
    "                    'max_assists' : [np.amax([filter_dat_blue.assists.tolist() + filter_dat_red.oppAssists.tolist()])],\n",
    "                    'min_kills' : [np.amin([filter_dat_blue.kills.tolist() + filter_dat_red.oppKills.tolist()])],\n",
    "                    'min_deaths' : [np.amin([filter_dat_blue.deaths.tolist() + filter_dat_red.oppDeaths.tolist()])],\n",
    "                    'min_assists' : [np.amin([filter_dat_blue.assists.tolist() + filter_dat_red.oppAssists.tolist()])],\n",
    "                    'opp_len' : [opp_length],\n",
    "                    'opp_avg_gamelength' : [np.mean([opp_filter_dat_blue.gameLength.tolist() + opp_filter_dat_red.gameLength.tolist()])],\n",
    "                    'opp_avg_kills' : [np.mean([opp_filter_dat_blue.oppKills.tolist() + opp_filter_dat_red.kills.tolist()])],\n",
    "                    'opp_avg_deaths': [np.mean([opp_filter_dat_blue.oppDeaths.tolist() + opp_filter_dat_red.deaths.tolist()])],\n",
    "                    'opp_avg_assists': [np.mean([opp_filter_dat_blue.oppAssists.tolist() + opp_filter_dat_red.assists.tolist()])],\n",
    "                    'opp_max_kills' : [np.amax([opp_filter_dat_blue.oppKills.tolist() + opp_filter_dat_red.kills.tolist()])],\n",
    "                    'opp_max_deaths' : [np.amax([opp_filter_dat_blue.oppDeaths.tolist() + opp_filter_dat_red.deaths.tolist()])],\n",
    "                    'opp_max_assists' : [np.amax([opp_filter_dat_blue.oppAssists.tolist() + opp_filter_dat_red.assists.tolist()])],\n",
    "                    'opp_min_kills' : [np.amin([opp_filter_dat_blue.oppKills.tolist() + opp_filter_dat_red.kills.tolist()])],\n",
    "                    'opp_min_deaths' : [np.amin([opp_filter_dat_blue.oppDeaths.tolist() + opp_filter_dat_red.deaths.tolist()])],\n",
    "                    'opp_min_assists' : [np.amin([opp_filter_dat_blue.oppAssists.tolist() + opp_filter_dat_red.assists.tolist()])]}\n",
    "                   \n",
    "\n",
    "\n",
    "\n",
    "        tot_dat = pd.DataFrame(tot_data)\n",
    "        stat_df = pd.concat([stat_df, tot_dat], join= 'outer',ignore_index = True)\n",
    "\n",
    "    return stat_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "919d3a8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def nearest(items, pivot):\n",
    "    return min(items, key=lambda x: abs(x - pivot))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a049e790",
   "metadata": {},
   "outputs": [],
   "source": [
    "#raw data 기반, featrue data (stat_df) 생성\n",
    "raw_data = raw_data[(raw_data.camp == 'blue')].reset_index()\n",
    "stat_df = timeline_team_fn(raw_data)\n",
    "stat_df = timeline_team_fn(raw_data)\n",
    "stat_df['kills'] = raw_data.kills\n",
    "stat_df['oppKills'] = raw_data.oppKills\n",
    "stat_df['result'] = raw_data.result\n",
    "#stat_df = stat_df[(stat_df.league != 'vcs')].reset_index()\n",
    "#stat_df = stat_df[(stat_df.league != 'lpl')].reset_index()\n",
    "stat_df['team1Prob'] = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "21dc4354",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ai prob 를 stat_df 에 붙이는 과정 \n",
    "for ijk in range(0,len(stat_df)) : \n",
    "    #print(ijk)\n",
    "    stat_add = stat_df.iloc[ijk,]\n",
    "    team = stat_add.team\n",
    "    oppTeam = stat_add.oppTeam\n",
    "    date = stat_add.date\n",
    "    ai_candi = ai_prob[(ai_prob.team1 == team)&(ai_prob.team2== oppTeam)]\n",
    "    if len(ai_candi) >= 1 : \n",
    "        ai_date = nearest(ai_candi.date,date)\n",
    "        ai_sel = ai_candi[(ai_prob.date == ai_date)]\n",
    "        prob =ai_sel.team1Prob.tolist()[0]\n",
    "        stat_df.at[ijk,'team1Prob']=prob\n",
    "        del prob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "feaa1c61",
   "metadata": {},
   "outputs": [],
   "source": [
    "#ai prob가 없는 매치들은 쓸수 없으므로 제외 \n",
    "stat_df = stat_df[(stat_df.team1Prob != 0)]\n",
    "stat_df = stat_df.dropna()\n",
    "stat_df = stat_df.reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "651d1221",
   "metadata": {},
   "outputs": [],
   "source": [
    "stat_df['team2Prob'] = 1-stat_df.team1Prob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bd11e4bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "#feature list 생성, 나중에 feature selection 같은걸 하려고 하였음. \n",
    "features = ['team1Prob','team2Prob','avg_gamelength','opp_avg_gamelength','avg_kills','avg_deaths','avg_assists','max_kills',\n",
    "            'max_deaths','max_assists','min_kills','min_deaths','min_assists','opp_avg_kills','opp_avg_deaths',\n",
    "            'opp_avg_assists','opp_max_kills','opp_max_deaths','opp_max_assists','opp_min_kills','opp_min_deaths',\n",
    "            'opp_min_assists']\n",
    "#킬 토탈 핸디 범위 설정 (전체)\n",
    "total_handi = np.arange(15.5,35.5,1).tolist()\n",
    "\n",
    "#킬 토탈 핸디 모델링할 핸디 설정 (핸디 가장 큰 값 / 작은 값/ 중간을 재고 나머지 핸디 값들은 저 세 값들을 사용한 \n",
    "#등차 수열로 만듬)\n",
    "std_total_handi = [total_handi[0],total_handi[round(len(total_handi)/2)],total_handi[len(total_handi)-1]]\n",
    "std_total_loc_0 = total_handi.index(std_total_handi[0])\n",
    "std_total_loc_1 = total_handi.index(std_total_handi[1])\n",
    "std_total_loc_2 = total_handi.index(std_total_handi[2])\n",
    "\n",
    "#등차 수열 레인지 \n",
    "total_range_0 = std_total_loc_1-std_total_loc_0\n",
    "total_range_1 = std_total_loc_2-std_total_loc_1\n",
    "\n",
    "#모델 리스트 설정 \n",
    "total_model_list = []\n",
    "for x in total_handi : \n",
    "    total_model_list.append('Lol-Pre-Kill-Total-Handicapped-' + str(x) + '-Team')\n",
    "\n",
    "#킬 차이 모델 범위 설정 (이후는 킬 토탈과 비슷하므로 생략)\n",
    "diff_handi = np.arange(-22.5,23.5,1).tolist()\n",
    "#킬 토탈과 다른 점은 no model cnt 값인데, 킬 차이 모델은 핸디캡 범위가 기존 생각보다 넓어서 \n",
    "#양 끝값 보다는 앞에서 no model cnt 번째 값, 뒤에서도 같은 no model cnt 번째 값만 가지고 모델링 하는 과정\n",
    "handi_nomodel_cnt = 7\n",
    "std_diff_handi = [diff_handi[handi_nomodel_cnt],diff_handi[round(len(diff_handi)/2)],\n",
    "                  diff_handi[len(diff_handi)-(handi_nomodel_cnt+1)]]\n",
    "\n",
    "std_diff_loc_0 = diff_handi.index(std_diff_handi[0])\n",
    "std_diff_loc_1 = diff_handi.index(std_diff_handi[1])\n",
    "std_diff_loc_2 = diff_handi.index(std_diff_handi[2])\n",
    "\n",
    "diff_range_0 = std_diff_loc_1-std_diff_loc_0\n",
    "diff_range_1 = std_diff_loc_2-std_diff_loc_1\n",
    "\n",
    "diff_model_list = []\n",
    "for x in diff_handi : \n",
    "    diff_model_list.append('Lol-Pre-MoreKillsHandicapped-' + str(x) + '-Team')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13ad0aec",
   "metadata": {},
   "outputs": [],
   "source": [
    "#킬 토탈 모델링, std_total_handi 안에 있는 값들만 local에 저장하는 형식으로 진행 \n",
    "#10월 8일 이전의 게임들로 학습하여, 그 이후의 게임들을 맞추는 형식으로 성능을 확인\n",
    "#성능 확인 값 (logloss, acc, auc)들을 각 모델별로 보여준다. \n",
    "#randomforest를 사용하도록 되어있으며 , 주석 처리된 부분은 logregression 사용이 필요할 경우 활용할 수 있다. \n",
    "total_summary_df = pd.DataFrame()\n",
    "for i in range(0,len(std_total_handi)) : \n",
    "    handi = std_total_handi[i]\n",
    "    model = 'Lol-Pre-totalKills-' + str(handi) + '-UnderOver'\n",
    "    stat_df['y'] = np.where(stat_df.kills + stat_df.oppKills <= handi,0,1)\n",
    "\n",
    "    #handi = diff_handi[0]\n",
    "    #model = 'Lol-Pre-Kill-Diff-Handicapped-' + str(handi) + '-Team'\n",
    "    #stat_df['y'] = np.where(stat_df.kills + handi <= stat_df.opp_kills ,0,1)\n",
    "\n",
    "    train_df = stat_df[(stat_df.date < datetime(2022,10,8,0,0,0))]\n",
    "    X_train  = train_df[features]\n",
    "    y_train  = train_df['y']\n",
    "\n",
    "    test_df  = stat_df[(stat_df.date >= datetime(2022,10,8,0,0,0))]\n",
    "    X_test   = test_df[features]\n",
    "    y_test   = test_df['y']\n",
    "\n",
    "    rf_reg=RandomForestClassifier(n_estimators=500, max_features = 3)\n",
    "    rf_reg.fit(X_train,y_train)\n",
    "\n",
    "    y_pred=rf_reg.predict_proba(X_test)\n",
    "\n",
    "#     clf = LogisticRegression(random_state=0).fit(X_train, y_train)\n",
    "#     y_pred = clf.predict_proba(X_test)\n",
    "\n",
    "\n",
    "#     summary = {'model' : [model], \n",
    "#                'log_loss' : [metrics.log_loss(y_test, y_pred)], \n",
    "#                'acc' : [metrics.accuracy_score(y_test, clf.predict(X_test))], \n",
    "#                'auc' : [metrics.auc(fpr, tpr)]}\n",
    "#     summary_df = pd.DataFrame(summary)\n",
    "\n",
    "#     fit_df = stat_df\n",
    "#     X_save = fit_df[features]\n",
    "#     y_save = fit_df['y']\n",
    "#     clf.fit(X_save, y_save)\n",
    "    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred[:,1], pos_label=1)\n",
    "    \n",
    "    summary = {'model' : [model], \n",
    "               'log_loss' : [metrics.log_loss(y_test, y_pred)], \n",
    "               'acc' : [metrics.accuracy_score(y_test, rf_reg.predict(X_test))], \n",
    "               'auc' : [metrics.auc(fpr, tpr)]}\n",
    "    summary_df = pd.DataFrame(summary)\n",
    "    \n",
    "    fit_df = stat_df\n",
    "    X_save = fit_df[features]\n",
    "    y_save = fit_df['y']\n",
    "    rf_reg.fit(X_save, y_save)\n",
    "\n",
    "    nm = \"/Users/davidkim/Documents/kill_total_diff_models/\" + model + '.joblib'\n",
    "    total_summary_df = pd.concat([summary_df, total_summary_df], join= 'outer',ignore_index = True)\n",
    "\n",
    "    joblib.dump(rf_reg,nm)\n",
    "#     joblib.dump(clf,nm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "020f8b9f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#킬 차이 모델, 킬 토탈과 같은 컨셉으로 이루어져 있다.\n",
    "diff_summary_df = pd.DataFrame()\n",
    "for i in range(0,len(std_diff_handi)) : \n",
    "\n",
    "    handi = std_diff_handi[i]\n",
    "    model = 'Lol-Pre-MoreKillsHandicapped-' + str(handi) + '-Team'\n",
    "    stat_df['y'] = np.where(stat_df.kills + handi <= stat_df.oppKills ,0,1)\n",
    "\n",
    "    #train_df = stat_df[(stat_df.league != 'wc')]\n",
    "    train_df = stat_df[(stat_df.date < datetime(2022,10,8,0,0,0))]\n",
    "    X_train  = train_df[features]\n",
    "    y_train  = train_df['y']\n",
    "\n",
    "    test_df  = stat_df[(stat_df.date >= datetime(2022,10,8,0,0,0))]\n",
    "    X_test   = test_df[features]\n",
    "    y_test   = test_df['y']\n",
    "\n",
    "#     clf = LogisticRegression(random_state=0).fit(X_train, y_train)\n",
    "#     y_pred = clf.predict_proba(X_test)\n",
    "\n",
    "#     fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred[:,1], pos_label=1)\n",
    "\n",
    "#     summary = {'model' : [model], \n",
    "#                'log_loss' : [metrics.log_loss(y_test, y_pred)], \n",
    "#                'acc' : [metrics.accuracy_score(y_test, clf.predict(X_test))], \n",
    "#                'auc' : [metrics.auc(fpr, tpr)]}\n",
    "#     summary_df = pd.DataFrame(summary)\n",
    "\n",
    "#     fit_df = stat_df\n",
    "#     X_save = fit_df[features]\n",
    "#     y_save = fit_df['y']\n",
    "#     clf.fit(X_save, y_save)\n",
    "    rf_reg=RandomForestClassifier(n_estimators=500, max_features = 5)\n",
    "    rf_reg.fit(X_train,y_train)\n",
    "\n",
    "    y_pred=rf_reg.predict_proba(X_test)\n",
    "\n",
    "    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred[:,1], pos_label=1)\n",
    "\n",
    "    summary = {'model' : [model], \n",
    "               'log_loss' : [metrics.log_loss(y_test, y_pred)], \n",
    "               'acc' : [metrics.accuracy_score(y_test, rf_reg.predict(X_test))], \n",
    "               'auc' : [metrics.auc(fpr, tpr)]}\n",
    "    summary_df = pd.DataFrame(summary)\n",
    "\n",
    "    fit_df = stat_df\n",
    "    X_save = fit_df[features]\n",
    "    y_save = fit_df['y']\n",
    "    rf_reg.fit(X_save, y_save)\n",
    "    \n",
    "    nm = \"/Users/davidkim/Documents/kill_total_diff_models/\" + model + '.joblib'\n",
    "    diff_summary_df = pd.concat([summary_df, diff_summary_df], join= 'outer',ignore_index = True)\n",
    "\n",
    "    joblib.dump(rf_reg,nm)\n",
    "#     joblib.dump(clf,nm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "992fdb12",
   "metadata": {},
   "outputs": [],
   "source": [
    "#wc 본선 진출 팀 모든 팀의 매치업을 만드는 과정 \n",
    "team_df = raw_data[(raw_data.league == \"wc\") & (raw_data.date >= datetime(2022,10,8,0,0,0))]\n",
    "# team_df = raw_data[(raw_data.league == \"wc\")]\n",
    "team_list = team_df.team.unique()\n",
    "\n",
    "team_comb_df = pd.DataFrame()\n",
    "\n",
    "for i in range(len(team_list)) : \n",
    "    team100 = team_list[i]\n",
    "\n",
    "    pair_df = pd.DataFrame()\n",
    "    pair_df['team'] = np.repeat(team100,len(team_list))\n",
    "    pair_df['oppTeam'] = list(team_list)\n",
    "    pair_df['date'] = np.repeat(datetime.now(),len(team_list))\n",
    "    pair_df['league'] = 'wc'\n",
    "    team_comb_df = pd.concat([team_comb_df, pair_df], join= 'outer',ignore_index = True)\n",
    "\n",
    "team_comb_df = team_comb_df[(team_comb_df.team != team_comb_df.oppTeam)]\n",
    "team_comb_df = team_comb_df.reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "913060e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "#상위에서 만든 매치업들에 feature들을 모델링과 동일한 방법으로 수행 \n",
    "#본선만의 데이터를 활용할 경우 condition을 wc로, 아닌경우 빼고 하는 것이 좋다 \n",
    "#다시 강조 : General한 적당한 결과를 error없이 처리하려면 daysnum을 길게 가져가고, \n",
    "#롤드컵 상위 라운드와 같이, 최근의 데이터가 중요한 경우 시간을 줄이는 것이 대체로 결과가 좋은 것 같음 \n",
    "pred_stat_df = timeline_team_fn(team_comb_df, condition = \"wc\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3e76537b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pred_stat_df에 가능한 ai prob를 추가\n",
    "for ijk in range(0,len(pred_stat_df)) : \n",
    "#     print(ijk)\n",
    "    stat_add = pred_stat_df.iloc[ijk,]\n",
    "    team = stat_add.team\n",
    "    oppTeam = stat_add.oppTeam\n",
    "    date = stat_add.date\n",
    "    ai_candi = ai_prob[(ai_prob.team1 == team)&(ai_prob.team2== oppTeam)]\n",
    "    if len(ai_candi) >= 1 : \n",
    "        ai_date = nearest(ai_candi.date,date)\n",
    "        ai_sel = ai_candi[(ai_prob.date == ai_date)]\n",
    "        prob = ai_sel.team1Prob.tolist()[0]\n",
    "#         pred_stat_df.at[ijk,'team1Prob']=np.where(prob >= 0.5 , prob + 0.15, prob-0.15)\n",
    "        pred_stat_df.at[ijk,'team1Prob']=prob\n",
    "        pred_stat_df.at[ijk,'team2Prob']=1-prob\n",
    "        del prob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "21b3d543",
   "metadata": {},
   "outputs": [],
   "source": [
    "#ai prob이 없는 것들은 예측이 안되므로 제외한다 \n",
    "pred_stat_df = pred_stat_df.dropna()\n",
    "pred_stat_df = pred_stat_df.reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "57d988cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "model_df = pred_stat_df[['team','oppTeam','date','league','team1Prob','team2Prob']]\n",
    "#model_df = model_df[,0:6]\n",
    "model_df.head(10)\n",
    "x_df = pred_stat_df[features]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "03c916ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "#킬 차이 모델을 상기 만든 pred_stat_df로 예측하고 결과대로 정리하는 과정 \n",
    "#모델이 있는 핸디 모델들을 확률을 계산하고, 모델이 없는 핸디캡 확률들은 확률이 있는 값들을 활용하여 산출 (등차수열)\n",
    "diff_summary_model_df = pd.DataFrame() \n",
    "for k in range(1,len(pred_stat_df)+1) : \n",
    "#     print(k)\n",
    "    row = pred_stat_df.iloc[k-1:k]\n",
    "    match = row[['team','oppTeam','date','league','team1Prob']]\n",
    "    x = row[features]\n",
    "    model_nm = \"Lol-Pre-MoreKillsHandicapped-\"\n",
    "    model_suffix = \"-Team\"\n",
    "    con_pred_df = pd.DataFrame()\n",
    "    team100Odds = []\n",
    "    team200Odds = []\n",
    "    for h in range(0,len(std_diff_handi)) : \n",
    "        model = model_nm + str(std_diff_handi[h]) + model_suffix\n",
    "        modelfile = model + \".joblib\"\n",
    "        pred_reg = joblib.load(\"/Users/davidkim/Documents/kill_total_diff_models/\" + modelfile)\n",
    "        pred =pred_reg.predict_proba(x)\n",
    "        pred_0 = [item[1] for item in pred]\n",
    "        team100Odds.append(pred_0[0])\n",
    "        models.append(model)\n",
    "        handis.append(std_diff_handi[h])\n",
    "    term_0 = np.linspace(team100Odds[0],team100Odds[1],diff_range_0+1).tolist()\n",
    "    term_1 = np.linspace(team100Odds[1],team100Odds[2],diff_range_1+1).tolist()\n",
    "    smooth_team100Odds = term_0 + term_1\n",
    "    del smooth_team100Odds[std_diff_loc_1]\n",
    "    lower_bound = smooth_team100Odds[0] - smooth_team100Odds[1]\n",
    "    upper_bound = smooth_team100Odds[len(smooth_team100Odds)-1] - smooth_team100Odds[len(smooth_team100Odds)-2]\n",
    "\n",
    "    lower_term = []\n",
    "    upper_term = []\n",
    "    lower_start = smooth_team100Odds[0]\n",
    "    upper_start = smooth_team100Odds[len(smooth_team100Odds)-1]\n",
    "    for m in range(1, handi_nomodel_cnt+1) : \n",
    "        lower_start += lower_bound\n",
    "        upper_start += upper_bound\n",
    "        lower_term.append(lower_start)\n",
    "        upper_term.append(upper_start)\n",
    "    smooth_team100Odds = lower_term + smooth_team100Odds + upper_term\n",
    "    # smooth_team100Odds\n",
    "    #     smooth_team100Odds = pd.Series(team100Odds).ewm(span = 5).mean()\n",
    "    match_df = match.append([match]*(len(diff_handi)-1), ignore_index=True)\n",
    "    match_df['model'] = diff_model_list\n",
    "    match_df['handi'] = diff_handi\n",
    "    match_df['team100Odds'] = smooth_team100Odds\n",
    "    match_df['team200Odds'] = 1-match_df.team100Odds\n",
    "#     match_df['team100ratio'] = (1/match_df.team100Odds)\n",
    "#     match_df['team200ratio'] = (1/match_df.team200Odds)\n",
    "    diff_summary_model_df = pd.concat([diff_summary_model_df, match_df], join= 'outer',ignore_index = True)\n",
    "    del match_df\n",
    "    del match\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7eec41f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "#킬 토탈 모델의 예측값 산출 : 킬 차이와 큰 다른 점 없음 \n",
    "total_summary_model_df = pd.DataFrame() \n",
    "for k in range(1,len(pred_stat_df)+1) : \n",
    "#     print(k)\n",
    "    row = pred_stat_df.iloc[k-1:k]\n",
    "    match = row[['team','oppTeam','date','league','team1Prob']]\n",
    "    x = row[features]\n",
    "    model_nm = \"Lol-Pre-totalKills-\"\n",
    "    model_suffix = \"-UnderOver\"\n",
    "    con_pred_df = pd.DataFrame()\n",
    "    team100Odds = []\n",
    "    team200Odds = []\n",
    "    for h in range(0,len(std_total_handi)) : \n",
    "        model = model_nm + str(std_total_handi[h]) + model_suffix\n",
    "        modelfile = model + \".joblib\"\n",
    "        pred_reg = joblib.load(\"/Users/davidkim/Documents/kill_total_diff_models/\" + modelfile)\n",
    "        pred =pred_reg.predict_proba(x)\n",
    "        pred_0 = [item[1] for item in pred]\n",
    "        team100Odds.append(pred_0[0])\n",
    "        models.append(model)\n",
    "        handis.append(std_total_handi[h])\n",
    "    term_0 = np.linspace(team100Odds[0],team100Odds[1],total_range_0+1).tolist()\n",
    "    term_1 = np.linspace(team100Odds[1],team100Odds[2],total_range_1+1).tolist()\n",
    "    smooth_team100Odds = term_0 + term_1\n",
    "    del smooth_team100Odds[std_total_loc_1]\n",
    "    match_df = match.append([match]*(len(total_handi)-1), ignore_index=True)\n",
    "    match_df['model'] = total_model_list\n",
    "    match_df['handi'] = total_handi\n",
    "    match_df['team100Odds'] = smooth_team100Odds\n",
    "    match_df['team200Odds'] = 1-match_df.team100Odds\n",
    "#     match_df['team100ratio'] = (1/match_df.team100Odds)\n",
    "#     match_df['team200ratio'] = (1/match_df.team200Odds)\n",
    "    total_summary_model_df = pd.concat([total_summary_model_df, match_df], join= 'outer',ignore_index = True)\n",
    "    del match_df\n",
    "    del match"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b8ff03ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "schedule_df = schedule[['uniqueId','matchId','eventDay','matchStartTime','team1','team2']]\n",
    "schedule_df = schedule_df.reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b19442cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "#킬 차이 모델을 정리 하여 db 에 올릴 테이블 형태로 정리 \n",
    "schedule_df = schedule[['uniqueId','matchId','eventDay','matchStartTime','team1','team2']]\n",
    "schedule_df = schedule_df.reset_index()\n",
    "schedule_df = schedule_df[(schedule_df.matchStartTime >= datetime(2022,10,8,0,0,0))].reset_index()\n",
    "db_df = pd.DataFrame()\n",
    "\n",
    "for p in range(1,len(schedule_df)+1) : \n",
    "#     print(p)\n",
    "    row = schedule_df.iloc[p-1:p]\n",
    "\n",
    "    team1 = row.team1.tolist()[0]\n",
    "    team2 = row.team2.tolist()[0]\n",
    "    team1id = teams[(teams.name.str.lower() == team1.lower())]\n",
    "    team1id = int(team1id.teamId.tolist()[0])\n",
    "    team2id = teams[(teams.name.str.lower() == team2.lower())]\n",
    "    team2id = int(team2id.teamId.tolist()[0])\n",
    "    row['team1Id'] = team1id\n",
    "    row['team2Id'] = team2id\n",
    "    comb1 = diff_summary_model_df[(diff_summary_model_df.team == team1) & (diff_summary_model_df.oppTeam == team2)]\n",
    "#     comb2 = diff_summary_model_df[(diff_summary_model_df.team == team2) & (diff_summary_model_df.oppTeam == team1)]\n",
    "    Odds1 = comb1.team100Odds.tolist()\n",
    "#     Odds2 = comb2.team200Odds.tolist()\n",
    "#     Odds2 = list(reversed(Odds2))\n",
    "    handi = comb1.handi.tolist()\n",
    "    model = comb1.model.tolist()\n",
    "    nm = [\"MoreKillsHandicapped\"]*len(model)\n",
    "#     Odds = [Odds1,Odds2]\n",
    "#     avg_Odds= np.mean(Odds,axis =0)\n",
    "    avg_Odds= Odds1\n",
    "    row = row.append([row]*(len(avg_Odds)-1), ignore_index=True)\n",
    "    row['marketName'] = [\"Lol-Pre-MoreKillsHandicapped-Team\"]*len(handi) \n",
    "    row['displayName'] = nm\n",
    "    row['handicap'] = handi\n",
    "    row['team1Prob'] = avg_Odds\n",
    "    row['team2Prob'] = 1-row.team1Prob\n",
    "    row = row[(row.team1Prob >= 0.15) & (row.team1Prob <= 0.85)]\n",
    "    db_df = pd.concat([db_df, row], join= 'outer',ignore_index = True)\n",
    "    del row\n",
    "# row.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3411eea5",
   "metadata": {},
   "outputs": [],
   "source": [
    "killtotal_db_df = db_df "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "77645acb",
   "metadata": {},
   "outputs": [],
   "source": [
    "#킬 토탈 모델을 정리 하여 db 에 올릴 테이블 형태로 정리 \n",
    "schedule_df = schedule[['uniqueId','matchId','eventDay','matchStartTime','team1','team2']]\n",
    "schedule_df = schedule_df.reset_index()\n",
    "schedule_df = schedule_df[(schedule_df.matchStartTime >= datetime(2022,10,8,0,0,0))].reset_index()\n",
    "db_df = pd.DataFrame()\n",
    "\n",
    "for p in range(1,len(schedule_df)+1) : \n",
    "#     print(p)\n",
    "    row = schedule_df.iloc[p-1:p]\n",
    "\n",
    "    team1 = row.team1.tolist()[0]\n",
    "    team2 = row.team2.tolist()[0]\n",
    "    team1id = teams[(teams.name.str.lower() == team1.lower())]\n",
    "    team1id = int(team1id.teamId.tolist()[0])\n",
    "    team2id = teams[(teams.name.str.lower() == team2.lower())]\n",
    "    team2id = int(team2id.teamId.tolist()[0])\n",
    "    row['team1Id'] = team1id\n",
    "    row['team2Id'] = team2id\n",
    "    comb1 = total_summary_model_df[(total_summary_model_df.team == team1) & (total_summary_model_df.oppTeam == team2)]\n",
    "#     comb2 = diff_summary_model_df[(diff_summary_model_df.team == team2) & (diff_summary_model_df.oppTeam == team1)]\n",
    "    Odds1 = comb1.team100Odds.tolist()\n",
    "#     Odds2 = comb2.team200Odds.tolist()\n",
    "#     Odds2 = list(reversed(Odds2))\n",
    "    handi = comb1.handi.tolist()\n",
    "#     model = comb1.model.tolist()\n",
    "    model = [\"Lol-Pre-totalKills-UnderOver\"]*len(handi)\n",
    "    nm = [\"totalKills\"]*len(handi)\n",
    "#     Odds = [Odds1,Odds2]\n",
    "#     avg_Odds= np.mean(Odds,axis =0)\n",
    "    avg_Odds= Odds1\n",
    "    row = row.append([row]*(len(avg_Odds)-1), ignore_index=True)\n",
    "    row['marketName'] = model \n",
    "    row['displayName'] = nm\n",
    "    row['handicap'] = handi\n",
    "    row['team1Prob'] = avg_Odds\n",
    "    row['team2Prob'] = 1-row.team1Prob\n",
    "    row = row[(row.team1Prob >= 0.15) & (row.team1Prob <= 0.85)]\n",
    "    db_df = pd.concat([db_df, row], join= 'outer',ignore_index = True)\n",
    "    del row\n",
    "# row.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4a89b178",
   "metadata": {},
   "outputs": [],
   "source": [
    "kill_diff_df = db_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "78004347",
   "metadata": {},
   "outputs": [],
   "source": [
    "total_db_df = pd.concat([kill_diff_df, kill_total_df], join= 'outer',ignore_index = True)\n",
    "total_db_df = total_db_df.drop(['level_0'], axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "59fe585c",
   "metadata": {},
   "outputs": [],
   "source": [
    "total_db_df.to_sql('ds_kill_total_diff_odds_table', con=md_engine, if_exists='replace')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "csgo_venv",
   "language": "python",
   "name": "csgo_venv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
