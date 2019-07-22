# -- coding: utf-8 --
from flask import Flask,request,make_response
import json #, os
import os
import os
import configparser
import pandas as pd


class eventi:
  def __init__(self, request):
    self.request = request   
    self.result = request.get("queryResult")
    self.intentName = request.get("queryResult").get("intent").get("displayName")
    self.city=request.get("queryResult").get("parameters").get("luogo")
    self.company=request.get("queryResult").get("parameters").get("aziende")
    self.luogo=request.get("queryResult").get("parameters").get("geo-city")
    conf = configparser.ConfigParser()
    conf.read(os.path.dirname(os.getcwd())+'/configurations/configurations.ini')
    self.path_ev=os.path.dirname(os.getcwd())+conf.get("INPUT_FILES","FORM_EVENTI")
    self.path_desc=os.path.dirname(os.getcwd())+conf.get("INPUT_FILES","company_desc")

  def contextChatbot(self):
    speech = ''
    print("IntentName: " + self.intentName)
    if self.intentName == 'Eventi-informazioni-aziende':
            speech= self.retrive_next_event_company() 
    elif self.intentName == 'Eventi-Positivo':
            speech= self.retrive_next_event_info()
    elif self.intentName == 'Eventi':
            speech= self.retrive_next_event()
    elif self.intentName == 'aziende':
            speech= self.retrive_company_info()
    elif self.intentName == 'Citta':
            speech= self.retrive_other_event()
            
            
    return speech

  def retrive_next_event(self):
    
    df = pd.read_csv(self.path_ev, delimiter=';')
    
    df1=df.sort_values(by='Data')
  
    
    event_name=df1.iloc[0,0]
  
    date=df1.iloc[0,4]
    date = str(date)[:10]
    place=df1.iloc[0,1]
    output= 'Riguardo al prossimo evento organizzato da Employerland abbiamo il ' + event_name + ' ' + ' il ' +  date + ' a ' + place + '. Potrebbe interessarti?'
    data = {}
    data['fulfillmentText'] = output
    json_data = json.dumps(data,ensure_ascii=False)
    return json_data


  def retrive_other_event(self):
    df = pd.read_csv(self.path_ev, delimiter=';')
    luogo = df.loc[:,'Citta']
    luogo = luogo.str.lower()
    imp_city= self.luogo.lower()
    luogo= luogo.str.contains(imp_city)
    df1=df.sort_values(by='Data').iloc[1::]
    df2=df1[luogo]
    event_name=df2.sort_values(by=['Data'], ascending=True).iloc[0,0]
    event_description=df2.sort_values(by=['Data'], ascending=True).iloc[0,6]
    event_data=df2.sort_values(by=['Data'], ascending=True).iloc[0,4]
    event_data = str(event_data)
    str_out= 'Abbiamo anche questo evento' + event_name +' '+ event_description +' il ' + event_data + '. Pensi potrebbe fare al caso tuo?'
    data = {}
    data['fulfillmentText'] = str_out
    json_data = json.dumps(data,ensure_ascii=False)
    return json_data
    
  def retrive_next_event_info(self):
    df = pd.read_csv(self.path_ev, delimiter=';')
    df1=df.sort_values(by='Data')   
    event_area=df1.iloc[0,7]
    date=df1.iloc[0,3]
    date = str(date)[:10]
    output= 'Questo evento riguarda i settori ' + event_area + '. Sei interessato alle aziende che ne prenderanno parte?'
    data = {}
    data['fulfillmentText'] = output
    json_data = json.dumps(data,ensure_ascii=False)
    return json_data

  def retrive_next_event_company(self):
    df = pd.read_csv(self.path_ev, delimiter=';')
    df1=df.sort_values(by='data')
    company=df1.iloc[0,8]
    company = company.encode(encoding ='UTF-8', errors='strict')
    output= 'Le aziende che parteciperanno a questo evento sono ' + company + '. Quale di queste aziende potrebbe interessarti?'
    data = {}
    data['fulfillmentText'] = output
    json_data = json.dumps(data,ensure_ascii=False)
    return json_data

  def retrive_company_info(self):
    df = pd.read_csv(self.path_desc, delimiter=';')
    company = df.loc[:,'name']
    company = company.str.lower()
    imp_cmp= self.company.lower()
    company= company.str.contains(imp_cmp) 
    df1=df[company]
    
    if df1.shape[0]>0:
        company_info= df1.iloc[0,1]
        str_out= 'Ecco alcune info su questa azienda:' + company_info +  ' Sulla nostra App troverai delle challenge dedicate a ' + self.company + ', usale per arrivare preparato e farti notare in anticipo da loro. Quali altri aziende ti interessano?'
        
    else:
        str_out=  'Su ' + imp_cmp +' Non ho altre informazioni al momento, mi dispiace! A quali altri potresti essere interessato?'
    data = {}     
    print(str_out) 
    data['fulfillmentText'] = str_out
    json_data = json.dumps(data,ensure_ascii=False)
    return json_data
  #processing the request from dialogflow

  