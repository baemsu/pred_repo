
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import joblib


def user_input_features() :
  KT_10 = st.sidebar.number_input("10시 KT")
  total_10 =st.sidebar.number_input("10시 종합")
  KT_11 = st.sidebar.number_input("11시 KT")
  total_11 =st.sidebar.number_input("11시 종합")
  KT_12 = st.sidebar.number_input("12시 KT")
  total_12 =st.sidebar.number_input("12시 종합")
  KT_13 = st.sidebar.number_input("13시 KT")
  total_13 =st.sidebar.number_input("13시 종합")
  KT_14 = st.sidebar.number_input("14시 KT")
  total_14 =st.sidebar.number_input("14시 종합")
  KT_15 = st.sidebar.number_input("15시 KT")
  total_15 =st.sidebar.number_input("15시 종합")
  KT_16 = st.sidebar.number_input("16시 KT")
  total_16 =st.sidebar.number_input("16시 종합")
  KT_17 = st.sidebar.number_input("17시 KT")
  total_17 =st.sidebar.number_input("17시 종합")

# ['KT_10','KT_11','KT_12','KT_13','KT_14','KT_15','KT_16','total_10','total_16']]
  data_KT_16 = {'KT_10' : KT_10,
          'KT_11' : KT_11,
          'KT_12' : KT_12,
          'KT_13' : KT_13,
          'KT_14' : KT_14,
          'KT_15' : KT_15,
          'KT_16' : KT_16,
          'total_10' : total_10,
          'total_16' : total_16,
          }
# [['KT_12','KT_13','KT_14','KT_15','KT_16','KT_17','total_12','total_17']]
  data_KT_17 = {'KT_12' : KT_12,
          'KT_13' : KT_13,
          'KT_14' : KT_14,
          'KT_15' : KT_15,
          'KT_16' : KT_16,
          'KT_17' : KT_17,
          'total_12' : total_12,
          'total_17' : total_17,
          }

# [['total_10','total_11','total_12','total_13','total_14','total_15','total_16','KT_12']]
  data_total_16 = {'total_10' : total_10,
          'total_11' : total_11,
          'total_12' : total_12,
          'total_13' : total_13,
          'total_14' : total_14,
          'total_15' : total_15,
          'total_16' : total_16,
          'KT_11' : KT_11,
          'KT_12' : KT_12,
          }
#['KT_12','total_12','total_13','total_14','total_15','total_16','total_17','KT_17']]
  data_total_17 = {'KT_12' : KT_12,
          'total_12' : total_12,
          'total_13' : total_13,
          'total_14' : total_14,
          'total_15' : total_15,
          'total_16' : total_16,
          'total_17' : total_17,          
          'KT_17' : KT_17,
          }

  data_chart_KT_16 = {"data":[KT_10,KT_11,KT_12,KT_13,KT_14,KT_15,KT_16]}

  features_KT_16 = pd.DataFrame(data_KT_16, index=[0])
  features_KT_17 = pd.DataFrame(data_KT_17, index=[0])
  features_total_16 = pd.DataFrame(data_total_16, index=[0])
  features_total_17 = pd.DataFrame(data_total_17, index=[0])
  features_chart_KT_16 = pd.DataFrame(data_chart_KT_16)

  return features_KT_16, features_KT_17, features_total_16, features_total_17, features_chart_KT_16, KT_16, total_16



#def main():

st.header("""
# MVNO 예측 솔루션
""")

predict_time = st.radio("어떤 시간에 예측?",('오후4시', '오후5시'))

if predict_time == '오후4시':
  st.subheader('오후 4시까지 데이터를 입력하고, 이를 가지고 예측해요.')
  st.sidebar.header('User Input Parameters')
  df_KT_16, df_KT_17, df_total_16, df_total_17, df_chart_KT_16, KT_16, total_16 = user_input_features()
  st.subheader("16시에 입력된 KT에서 KT로 온 수량")
  st.write(df_KT_16)
  st.subheader("16시에 입력된 종합수량")
  st.write(df_total_16)
  loaded_model = joblib.load("./regression_KT_16.pkl")
  predict_ = loaded_model.predict(df_KT_16)
  st.subheader("16시에 예측한 마감때의 KT 수량")
  st.write(predict_)
  loaded_model2 = joblib.load("./regression_total_16.pkl")
  predict_2 = loaded_model2.predict(df_total_16)
  st.subheader("16시에 예측된 마감때의 종합 수량")
  st.write(predict_2)
  st.subheader("예상되는 KT에서 KT MNP비율")
  st.write(predict_/predict_2)
#   st.line_chart(df_chart_KT_16)
  st.subheader("최종 추가 가능한 수량 최대치")

#  st.write(predict_2)
#  st.write(KT_16)
#  st.write(total_16)
  a=0.49906244085726026
  b=-14.99416837030202

  optim_no = int(predict_2) - int(total_16)

  for i in range(int(predict_2) - int(total_16)):
    MNP_percent_est = (KT_16 + a*(i+1) + b  )/(total_16 + (i+1))
    #print(i+1, round(MNP_percent_est,2))
    if(round(MNP_percent_est,2)>=0.45):
      #print("45%를 넘지 않을 최대추가갯수", i)
      optim_no = i

  st.write(optim_no)

else:
  st.subheader('오후 5시까지 데이터를 입력하고, 이를 가지고 예측해요.')
  st.sidebar.header('User Input Parameters')
  df_KT_16, df_KT_17, df_total_16, df_total_17, df_chart_KT_16, KT_16, total_16 = user_input_features()
  st.subheader("17시에 입력된 KT에서 KT로 온 수량")
  st.write(df_KT_17)
  st.subheader("17시에 입력된 종합수량")
  st.write(df_total_17)
  loaded_model = joblib.load("./regression_KT_17.pkl")
  predict_ = loaded_model.predict(df_KT_17)
  st.subheader("17시에 예측한 마감때의 KT 수량")
  st.write(predict_)
  loaded_model2 = joblib.load("./regression_total_17.pkl")
  predict_2 = loaded_model2.predict(df_total_17)
  st.subheader("17시에 예측된 마감때의 종합 수량")
  st.write(predict_2)
  st.subheader("예상되는 KT에서 KT MNP비율")
  st.write(predict_/predict_2)


#if __name__ == '__main__':
#	main()