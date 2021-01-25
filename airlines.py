

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

air= pd.read_excel("D:\\excelR\\Data science notes\\forecasting\\asgmnt\\Airlines+Data.xlsx")
air.info()
air.describe()
air.head()
air.isnull().sum()
air.dtypes
air.shape

# Histogram 
plt.hist(air['Passengers'], color='red');plt.title('Histogram of Passengers')


month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
air['months'] = air['Month'].apply(lambda x: x.strftime('%B-%Y'))
air=air.drop("Month", axis=1)
air=air[["months","Passengers"]]

p= air['months'][0]
p[0:3]
air['month']=0

for i in range(96):
    p= air["months"][i]
    air["month"][i]=p[0:3]
    
month_dummies=pd.DataFrame(pd.get_dummies(air["month"]))
air1=pd.concat([air,month_dummies],axis=1)

# Drop the "Month" column
air1.drop('Month', axis=1, inplace=True)

air1["t"] = np.arange(1,97)
air1["t_sqr"] = air1["t"]*air1["t"]
air1.columns
air1["log_pas"] = np.log(air1["Passengers"])
air1.Passengers.plot()
Train = air1.head(84)
Test = air1.tail(12)

import statsmodels.formula.api as smf 
####################### L I N E A R ##########################

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear #53.19

##################### Exponential ##############################

Exp = smf.ols('log_pas~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp #46.05

#################### Quadratic ###############################

Quad = smf.ols('Passengers~t+t_sqr',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_sqr"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad #48.05

################### Additive seasonality ########################

add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea #132.81

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Passengers~t+t_sqr+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_sqr']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad #26.36

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_pas~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea #140.06

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_pas~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea #10.51

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
RMSE_table =pd.DataFrame(data)
RMSE_table

#### so Multiplicative additive seasonality has the least value among the models,i.e.10.51 