

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plastic= pd.read_csv("D:\\excelR\\Data science notes\\forecasting\\asgmnt\\PlasticSales.csv")
plastic.info()
plastic.describe()
plastic.head()
plastic.isnull().sum()
plastic.dtypes
plastic.shape

# Histogram 
plt.hist(plastic['Sales'], color='red');plt.title('Histogram of Sales')


month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
p=plastic['Month'][0]
p[0:3]
plastic['months']=0
        
        for i in range(60):
            p=plastic["Month"][i]
            plastic["months"][i]=p[0:3]
        month_dummies=pd.DataFrame(pd.get_dummies(plastic["months"]))
        plastic1=pd.concat([plastic,month_dummies],axis=1)

# Drop the "Month" column
plastic1.drop('Month', axis=1, inplace=True)

plastic1["t"]=np.arange(1,61)
plastic1["t_sqr"]=plastic1["t"]*plastic1["t"]
plastic1["log_sales"]=np.log(plastic1["Sales"])
plastic1.columns
plastic1.Sales.plot()

train= plastic1.head(45)
test=plastic1.tail(12)

import statsmodels.formula.api as smf 

###  Linear ##################################

linear_model= smf.ols("Sales~t", data=train).fit()
pred_linear=pd.Series(linear_model.predict(pd.DataFrame(test["t"])))
rmse_linear = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_linear))**2))
rmse_linear ## 254.948

### Exponential ##########################

exp_model= smf.ols("log_sales~t", data=train).fit()
pred_exp=pd.Series(exp_model.predict(pd.DataFrame(test["t"])))
rmse_exp = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_exp)))**2))
rmse_exp ## 262.11

#### Quadratic #######################

quad_model=smf.ols("Sales~t+t_sqr", data=train).fit()
pred_quad=pd.Series(quad_model.predict(test[["t","t_sqr"]]))
rmse_quad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_quad))**2))
rmse_quad ## 296.46

#### Additive Seasonality ########################

add_sea=smf.ols("Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov", data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea ## 240.15

#### Additive Seasonality Quadratic #############

add_sea_Quad = smf.ols('Sales~t+t_sqr+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_sqr']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad  ## 163.35

#### Multiplicative Seasonality ###################

mult_sea = smf.ols('log_sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_mult_sea = pd.Series(mult_sea.predict(test))
rmse_mult_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_mult_sea)))**2))
rmse_mult_sea ## 244.41

#### Multiplicative Additive Seasonality ###################

mult_add_sea = smf.ols('log_sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_mult_add_sea = pd.Series(mult_add_sea.predict(test))
rmse_mult_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_mult_add_sea)))**2))
rmse_mult_add_sea ## 135.11

####### Testing #############
data={"Model":pd.Series(["rmse_linear","rmse_exp","rmse_quad","rmse_add_sea","rmse_add_sea_quad","rmse_mult_sea","rmse_mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_exp,rmse_quad,rmse_add_sea,rmse_add_sea_quad,rmse_mult_sea,rmse_mult_add_sea])}
RMSE_table = pd.DataFrame(data)
RMSE_table

# so Multiplicative_additive_seasonality has the least RMSE value among the models.i.e 135.11