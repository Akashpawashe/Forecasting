
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

coco= pd.read_excel("D:\\excelR\\Data science notes\\forecasting\\asgmnt\\CocaCola_Sales_Rawdata.xlsx")
coco.info()
coco.describe()
coco.head()
coco.isnull().sum()
coco.dtypes
coco.shape
quarters= ['Q1','Q2','Q3','Q4']
p= coco['Quarter'][0]

p[0:2]

for i in range(42):
    p= coco["Quarter"][i]
    coco["Quarter"][i]=p[0:2]
qrtr_dummies=pd.DataFrame(pd.get_dummies(coco["Quarter"]))
cococola=pd.concat([coco,qrtr_dummies],axis=1)

cococola["t"]= np.arange(1,43)
cococola["t_sqr"]= cococola["t"]*cococola["t"]
cococola["log_sales"]= np.log(cococola["Sales"]) 

cococola.columns
cococola.Sales.plot()

train= cococola.head(35)
test=cococola.tail(7)

import statsmodels.formula.api as smf 

###  Linear ##################################

linear_model= smf.ols("Sales~t", data=train).fit()
pred_linear=pd.Series(linear_model.predict(pd.DataFrame(test["t"])))
rmse_linear = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_linear))**2))
rmse_linear ## 636.50

### Exponential ##########################

exp_model= smf.ols("log_sales~t", data=train).fit()
pred_exp=pd.Series(exp_model.predict(pd.DataFrame(test["t"])))
rmse_exp = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_exp)))**2))
rmse_exp ## 493.34

#### Quadratic #######################

quad_model=smf.ols("Sales~t+t_sqr", data=train).fit()
pred_quad=pd.Series(quad_model.predict(test[["t","t_sqr"]]))
rmse_quad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_quad))**2))
rmse_quad ## 511.411

#### Additive Seasonality ########################


add_sea = smf.ols('Sales~Q4+Q2+Q3',data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[['Q4','Q2','Q3']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea ## 1861.87

#### Additive Seasonality Quadratic #############

add_sea_Quad = smf.ols('Sales~t+t_sqr+Q4+Q2+Q3',data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[['Q4','Q2','Q3','t','t_sqr']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad #309.46

#### Multiplicative Seasonality ###################

mul_sea = smf.ols('log_sales~Q4+Q2+Q3',data = train).fit()
pred_mult_sea = pd.Series(mul_sea.predict(test[['Q4','Q2','Q3']]))
rmse_mult_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_mult_sea)))**2))
rmse_mult_sea#1938.93

#### Multiplicative Additive Seasonality ###################

mul_Add_sea = smf.ols('log_sales~t+Q4+Q2+Q3',data = train).fit()
pred_mult_add_sea = pd.Series(mul_Add_sea.predict(test))
rmse_mult_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_mult_add_sea)))**2))
rmse_mult_add_sea #339.18

####### Testing #############
data={"Model":pd.Series(["rmse_linear","rmse_exp","rmse_quad","rmse_add_sea","rmse_add_sea_quad","rmse_mult_sea","rmse_mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_exp,rmse_quad,rmse_add_sea,rmse_add_sea_quad,rmse_mult_sea,rmse_mult_add_sea])}
RMSE_table = pd.DataFrame(data)
RMSE_table

# so additive_seasonality_quadratic has the least RMSE value among the models.i.e  309.11