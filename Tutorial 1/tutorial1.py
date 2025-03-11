import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

data=pd.read_csv('Advertising.csv') #reading dataset using panda

#define predictor variables
tv=data['TV']
radio=data['Radio']
newspaper=data['Newspaper']
y=data['Sales']

#function to calculate model statistics for each feature
def calc_values(x,y,feature):#x-predictors;y-target
    x=sm.add_constant(x)#adding intercept to predictor variable using statsmodels.api
    model=sm.OLS(y,x).fit()#creating an ols model to reduce errors between actual and predicted value;
    #.fits()train the model on given data
    prediction=model.predict(x) #predict sales using the ols model

    #rse
    mse=mean_squared_error(y,prediction)#computing mse to find rse using sklearn.metrics.mean_squared_error
    rse=np.sqrt(mse)#rse is sq root of mse using numpy for square root calculation

    #r2
    r2 = r2_score(y,prediction)#sklearn.metrics.r2_score

    #f-statistics
    f_stat = model.f_pvalue
    
    print(f"Results for {feature}:")
    print(f"Residual Standard Error: {rse}")
    print(f"R-squared: {r2}")
    print(f"F-statistic: {f_stat}\n")
    
    return rse,r2,f_stat
    
#call calac_value function to print result of impact of advertising using different medium
calc_values(tv,y,"TV")
calc_values(radio,y,"Radio")
calc_values(newspaper,y,"Newspaper")
