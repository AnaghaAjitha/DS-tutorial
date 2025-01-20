import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

data=pd.read_csv('Advertising.csv')

tv=data['TV']
radio=data['Radio']
newspaper=data['Newspaper']
y=data['Sales']


def calc_values(x,y,feature):#x-predictors;y-target
    x=sm.add_constant(x)
    model=sm.OLS(y,x).fit()
    prediction=model.predict(x)

    #rse
    mse=mean_squared_error(y,prediction)
    rse=np.sqrt(mse)

    #r2
    r2 = r2_score(y,prediction)

    #f-statistics
    f_stat = model.f_pvalue
    
    print(f"Results for {feature}:")
    print(f"Residual Standard Error: {rse}")
    print(f"R-squared: {r2}")
    print(f"F-statistic: {f_stat}\n")
    
    return rse,r2,f_stat

calc_values(tv,y,"TV")
calc_values(radio,y,"Radio")
calc_values(newspaper,y,"Newspaper")