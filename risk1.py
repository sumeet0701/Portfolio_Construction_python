import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
import numpy as np
import seaborn as sns



def drawdown(return_series: pd.Series):
    """
    Takes a time series of assets return
    return a dataframe with columns for 
    the wealth index
    the previous peak, and 
    percentage drawdown

    """
    weath_index = 1000*(1+return_series).cumprod()
    previous_peak = weath_index.cummax()
    drawdown = (weath_index-previous_peak)/previous_peak
    return pd.DataFrame({
        "wealth": weath_index,
        "previous_peak": previous_peak,
        "drawdown": drawdown
    })

def get_ffme_return():
    """
    load the fame-french dataset for the return of the Top and Bottom Deciles marketcap
    """
    me_n = pd.read_csv(r"C:\Users\Sumeet Maheshwari\Desktop\data dump\VweKqLJfEemJ1w4LYV5qDg_2c089d97f24e49daa70b757b8337a76f_data (1)\data\Portfolios_Formed_on_ME_monthly_EW.csv")
    rst = me_n[["Lo 10","Hi 10" ]]
    rst.columns = ['small_cap','large_cap']
    rst = rst/100
    rst.index = pd.to_datetime(rst.index, format="%Y%m", errors='coerce').to_period("M")
    return rst

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv(r"C:\Users\Sumeet Maheshwari\Desktop\data dump\VweKqLJfEemJ1w4LYV5qDg_2c089d97f24e49daa70b757b8337a76f_data (1)\data\edhec-hedgefundindices.csv",
                      header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def plot_skewness(data = pd.DataFrame):
    #data = skewness(data)
    data = pd.Series(data.skew().sort_values())

    # Plot skewness values
    plt.figure(figsize=(10, 6))
    data.plot(kind='bar', color='skyblue')
    plt.title('Skewness of Data')
    plt.xlabel('Columns')
    plt.ylabel('Skewness')
    plt.show()

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def is_normal(r, level = 0.01):
    '''
    Applies the jarque Bera test to determine if serires is normal or not 
    test is applied at the 1% level by default
    returns True if the hypothesis of normal is accepted, fales ortherwise
    '''
    statistic , p_value = sts.jarque_bera(r)
    return p_value > level

def semideviation3(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    excess= r-r.mean()                                        # We demean the returns
    excess_negative = excess[excess<0]                        # We take only the returns below the mean
    excess_negative_square = excess_negative**2               # We square the demeaned returns below the mean
    n_negative = (excess<0).sum()                             # number of returns under the mean
    return (excess_negative_square.sum()/n_negative)**0.5     # semideviation



def var_historic(r, level = 5):
    """
    VaR historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic,level = level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise Exception("Expected r to be a Series or Dataframe")
    

def var_gaussian(r, level=5, modified = False):
    """
    compute Z score assuming it was Guassian 
    """
    z = sts.norm.ppf(level/100)
    if modified:
        # modifiy the Z score based on observed skewness and Krutosis
        s = skewness(r)
        k = kurtosis(r)
        z = z + (
            (z**2 -1)*s/6 +(z**3 -3*z)*(k-3)/24 - (2*z**3 - 5*z)*(s**2)/36
        )

    return abs(r.mean() + z*r.std(ddof =0))

def plot_var_comparision(var_list: list):
    comparsion = pd.concat(var_list, axis=1)
    comparsion.columns = ['Gaussian','cornish-Fisher','historic']
    comparsion.plot.bar(title ='EDHEC Hedge Fund Indices: VaR')


def cvar_historic(r,level=5):
    '''
    computes the Conditional VaR of Series or DataFrame
    '''
    if isinstance(r,pd.Series):
        is_beyond = r <= -var_historic(r, level = level)
        return -r[is_beyond].mean()
    elif isinstance(r,pd.DataFrame):
        return r.aggregate(cvar_historic, level = level)
    else:
        raise Exception("Expected r to be a Series or Dataframe")




