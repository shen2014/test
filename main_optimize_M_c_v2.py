


import pandas as pd
import numpy as np



df_a = pd.read_csv('stock_A.csv', index_col = None)

df_b = pd.read_csv('stock_B.csv', index_col = None)


def pos_change_greater_than_c(x, c):
    
    """
    find the index positions and signs where the price change (x) is greater than c
    """
    
    x1  = x[:-1]
    x2  = x[1:]
    
    diff = x2 - x1
    
    pos = np.where(np.abs(diff) > c)[0]
    
    sign = np.sign(diff[pos])
    
    return (pos, sign)


def close_price_and_time(df):
    
    """
    find the close price and time of a trade of stock b
    """
    x = df.loc[:, 'price'].values
    
    pos, sign = pos_change_greater_than_c(x, 0.0)
    
    close_price = x[pos[0] + 1]
    
    close_time = df.loc[pos[0] + 1, 'timestamp']
    
    return (close_price, close_time)



def  PL_optimize(M, c ):
    
    '''
    optimize M and c
    '''
    


     # M is the time in  milliseconds that data can be transferred btw exchanges
     # c is the threshold of the change of the price  of stock A
    
    
    price_a  = df_a.loc[:, 'price'].values
    
    
    where_change_a, sign = pos_change_greater_than_c(price_a, c)
    
    if len(where_change_a) == 0:
        
        print( 'when c = ', c, ', the strategy does not work')
        
        return (None, None, None, None)
    
    
    price_a_before_change = df_a.loc[where_change_a, 'price']
    
    price_a_after_change  = df_a.loc[1 + where_change_a, 'price']
    
    change_a              = price_a_after_change.values - price_a_before_change.values
    
    assert(np.all(np.abs(change_a) >= c))
    
    
    
    time_change_a         = df_a.loc[where_change_a, 'timestamp'].values
    
    n_change_a            = len(time_change_a)
    
    
    # the trading strategy begins with 1 unit stock of b
    
    first_trade_time     = time_change_a[0] + M   # first time to enter into the strategy
    
    enter_trade_time     = first_trade_time
    
    df_b_temp            = df_b.loc[df_b.loc[:, 'timestamp'] >=  enter_trade_time , :].copy()  # only for the purpose of making fast prototype. here it can be optimized
    df_b_temp.index      = range(len(df_b_temp))
    
    
    if len(df_b_temp) == 0:
        
        print( 'when M = ', M, ' and c = ', c, ' the strategy does not work')
        
        return (None, None, None, None)
    
    
    
    enter_price_b        = df_b_temp.loc[:, 'price'].values[0]   # price of b to enter at the trade
    
    first_enter_price    = enter_price_b
    
    amount               =   1 * sign[0]   #  amount of stock b to trade
    
    v_trade              = np.zeros(n_change_a + 1)
    
    
    v_trade[0]           =  amount * enter_price_b  # initial value of the trading strategy 
    
    
    if sign[0] < 0:
        
        v_trade[0]  = np.abs(v_trade[0])  # normalize the inital portfolio value to be positive for an easier interpretation
    
    close_price_b, close_time  =  close_price_and_time(df_b_temp)   # price and time to enter the opposite trade of stock b when the price changes
    
    v_trade[1]      = v_trade[0] +  ( close_price_b - enter_price_b) * amount
    
    
    # we put the rest of the trades in a loop
    
    for i in range(1, n_change_a):
    
        enter_trade_time     =  max(time_change_a[i] + M, close_time)
    
        
        df_b_temp             = df_b.loc[df_b.loc[:, 'timestamp'] >= enter_trade_time , :].copy()
        df_b_temp.index       = range(len(df_b_temp))
        
        if len(df_b_temp) == 0:
        
            print( 'the strategy stops')
        
            v_trade[i + 1:] = v_trade[i]
            
            break
        
        
        
        enter_price_b        = df_b_temp.loc[:, 'price'].values[0]   
        amount               =   1 * sign[i]   
        
        close_price_b, close_time  =  close_price_and_time(df_b_temp)
        
        v_trade[i+ 1]      = v_trade[i] +  ( close_price_b - enter_price_b) * amount
        
    
    pl = v_trade[-1] - v_trade[0]
    
    vol = np.std(v_trade)
    
    return (pl, vol, v_trade, first_enter_price)


max_M  = 60
M_grid = np.linspace(start = 40, stop = max_M, num = max_M  - 40 + 1)
M_grid  = M_grid/1000
max_c  = 0.020
c_grid = np.linspace(start = 0.015,  stop = max_c, num = int((max_c - 0.015)/0.001 + 1 ))



n_M_grid = len(M_grid)
n_c_grid = len(c_grid)

pl_scn = np.zeros((n_M_grid, n_c_grid))
vol_scn = np.zeros((n_M_grid, n_c_grid))   # std of the values of the trading strategy

for i in range(n_M_grid):
    
    M = M_grid[i]
    
    for j in range(n_c_grid):
        
        c = c_grid[j]
        

        
        pl, vol, v_trade, first_enter_price = PL_optimize(M, c )
        
        pl_scn[i, j]  = pl
        vol_scn[i, j]  = vol



pos_M = np.where(pl_scn[:, 0]> 0)[0][-1]

max_M = M_grid[pos_M]


print('given c = 0.015, the largest M for which the algorithm is still profitable is ', max_M )


for i in range(len(c_grid)):
    
    pos_M = np.where(pl_scn[:, i]> 0)[0][-1]

    max_M = M_grid[pos_M]

    print('given c = ', c_grid[i],  'the largest M for which the algorithm is still profitable is ', max_M )







