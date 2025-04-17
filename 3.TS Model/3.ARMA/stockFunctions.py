def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
    # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
        from numpy import array
    return array(X), array(y)

def conversion(y_train,stk_data):
    import pandas as pd
    Actual_y_train=pd.DataFrame(index=range(len(y_train)),columns=stk_data.columns)
    for i in range(len(y_train)):
        Actual_y_train.iloc[i]=y_train[i]
    return Actual_y_train

def graph(Actual,predicted,Actlabel,predlabel,title,Xlabel,ylabel):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(Actual, color = 'blue', label=Actlabel)
    plt.plot(predicted, color = 'green', label =predlabel)
    plt.title(title)
    plt.xlabel(Xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    
def rmsemape(y_Test,predicted_stock_price_test_ori):
    from sklearn.metrics import mean_squared_error
    print("RMSE-Testset:",mean_squared_error(y_Test,predicted_stock_price_test_ori))
    #print("RMSE-Trainset:",mean_squared_error(y_Train,predicted_stock_price_train_ori,squared=False))
    from sklearn.metrics import mean_absolute_percentage_error
    print("maPe-Testset:",mean_absolute_percentage_error(y_Test,predicted_stock_price_test_ori))
    #print("mape-Trainset:",mean_absolute_percentage_error(y_Train,predicted_stock_price_train_ori))

def rmsemape(y_test, y_pred):
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

    rmse = mean_squared_error(y_test, y_pred)

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

    print("RMSE-Testset:", rmse)
    print("maPe-Testset:", mape)

    return rmse, mape



def conversionSingle(y_train,stk_data):
    import pandas as pd
    Actual_y_train=pd.DataFrame(index=range(len(y_train)),columns=stk_data)
    for i in range(len(y_train)):
        Actual_y_train.iloc[i]=y_train[i]
    return Actual_y_train
















    