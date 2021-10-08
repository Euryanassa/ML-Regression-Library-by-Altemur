import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import pandas as pd
import math
from scipy import signal


class Built_in_Datas():   
    def datas():
        """
        This is built-in data. For instance, you can call as:
        Signal , time = Built_in_Datas.datas()
        """
        Signal = np.array(
            [
                100, 11, 101, 99, 105,
                110, 110, 125, 115, 120,
                120, 12, 127, 130, 133,
                136, 140, 145, 147, 150,
                170, 10, 170, 510, 510,
                510, 155, 158, 140, 162,
                165, 169, 175, 160, 177,
                122, 159, 176, 130, 197,
                10, 0, 0, 10, 0,
                170, 10, 170, 510, 510,
                130, 110, 125, 115, 120,
                140, 155, 167, 230, 133,
            ]
        )
        Signal = Signal/max(Signal)
        time_Array = list(range(len(Signal)))
        time = np.array(time_Array)
        return Signal,time

class Linear_Regression():   
    def _model_data(Signal,time,rule = None):
        train = pd.DataFrame(list(zip(time,Signal)),columns=['time','Signal'])
        Regr=linear_model.LinearRegression()


        train_x = np.asanyarray(train[['time']])
        train_y = np.asanyarray(train[['Signal']])

        Regr.fit(train_x,train_y)
        test_x = np.asanyarray(train[['time']])

        test_y_ = Regr.predict(test_x)
        if   rule == 1:  return train_x , Regr
        elif rule == 2:  return train
        else          :  return Signal,test_y_

    def Reg_Line(Signal,time):
        """
        This function creates Linear Regression line between two dimensional data.
        CALL: Linear_Regression.Reg_Line(Data1,Data2)
        Use matplotlib.pyplot.plot() function recommended for observation.
        
        """
        train_x, Regr = Linear_Regression._model_data(Signal,time,1)
        LineFunc = Regr.coef_[0][0]*train_x + Regr.intercept_[0]
        return LineFunc
    
    def R2_Score(Signal,time):
        """
        This function gives float output of R Square score from data.
        CALL: Linear_Regression.R2_Score(Data1,Data2)
        
        """
        Signal,test_y_ = Linear_Regression._model_data(Signal,time)
        R2_Handled = 1-np.sum(np.mean((Signal - test_y_)**2)/np.mean(Signal-np.mean(test_y_)**2))
        return R2_Handled
    
    def MSE(Signal,time):
        """
        This function gives float output of Mean Square Error from data.
        CALL: Linear_Regression.MSE(Data1,Data2)
        
        """
        Signal,test_y_ = Linear_Regression._model_data(Signal,time)
        MSE_Handled = np.mean((Signal - test_y_) ** 2)
        return MSE_Handled
  
class Polynomial_Regression():
    
    def _model_data(Signal,time,rule = None,degree=None):
        
        train = pd.DataFrame(list(zip(time,Signal)),columns=['time','Signal'])

        train_x = np.asanyarray(train[['time']])
        train_y = np.asanyarray(train[['Signal']])

        poly = PolynomialFeatures(degree=degree)
        train_x_poly = poly.fit_transform(train_x)

        model = linear_model.LinearRegression()
        model.fit(train_x_poly, train_y)
        test_y_ = model.predict(train_x_poly)
        
        if   rule == 1:  return train_x , model, degree
        elif rule == 2:  return train
        else          :  return Signal,test_y_

    def Reg_Line(Signal,time,degree):
        """
        This function creates Polynomial Regression line between two dimensional data witihin given degree.
        CALL: Polynomial_Regression.Reg_Line(Data1,Data2,degree)
        Use matplotlib.pyplot.plot() function recommended for observation.
        
        """
        _, model,degree= Polynomial_Regression._model_data(Signal,time ,1,degree)
        Signal,_ = Polynomial_Regression._model_data(Signal,time,None,degree)
        XX = np.arange(0.0, len(Signal), 1)
        LineFunc = Polynomial_Regression.FormulaWriter(degree, model, XX)
        #LineFunc = model.intercept_[0]+ model.coef_[0][1]*XX+ model.coef_[0][2]*np.power(XX, 2)
        return LineFunc
    
    def FormulaWriter(i,model,XX):
        result_sum = model.intercept_[0]
        for j in range(i):
            result_sum += model.coef_[0][j+1]*np.power(XX,j+1)  
        return result_sum

    def R2_Score(Signal,time,degree):
        """
        This function gives float output of R Square score from data.
        CALL: Polynomial_Regression.R2_Score(Data1,Data2)
        
        """
        Signal,test_y_ = Polynomial_Regression._model_data(Signal,time ,None,degree)
        R2_Handled = 1-np.sum(np.mean((Signal - test_y_)**2)/np.mean(Signal-np.mean(test_y_)**2))
        return R2_Handled
    
    def MSE(Signal,time,degree):
        """
        This function gives float output of Mean Square Error from data.
        CALL: Polynomial_Regression.MSE(Data1,Data2,degree)
        
        """
        Signal,test_y_ = Polynomial_Regression._model_data(Signal,time ,None,degree)
        MSE_Handled = np.mean((Signal - test_y_) ** 2)
        return MSE_Handled

class Loess_Regression():
    def _plotter(Signal,time,k):
        xx, yy = Loess_Regression._model_data(Signal,time)
        Signal,time = Loess_Regression._model_data(Signal,time,1)
        loess = Loess(time,Signal)     
        for i in range(len(xx)):
            yy[i] = loess.estimate(xx[i], window=k)
        return yy
            
    def _model_data(Signal,time,rule = None):
        xx = np.arange(start=0.0, stop=20.0 * math.pi, step=4.0 * math.pi / 100.0, dtype=float)
        yy = np.zeros_like(xx) 
        if rule == 1: return Signal,time
        else:         return xx,yy

    def Reg_Line(Signal,time,RegLen=30):
        """
        This function creates Loess Regression line between two dimensional data within given window.
        CALL: PLoess_Regression.Reg_Line(Data1,Data2,Regression_Window_for_per_estimation)
        Use matplotlib.pyplot.plot() function recommended for observation.
        
        """
        LineFunc =  signal.resample(Loess_Regression._plotter(Signal,time,RegLen), len(Signal))
        return LineFunc

    def R2_Score(Signal, time, RegLen=30):
        """
        This function gives float output of R Square score from data.
        CALL: Loess_Regression.R2_Score(Data1,Data2)
        
        """
        Y_pred = Loess_Regression.Reg_Line(Signal, time, RegLen)
        Signal,_ = Loess_Regression._model_data(Signal,time,1)
        R2_Handled = 1-np.sum(np.mean((Signal - Y_pred)**2)/np.mean(Signal-np.mean(Y_pred)**2))
        return R2_Handled
    
    def MSE(Signal, time, RegLen=30):
        """
        This function gives float output of Mean Square Error from data.
        CALL: Polynomial_Regression.MSE(Data1,Data2,degree)
        
        """
        Y_pred = Loess_Regression.Reg_Line(Signal, time, RegLen)
        Signal,_ = Loess_Regression._model_data(Signal,time,1)
        MSE_Handled = np.mean((Signal - Y_pred) ** 2)
        return MSE_Handled
    
    
def tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y

class Loess(object):

    @staticmethod
    def normalize_array(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val), min_val, max_val

    def __init__(self, xx, yy, degree=1):
        self.n_xx, self.min_xx, self.max_xx = self.normalize_array(xx)
        self.n_yy, self.min_yy, self.max_yy = self.normalize_array(yy)
        self.degree = degree

    @staticmethod
    def get_min_range(distances, window):
        min_idx = np.argmin(distances)
        n = len(distances)
        if min_idx == 0:
            return np.arange(0, window)
        if min_idx == n-1:
            return np.arange(n - window, n)

        min_range = [min_idx]
        while len(min_range) < window:
            i0 = min_range[0]
            i1 = min_range[-1]
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n-1:
                min_range.insert(0, i0 - 1)
            elif distances[i0-1] < distances[i1+1]:
                min_range.insert(0, i0 - 1)
            else:
                min_range.append(i1 + 1)
        return np.array(min_range)

    @staticmethod
    def get_weights(distances, min_range):
        max_distance = np.max(distances[min_range])
        weights = tricubic(distances[min_range] / max_distance)
        return weights

    def normalize_x(self, value):
        return (value - self.min_xx) / (self.max_xx - self.min_xx)

    def denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy

    def estimate(self, x, window, use_matrix=False, degree=1):
        n_x = self.normalize_x(x)
        distances = np.abs(self.n_xx - n_x)
        min_range = self.get_min_range(distances, window)
        weights = self.get_weights(distances, min_range)

        if use_matrix or degree > 1:
            wm = np.multiply(np.eye(window), weights)
            xm = np.ones((window, degree + 1))

            xp = np.array([[math.pow(n_x, p)] for p in range(degree + 1)])
            for i in range(1, degree + 1):
                xm[:, i] = np.power(self.n_xx[min_range], i)

            ym = self.n_yy[min_range]
            xmt_wm = np.transpose(xm) @ wm
            beta = np.linalg.pinv(xmt_wm @ xm) @ xmt_wm @ ym
            y = (beta @ xp)[0]
        else:
            xx = self.n_xx[min_range]
            yy = self.n_yy[min_range]
            sum_weight = np.sum(weights)
            sum_weight_x = np.dot(xx, weights)
            sum_weight_y = np.dot(yy, weights)
            sum_weight_x2 = np.dot(np.multiply(xx, xx), weights)
            sum_weight_xy = np.dot(np.multiply(xx, yy), weights)

            mean_x = sum_weight_x / sum_weight
            mean_y = sum_weight_y / sum_weight

            b = (sum_weight_xy - mean_x * mean_y * sum_weight) / \
                (sum_weight_x2 - mean_x * mean_x * sum_weight)
            a = mean_y - b * mean_x
            y = a + b * n_x
        return self.denormalize_y(y)
