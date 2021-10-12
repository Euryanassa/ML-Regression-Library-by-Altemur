from AutoRegPilot import Linear_Regression, Loess_Regression, Polynomial_Regression
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

"""
Line_Plotter.plot(plotlist="LR,PR,CF,LSS",title="Tablom")

Loess_Regression.Plotter()
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
        140, 155, 167, 230, 133,510, 155, 158, 140, 162,
        165, 169, 175, 160, 177,
        122, 159, 176, 130, 197,510, 155, 158, 140, 162,
        165, 169, 175, 160, 177,
        122, 159, 176, 130, 197,
    ]
)
Signal = Signal/max(Signal)
time_Array = list(range(len(Signal)))
time = np.array(time_Array)

degree = 5
RegLen = 52

t0_1= timer()
Poly1 = Polynomial_Regression.Reg_Line(Signal,time,degree = degree)
t1 = timer() - t0_1

t0_2 = timer()
Loess1 = Loess_Regression.Reg_Line(Signal,time,RegLen = RegLen)
t2 = timer() - t0_2

t0_3 = timer()
Linear1 = Linear_Regression.Reg_Line(Signal,time)
t3 = timer() - t0_3

Elapsed_Time = "Elaspsed Time of;  Poly: %.3f, Loess: %.3f, Linear: %.3f "%(t1,t2,t3)

plt.scatter(time,Signal)
plt.plot(Poly1,"m")
plt.plot(Loess1,"r")
plt.plot(Linear1,"y")

Legend_Polynomial = "Poly R. R2 = %.3f" % (Polynomial_Regression.R2_Score(Signal,time,degree = degree ))
Legend_Linear = "Linear R. R2 = %.3f" % (Linear_Regression.R2_Score(Signal,time))
Legend_Loess = "Loess R. R2 = %.3f" % (Loess_Regression.R2_Score(Signal,time,RegLen = RegLen))
plt.legend([Legend_Polynomial,Legend_Loess,Legend_Linear])
plt.xlabel(Elapsed_Time)
plt.ylabel("Signal")
plt.show()
