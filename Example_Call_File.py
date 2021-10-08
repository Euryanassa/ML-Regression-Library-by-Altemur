from GraphMain import Linear_Regression, Loess_Regression, Polynomial_Regression
import numpy as np
import matplotlib.pyplot as plt

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
        140, 155, 167, 230, 133
    ]
)
Signal = Signal/max(Signal)
time_Array = list(range(len(Signal)))
time = np.array(time_Array)


plt.scatter(time,Signal)
plt.plot(Polynomial_Regression.Reg_Line(Signal,time,2),"m")
plt.plot(Loess_Regression.Reg_Line(Signal,time,45),"r")
plt.plot(Linear_Regression.Reg_Line(Signal,time),"y")

Legend_Polynomial = "Poly R. R2 = %.3f" % (Polynomial_Regression.R2_Score(Signal,time,2))
Legend_Linear = "Linear R. R2 = %.3f" % (Linear_Regression.R2_Score(Signal,time))
Legend_Loess = "Loess R. R2 = %.3f" % (Loess_Regression.R2_Score(Signal,time))
plt.legend([Legend_Polynomial,Legend_Loess,Legend_Linear])
plt.xlabel(Loess_Regression.MSE(Signal,time))
plt.show()
