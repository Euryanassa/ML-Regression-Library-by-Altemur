# ML-Regression-Library-by-Altemur
### This library creates three types of regression line from given data.

![autoreg](https://user-images.githubusercontent.com/67932543/136551737-f6ee9c49-9981-417e-a89f-f9c17b6d88e6.PNG)

![0000000000000000000](https://user-images.githubusercontent.com/67932543/136550528-1b93e96e-4bfe-4a63-aad6-4ae131f83a3f.png)
Example Data Calls:

	Signal,time = Built_in_Datas.datas()

Linear Regression Commands:
	
	Line Creator:
	Linear_Regression.Reg_Line(Data1,Data2)
	
	R Square Score:
	Linear_Regression.R2_Score(Data1,Data2)

	Mean Square Error:
	Linear_Regression.MSE(Data1,Data2)

Polynomial Regression Commands:

	Line Creator:
	Polynomial_Regression.Reg_Line(Data1,Data2,degree)

	R Square Score:
	Polynomial_Regression.R2_Score(Data1,Data2,degree)

	Mean Square Error:
	Polynomial_Regression.MSE(Data1,Data2,degree)

Loess Regression Commands:

	Line Creator:
	Loess_Regression.Reg_Line(Data1,Data2,Regression_Window_for_per_estimation)

	R Square Score:
	Loess_Regression.R2_Score(Data1,Data2,Regression_Window_for_per_estimation)

	Mean Square Error:
	Loess_Regression.MSE(Data1,Data2,Regression_Window_for_per_estimation)

Regression Library by Altemur
