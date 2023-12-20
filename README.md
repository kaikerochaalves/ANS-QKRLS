# ANS-QKRLS (adaptive normalized sparse quantized kernel recursive least squares)

The adaptive normalized sparse quantized kernel recursive least squares (ANS-QKRLS) is a model proposed by Han et al. [1].

- [ANS-QKRLS](https://github.com/kaikerochaalves/ANS-QKRLS/blob/fa0a6efc02c7fa3cf8f877073ce2f061f82a49be/Model/ANS_QKRLS.py) is the ANS-QKRLS model.

- [GridSearch_AllDatasets](https://github.com/kaikerochaalves/ANS-QKRLS/blob/fa0a6efc02c7fa3cf8f877073ce2f061f82a49be/GridSearch_AllDatasets.py) is the file to perform a grid search for all datasets and store the best hyper-parameters.

- [Runtime_AllDatasets](https://github.com/kaikerochaalves/ANS-QKRLS/blob/fa0a6efc02c7fa3cf8f877073ce2f061f82a49be/Runtime_AllDatasets.py) perform 30 simulations for each dataset and compute the mean runtime and the standard deviation.

- [MackeyGlass](https://github.com/kaikerochaalves/ANS-QKRLS/blob/fa0a6efc02c7fa3cf8f877073ce2f061f82a49be/MackeyGlass.py) is the script to prepare the Mackey-Glass time series, perform simulations, compute the results and plot the graphics. 

- [Nonlinear](https://github.com/kaikerochaalves/ANS-QKRLS/blob/fa0a6efc02c7fa3cf8f877073ce2f061f82a49be/Nonlinear.py) is the script to prepare the nonlinear dynamic system identification time series, perform simulations, compute the results and plot the graphics.

- [LorenzAttractor](https://github.com/kaikerochaalves/ANS-QKRLS/blob/fa0a6efc02c7fa3cf8f877073ce2f061f82a49be/LorenzAttractor.py) is the script to prepare the Lorenz Attractor time series, perform simulations, compute the results and plot the graphics. 

[1] M. Han, S. Zhang, M. Xu, T. Qiu, N. Wang, Multivariate chaotic time series online prediction based on improved kernel recursive least squares algorithm, IEEE Transactions on Cybernetics 49 (4) (2018) 1160–1172.
doi:https://doi.org/10.1109/TCYB.2018.2789686.
