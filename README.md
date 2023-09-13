# Forecasting the Power output of a Wind Turbine

### Outline
This is the final project for the Seminar *"Machine Learning in Renewable Energy Systems"*. Datasets of a British and a Brazilian wind farm were provided. The primary goal involved predicting the wind power for a minimum of one turbine from each farm for the subsequent time step, hour, and day. Additionally, participants were tasked with the selection of relevant features, either through manual or automated means. This process, along with all undertaken steps, was to be documented and explained.

## How to use this GitHub repository
1. Clone this repository
2. Set up a new virtual environment with the given environment.yml file
```
conda env create -f environment.yml
```
3. Download the [British data](https://zenodo.org/record/5841834#.ZEajKXbP2BQ) and the [Brazilian data](https://zenodo.org/record/1475197#.ZD6iMxXP2WC) and put them into their corresponding data folder
4. Make sure that your folder structure looks like this: 
```
├───data
│   ├───British
│   │   ├───Kelmarsh_SCADA_2016_3082
│   │   ├───Kelmarsh_SCADA_2017_3083
│   │   ├───Kelmarsh_SCADA_2018_3084
│   │   ├───Kelmarsh_SCADA_2019_3085
│   │   ├───Kelmarsh_SCADA_2020_3086
│   │   └───Kelmarsh_SCADA_2021_3087
│   └───Brazilian
│       └───UEBB_v1.nc
├───notebooks
│   ├───DataInspection.ipynb
│   ├───MovingAverage.ipynb
│   ├───RegressionVariants.ipynb
│   ├───XGBoost.ipynb
│   └───Evaluation.ipynb
├───src
│   ├───DataHandling
│   ├───Models
│   └───utils.py
├───.gitignore
├───environment.yml
└───README.md
```
5. You should be now able to run all the notebooks. For every examined model there is a corresponding notebook. Additionally, there is a notebook for the data inspection and one for the evaluation of the models. Please make sure that you select the correct kernel, i.e. the one from the virtual environment you created in step 2. We recommend to run the notebooks in the following order:
    1. DataInspection.ipynb
    2. MovingAverage.ipynb
    3. RegressionVariants.ipynb
    4. XGBoost.ipynb
    5. Evaluation.ipynb

Please run the cells of the corresponding order since we overwrite some variables in rare cases.

<!-- 6. Furthermore, there is a command line interface (CLI) available. You can run it by typing
```
conda activate res_env
python src/main.py --help
``` -->

## Models
In the following, the models are briefly described. For more details, please refer to the corresponding notebooks.

### Moving Average
The Moving Average Model is the simplest model implemented in this project. It is based on the assumption that the next value is the average of the last n values. Additionally, the model can be extended with a discount factor that weights the values differently. Therefore, it is really similar to the discount used in calculating future rewards in reinforcement learning. The formula for the Moving Average Model is given by:

$$\hat{y}_ {t+1} = \tfrac{1}{n} \sum_ {i=0}^{n-1} \gamma^i \cdot y_ {t-i}$$

where $\gamma$ is the discount factor and $n$ is the number of values used for the average which we call the window_size of the model. Both parameters can be set by the user. The model is implemented in the file `src/Models/ma.py`. 


### Regression Variants

1. Ridge Regression

Ridge Regression solves the following optimization problem: 

$$\min_{\omega} ||X\omega - y||_2^2 + \alpha ||\omega||_2^2$$

where $X$ is the feature matrix, $y$ is the target vector, $\omega$ is the weight vector and $\alpha$ is the regularization parameter. This can be done in closed form:

$$\omega = (X^TX + \alpha I)^{-1}X^Ty$$

2. Kernel Ridge Regression

Kernel Ridge Regression replaces the feature matrix $X$ with the kernel matrix $K$ and solves the following optimization problem:

$$\min_{\omega} ||K\omega - y||_2^2 + \alpha ||\omega||_2^2$$

where $K$ is the kernel matrix, $y$ is the target vector, $\omega$ is the weight vector and $\alpha$ is the regularization parameter. This can be also done in closed form:

$$\omega = (K + \alpha I)^{-1}y$$

Since the dimensions of the kernel matrix are too large, we use the Nyström approximation to approximate the kernel matrix.



### XGBoost

## Results

## Conclusion
--> short summary of what I did
--> best model is... because
--> Possible steps to improve results

### References
--> add references
