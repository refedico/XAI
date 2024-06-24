# XAI with SHAP and LIME

## Overview
This project demonstrates the use of Explainable AI (XAI) techniques, specifically SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations), to explain the predictions of a machine learning model trained on the California and Boston Housing Prices dataset. The goal is to provide insights into how the model makes its predictions and the contribution of each feature to the final output.

## Dataset
The dataset used in this project is the California Housing Prices dataset and the Boston Housing Prices. The target variable is the median house value for California districts, and the features include various socio-economic factors.

## Model
A Random Forest Regressor is trained on the dataset to predict the housing prices. The performance of the model is evaluated using the Mean Squared Error (MSE) metric.

## Explainable AI Techniques
### SHAP
SHAP values are computed to understand the contribution of each feature to the model's predictions. The project includes various visualizations to interpret these values:
- Bar Plot
- Summary Plot
- Beeswarm Plot
- Violin Plot
- Waterfall Plot
- Force Plot

### LIME
LIME is used to generate local surrogate models that approximate the predictions of the black-box model locally. The explanations provided by LIME help in understanding the behavior of the model for individual predictions.

## Installation

To run this project, you need to have Python installed along with the necessary libraries. You can install the required libraries using the following command:

'''bash

pip install -r requirements.txt
'''

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

    Fork the repository
    Create a new branch (git checkout -b feature-branch)
    Make your changes and commit them (git commit -m 'Add new feature')
    Push to the branch (git push origin feature-branch)
    Open a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
