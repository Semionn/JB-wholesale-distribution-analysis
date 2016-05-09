class PredictionModel:
    """Abstract model for prediction

    This class is designed to create cascade of regression and classification models

    Parameters
    ----------
    models: list of PredictionModel

    Attributes
    ----------
    features : list of strings
        List params for fitting and prediction

    See also
    --------
    RegressionModel
    """

    def __init__(self, base_model, models=[], input_features=[], out_features=[]):
        self.base_model = base_model
        self.input_features = []
        for model in models:
            self.input_features += model.out_features.keys()
        self.input_features += input_features
        self.out_features = out_features

    def fit(self, x, y=None):
        """Fit estimator.

        Parameters
        ----------
        x : The input samples, shape = [n_samples, n_features]

        y: The output values, shape = [n_samples] or [n_samples, n_outputs]

        Returns
        -------
        self : object
            Returns self.

        """
        self.base_model.fit(x, y)
        return self

    def predict(self, x):
        """Predict regression target for x.

        Parameters
        ----------
        x : The input samples, shape = [n_samples, n_features]

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.
        """
        return self.base_model.predict(x)
