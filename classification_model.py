import prediction_model


class ClassificationModel(prediction_model.PredictionModel):
    def __init__(self, classifier, models=[], input_features=[], out_feature=None, class_names_dict=None):
        if out_feature is None:
            out_feature = []
        else:
            out_feature = [out_feature]
        super(ClassificationModel, self).__init__(classifier,
                                                  models=models,
                                                  input_features=input_features,
                                                  out_features=out_feature)
        self.class_names_dict = class_names_dict


