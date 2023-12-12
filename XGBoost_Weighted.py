from typing import Any, Dict
import xgboost as xgb
import numpy as np

class XGBClassifier_w(xgb.XGBClassifier):


    def __init__(self, w0, w1, w2, w3, w4, w5, w6, w7, w8, **kwargs):
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6
        self.w7 = w7
        self.w8 = w8
        self.w = np.array([w0, w1, w2, w3, w4, w5, w6, w7, w8])
        super().__init__(**kwargs)


    def fit(self, X, y, sample_weight=None,
                    base_margin = None,
                    eval_set = None,
                    eval_metric = None,
                    early_stopping_rounds = None,
                    verbose = True,
                    xgb_model = None,
                    sample_weight_eval_set = None,
                    base_margin_eval_set = None,
                    feature_weights = None,
                    callbacks = None):

        sample_weight=[np.sum(self.w * i) for i in y]

        super().fit(X, y, sample_weight = sample_weight,
                    base_margin = base_margin,
                    eval_set = eval_set,
                    eval_metric = eval_metric,
                    early_stopping_rounds = early_stopping_rounds,
                    verbose = verbose,
                    xgb_model = xgb_model,
                    sample_weight_eval_set = sample_weight_eval_set,
                    base_margin_eval_set = base_margin_eval_set,
                    feature_weights = feature_weights,
                    callbacks = callbacks)
