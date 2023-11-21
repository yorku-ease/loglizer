#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import LR
from loglizer import dataloader, preprocessing

struct_log = '../../log-anomaly-benchmark/OpenStack_structured/OpenStack_full.log_structured.csv' # The structured log file
label_file = '../../log-anomaly-benchmark/processed/datetime.csv'

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_OpenStack(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='uniform')

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)

    model = LR()
    model.fit(x_train, y_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)
