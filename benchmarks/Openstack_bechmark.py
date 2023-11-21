#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
import pandas as pd
from loglizer.models import *
from loglizer import dataloader, preprocessing
import matplotlib.pyplot as plt

run_models = ['PCA','LogClustering', 'IsolationForest', 
              'SVM', 'DecisionTree']

struct_log = '../../log-anomaly-benchmark/OpenStack_structured/OpenStack_full.log_structured.csv' # The structured log file
label_file = '../../log-anomaly-benchmark/processed/datetime.csv' # The anomaly label file

if __name__ == '__main__':
    (x_tr, y_train), (x_te, y_test) = dataloader.load_OpenStack(struct_log,
                                                           window='session', 
                                                           label_file=label_file,
                                                           train_ratio=0.5,
                                                           save_csv=True,
                                                           split_type='uniform',
                                                           normalize=True)
    benchmark_results = []
    for _model in run_models:
        print('Evaluating {} on HDFS:'.format(_model))
        if _model == 'PCA':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf', 
                                                      normalization='zero-mean')
            
            model = PCA(n_components=0.5)
            model.fit(x_train)
        
        elif _model == 'InvariantsMiner':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr)
            model = InvariantsMiner(epsilon=0.5)
            model.fit(x_train)

        elif _model == 'LogClustering':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = LogClustering(max_dist=0.8, anomaly_threshold=0.05)
            model.fit(x_train[y_train == 0, :]) # Use only normal samples for training

        elif _model == 'IsolationForest':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr)
            model = IsolationForest(contamination=0.5, n_estimators=4)
            model.fit(x_train)

        elif _model == 'LR':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = LR()
            model.fit(x_train, y_train)

        elif _model == 'SVM':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = SVM(C=5)
            model.fit(x_train, y_train)

        elif _model == 'DecisionTree':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf', normalization='sigmoid')
            model = DecisionTree()
            model.fit(x_train, y_train)
        
        x_test = feature_extractor.transform(x_te)
        print('Train accuracy:')
        precision, recall, f1 = model.evaluate(x_train, y_train)
        benchmark_results.append([_model + '-train', precision, recall, f1])
        print('Test accuracy:')
        precision, recall, f1 = model.evaluate(x_test, y_test)
        benchmark_results.append([_model + '-test', precision, recall, f1])

    df = pd.DataFrame(benchmark_results, columns=['Model', 'Precision', 'Recall', 'F1']) \
        .to_csv('benchmark_result.csv', index=False)

    labels = ['Precision', 'Recall', 'F1-score']
    
    f, ax = plt.subplots(2, 3, figsize=(20,10))
    for i in range(len(run_models)):
        result = benchmark_results[i*2+1]
        model = run_models[i]
        values = [result[1], result[2], result[3]]
        ax[i//3, i%3].bar(labels, values)
        ax[i//3, i%3].title.set_text(model)
        for j in range(3):
            ax[i//3, i%3].text(labels[j], values[j], "{:.3f}".format(values[j]))
        
    ax[1][2].set_visible(False)
    f.suptitle("OpenStack dataset benchmark results")
    plt.savefig('benchmark_result.png')