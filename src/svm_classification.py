import numpy as np
import torch
import random
import argparse
import matplotlib.pyplot as plt
import time
from pickle import dump
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

def get_args_parser():
    parser = argparse.ArgumentParser('SVM feature computation', add_help = False)
    parser.add_argument('--C', default = 10e1, type = int)
    parser.add_argument('--gamma', default = 2.0)
    parser.add_argument('--sample_ratio', default = 0.01, type = float)
    parser.add_argument('--test_split', default = 0.8, type = float)
    parser.add_argument('--features_path', default = './features/')
    parser.add_argument('--linear', default = False, type = bool) # Whether to use linear or rbf kernel
    parser.add_argument('--normalize', default = True, type = bool) # Whether to normalize the data, suggested
    parser.add_argument('--cv', default=0, type = int) # k of k-fold cross validation. 0=no cross validation    
    return parser

def main(args):
    # Load predictors
    data =  np.load(args.features_path+'_data.npy').flatten()
    alpha = np.load(args.features_path+'_alpha.npy').flatten()
    beta =  np.load(args.features_path+'_beta.npy').flatten()
    ent =   np.load(args.features_path+'_ent.npy').flatten()
    pos =   np.load(args.features_path+'_pos.npy').flatten()
    kl  =   np.load(args.features_path+'_kl.npy').flatten()
    rel =   np.load(args.features_path+'_rel.npy').flatten()
    
    # Stack predictors, load segmentation
    X = np.stack([data,alpha,beta,ent,pos,kl,rel]).T
    y = np.load(args.features_path+'_seg.npy').flatten()

    # Normalize predictors
    if args.normalize:
        X = (X-X.mean(axis = 0))/X.std(axis=0)

    # Remove uncertain class and air
    rows_to_keep = np.logical_and(y != 4, y !=0)
    X = X[rows_to_keep,:]
    y = y[rows_to_keep]

    # Keep only 1% at random
    keep = random.sample(range(len(y)), k=int(len(y)*args.sample_ratio))
    X = X[keep,:]
    y = y[keep]

    # Columns to keep
    X = X[:,[0,1,2,3,4,5]]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_split, random_state=42)
    print('\nTrain samples:',y_train.shape)
    print('Test samples',y_test.shape)

    # Create SVM classifier
    t0 = time.time()
    if args.linear:
        classifier = svm.SVC(C = args.C, gamma=args.gamma, kernel='linear', verbose = True, max_iter=1000)
    else:
        classifier = svm.SVC(C = args.C, gamma=args.gamma, kernel='rbf', verbose = True, max_iter=1000)

    # Fit classifier on training set
    fitted = classifier.fit(X_train,y_train)
    print('\nFitted in',time.time()-t0,'seconds\n\n')

    # Make predictions on train and test sets
    y_train_pred = fitted.predict(X_train)
    y_test_pred  = fitted.predict(X_test)

    # Save fitted model
    with open("latest_model.pkl", "wb") as f:
        dump(fitted, f, protocol=5)

    # Evaluate performance
    target_names = ['Ice', 'Bedrock', 'Noise']
    # ------ TRAIN SET PERFORMANCE ------
    print("Training set performance:")
    print(classification_report(y_train, y_train_pred, target_names = target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))

    # ------ TEST SET PERFORMANCE ------
    print("\nTest set performance:")
    print(classification_report(y_test, y_test_pred, target_names = target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    #K-Fold cross validation
    if args.cv > 0:
        print('\nCross Validation:')
        scores = cross_val_score(classifier, X_train, y_train, cv = args.cv)
        print('\nCV Score mean:{}'.format(scores.mean()))
        print('CV Score std:{}'.format(scores.std()))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)