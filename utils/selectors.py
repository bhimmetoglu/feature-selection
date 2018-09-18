# Python 3
# Burak Himmetoglu (burakhmmtgl@gmail.com)
# Utilities for feature selection

# imports
import numpy as np
import pandas as pd
from scipy import stats
from itertools import product, combinations

# joblib (parallel execution)
from joblib import Parallel, delayed

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterGrid, StratifiedKFold, RepeatedStratifiedKFold

# -- Feature differentials
def diff_express(X_p, X_n):
    """
    Return p-value (t-test) and Cohen's d effect size from binary data
    matrices (X_p, X_n)

    X_n : Data matrix for negative class
    X_p : Data matrix for positive class
    """
    # Positive/Negative sample numbers
    n_p = len(X_p)
    n_n = len(X_n)
    
    # Perform t-test for all features
    t, p = stats.ttest_ind(X_p, X_n, axis=0, equal_var=False)
    
    # Means and variances
    mu_p = X_p.mean(axis=0)
    mu_n = X_n.mean(axis=0)
    mean_diff = mu_p - mu_n
    var_p = X_p.var(axis=0)
    var_n = X_n.var(axis=0)
    
    # Pooled variance and Effect size
    S_pooled = np.sqrt( ((n_p-1)*var_p + (n_n-1)*var_n)/(n_p + n_n-2) )
    Cd = np.divide(mean_diff,S_pooled)
    
    # Overlap and misclassification rate
    thr = (mu_p * np.sqrt(var_n) + mu_n * np.sqrt(var_p)) / (np.sqrt(var_n) + np.sqrt(var_p))
    n_above = np.sum(X_n > thr, axis=0)
    p_below = np.sum(X_p < thr, axis=0)
    
    mc_rate = 0.5* ( n_above / n_n + p_below / n_p)
   
    return p, Cd, mc_rate

# -- Volcano generator
def volcano(dat, col_label='label'):
    """
    Given dataframe (dat), return p-value and effect size (overlap)
    """
    # Positive/Negative
    positive = dat[dat[col_label]==1].drop(col_label,axis=1).values; n_p = len(positive)
    negative = dat[dat[col_label]==0].drop(col_label, axis=1).values; n_n = len(negative)
    
    # Get p-value and effect size
    pval, cd, mcr = diff_express(positive, negative)
    
    # As df
    arr = np.hstack((pval.reshape(-1,1),mcr.reshape(-1,1)))
    df_ = pd.DataFrame(arr, index=list(dat.columns)[:-1], columns=['pval','mc_rate'])
    
    return df_

# -- Cross-validation score calculator for forward feature selection
def cv_score_ridge(fc, dat, current_feats, label_col, n_splits, n_repeats):
    """
    Return CV score using Ridge Regression for forward feature selection
        
    Inputs:
    -----------------------------------------------------------------
    fc            : tuple (feature, C)
    dat           : Dataframe (features, label)
    current_feats : Current set of features in forward selection
    label_col     : Name of the label column
    n_splits      : Number of splits
    n_repeats     : Number of repeats
        
    Output:
    -----------------------------------------------------------------
    Mean CV score when C is used as hyperparameter and 
    current_feats + [feature] is used as features
    
    """
    # Initiate
    scores = [] 
    skf = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state=12345)
    
    # Get tested feature and C
    feat_, c_ = fc
    
    # Label and design matrix
    y = dat[label_col].values
    X = dat[current_feats + [feat_]].values
    
    # Loop over folds
    for idx_tr, idx_vld in skf.split(X,y):
        x_tr = X[idx_tr]; y_tr = y[idx_tr]
        x_vld = X[idx_vld]; y_vld = y[idx_vld]
        
        # Logistic regression
        clf = LogisticRegression(C=c_, penalty='l2')
        clf.fit(x_tr, y_tr)
        
        # Predict on validation data
        y_prob_vld = clf.predict_proba(x_vld)
        
        # Get score
        sc = -log_loss(y_vld, y_prob_vld[:,1])
        scores.append(sc)
        
    return np.mean(scores)


# -- Forward feature selector 
def forward_select_ridge(dat, feature_list, Cs = np.logspace(-2,2,20), 
                         label_col='label', n_repeats=5, n_splits=5, n_max = 10):
    """
    This function performs forward feature selection with Ridge Regression
    
    Inputs:
    -----------------------------------------------------------------------
    dat          : DataFrame (features, label)
    feature_list : List of features to forward select from
    Cs           : Ridge Regression Cs to search
    label_col    : Name of the label column
    n_splits      : Number of splits
    n_repeats     : Number of repeats
    n_max        : Maximum number of features to select
    
    Output:
    -----------------------------------------------------------------------
    A list of tuples where each tuple contains:
    (selected_features, best_C, best_cv_score)
    
    """
    # Initiate 
    current_features = []
    n_feat = 0
    y = dat[label_col].values
    
    # Final results to be collected in report
    report = []
    
    # Main selection loop
    while (n_feat < n_max):
        scores = []
        
        # Features and Cs combination
        feat_cs = list(product(feature_list, Cs))
        
        # Compute the CV scores in parallel (use all CPUs)
        with Parallel(n_jobs=-1) as par:
            scores = par(delayed(cv_score_ridge)(fc, dat, current_features, label_col, n_splits, n_repeats) for fc in feat_cs)
        
        # Best score and chosen feature
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_feature, best_C = feat_cs[best_idx]
        
        # Print message
        print("Selected feature = {:s} :: best C = {:.4f} :: current CV score = {:.4f}".format(
            best_feature, best_C, best_score))
        
        # Append best feature to current_motifs
        current_features.append(best_feature)
        n_feat +=1
        
        # Pop selected feature from starting list
        feature_list.remove(best_feature)
        
        # Update report
        feats_ = [m for m in current_features] # local copy
        report.append((feats_, best_C, best_score))
        
    return report
