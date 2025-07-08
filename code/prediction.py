import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score
import random


def get_old_data():
    filename = '../data/VLP200.csv'
    df = pd.read_csv(filename)
    data = []
    for i in range(200):
        pdb = df['PDB code'][i]
        seq = df['Protein sequence'][i]
        typ = df['Stoichiometry'][i]
        label = 1
        if typ==60:
            label = 0
        #print(i,pdb,label)
        data.append([ pdb,seq,label ])
    return data



def get_feature():
    data = get_old_data()
    feature = []
    label = []
    
    for i in range(200):
        pdb,_,y = data[i]
        filename = './feature/' + pdb + '/' + 'topo.npy'
        topo = np.load(filename).tolist()
        tmp = topo
        feature.append(tmp)
        label.append(y)
    
    feature = np.array(feature)
    label = np.array(label)
    return feature,label
    




def gradientboostingtree(train_feature,train_label,test_feature,test_label):
    params={'n_estimators': 4000, 'max_depth': 7, 'min_samples_split': 3,
                'learning_rate': 0.01,'max_features':'sqrt','subsample':0.7,
                }
    model = GradientBoostingClassifier(**params)
    model.fit(train_feature, train_label)
    y_pred = model.predict(test_feature)
    y_prob = model.predict_proba(test_feature)[:, 1]
    
    auc = roc_auc_score(test_label, y_prob)
    tn, fp, fn, tp = confusion_matrix(test_label, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return auc,sensitivity,specificity,precision,npv
    
    

def cv10():
    feature,label = get_feature()
    
    seed = random.randint(1, 10000)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    auc_list = []
    sen_list = []
    spe_list = []
    pre_list = []
    npv_list = []


    for train_index, test_index in cv.split(feature, label):
        train_feature, test_feature = feature[train_index], feature[test_index]
        train_label, test_label = label[train_index], label[test_index]
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_feature = scaler.fit_transform(train_feature)
        test_feature = scaler.transform(test_feature)
        
        auc,sen,spe,pre,npv = gradientboostingtree(train_feature,train_label,test_feature,test_label)
        auc_list.append(auc)
        sen_list.append(sen)
        spe_list.append(spe)
        pre_list.append(pre)
        npv_list.append(npv)
    
    print('auc:',np.round(np.mean(auc_list),3),'sensitivity:',np.round(np.mean(sen_list),3),'specificity:',np.round(np.mean(spe_list),3),'precision:',np.round(np.mean(pre_list),3),'npv:',np.round(np.mean(npv_list),3))
    
        
    
cv10()
    