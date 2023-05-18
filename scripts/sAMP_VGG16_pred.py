import os
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import pathlib
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

def get_data1():
    data_path = pathlib.Path(r'../Dataset/')
    # pos_img 
    img_path_pos = list(data_path.glob('training_dataset/*pos*/*.jpeg')) 
    img_labels_pos = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], img_path_pos)) 
    # neg_img 
    img_path_neg = list(data_path.glob('training_dataset/*neg*/*.jpeg')) 
    img_labels_neg = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], img_path_neg))
    
    pd_img_path_pos = pd.Series(img_path_pos, name='PATH').astype(str) 
    pd_img_labels_pos = pd.Series(img_labels_pos, name='LABELS').astype(str) 
    pd_img_path_neg = pd.Series(img_path_neg, name='PATH').astype(str) 
    pd_img_labels_neg = pd.Series(img_labels_neg, name='LABELS').astype(str) 
    
    img_df_pos = pd.merge(pd_img_path_pos, pd_img_labels_pos, right_index=True, left_index=True) 
    img_df_neg = pd.merge(pd_img_path_neg, pd_img_labels_neg, right_index=True, left_index=True)
    img_train_pos = img_df_pos.sample(frac = 1).reset_index(drop=True) 
    img_train_neg = img_df_neg.sample(frac = 1).reset_index(drop=True)
    

    # pos_img 
    img_path_pos = list(data_path.glob('benchmark_dataset/*pos*/*.jpeg')) 
    img_labels_pos = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], img_path_pos)) 
    # neg_img 
    img_path_neg = list(data_path.glob('benchmark_dataset/*neg*/*.jpeg')) 
    img_labels_neg = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], img_path_neg))
    
    pd_img_path_pos = pd.Series(img_path_pos, name='PATH').astype(str) 
    pd_img_labels_pos = pd.Series(img_labels_pos, name='LABELS').astype(str) 
    pd_img_path_neg = pd.Series(img_path_neg, name='PATH').astype(str) 
    pd_img_labels_neg = pd.Series(img_labels_neg, name='LABELS').astype(str) 
    
    img_df_pos = pd.merge(pd_img_path_pos, pd_img_labels_pos, right_index=True, left_index=True) 
    img_df_neg = pd.merge(pd_img_path_neg, pd_img_labels_neg, right_index=True, left_index=True)
    img_indp1_pos = img_df_pos.sample(frac = 1).reset_index(drop=True) 
    img_indp1_neg = img_df_neg.sample(frac = 1).reset_index(drop=True)
    

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    datagen_test = ImageDataGenerator(preprocessing_function=preprocess_input)
    return img_train_pos, img_train_neg, img_indp1_pos, img_indp1_neg, datagen, datagen_test

def get_matrix(ds,label,pred, dataset_inp,dataset, acc, roc_auc, pr_auc, cohen_kappa_sc, sensitivity, specificity, mcc):
    y_score = model.predict(ds).ravel() 
    fpr,tpr, _ = roc_curve(ds.classes,y_score)
    roc_auc.append(metrics.auc(fpr,tpr))
    precision, recall, _ = precision_recall_curve(ds.classes,y_score)
    pr_auc.append(auc(recall, precision))
    cohen_kappa_sc.append(cohen_kappa_score(list(label.values),pred.flatten()))
    cm = confusion_matrix(list(label.values),pred.flatten())
    TP=cm[0,0] 
    TN=cm[1,1] 
    FN=cm[0,1] 
    FP=cm[1,0]
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    Recall = TP/(TP+FN)
    Specificity = TN/(TN+FP)
    MCC= ((TP*TN) -(FP*FN))/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    acc.append(Accuracy)
    sensitivity.append(Recall)
    specificity.append(Specificity)
    mcc.append(MCC)
    dataset.append(dataset_inp)
    return dataset, acc, roc_auc, pr_auc, cohen_kappa_sc, sensitivity, specificity, mcc
    
    
if __name__ == "__main__":  
    
    img_train_pos, img_train_neg, img_indp1_pos, img_indp1_neg, datagen, datagen_test=get_data1()
    
    width = 224; height = 224;
    batchn=24;
    df=pd.DataFrame(); acc=[]; roc_auc=[];pr_auc=[]; cohen_kappa_sc=[]; sensitivity=[]; specificity=[]; precision=[]; f1_score=[]; mcc=[]; dataset=[];
    train_ds = datagen.flow_from_dataframe(pd.concat([img_train_pos,img_train_neg]), 
                                         x_col='PATH', y_col='LABELS',
                                         target_size=(width,height),
                                         class_mode = 'binary', color_mode = 'rgb',
                                         batch_size = batchn, shuffle = False)
    indp1_ds = datagen.flow_from_dataframe(pd.concat([img_indp1_pos,img_indp1_neg]), 
                                         x_col='PATH', y_col='LABELS',
                                         target_size=(width,height),
                                         class_mode = 'binary', color_mode = 'rgb',
                                         batch_size = batchn, shuffle = False)

    train_label=pd.concat([img_train_pos['LABELS'],img_train_neg['LABELS']])
    indp1_label=pd.concat([img_indp1_pos['LABELS'],img_indp1_neg['LABELS']])
    
    train_label[train_label=='pos_img']=0; train_label[train_label=='neg_img']=1;
    indp1_label[indp1_label=='pos_img']=0; indp1_label[indp1_label=='neg_img']=1;
    
    model= keras.models.load_model('../sAMP_VGG16_model/sAMP_VGG16_model.h5') 
    pred=model.predict(train_ds)
    pred_train = np.where(pred > 0.5, 0, 1)
    pred=model.predict(indp1_ds)
    pred_indp1 = np.where(pred > 0.5, 0, 1)
    dataset, acc, roc_auc, pr_auc, cohen_kappa_sc, sensitivity, specificity, mcc=get_matrix(train_ds,train_label,pred_train, 'Training',dataset, acc, roc_auc, pr_auc, cohen_kappa_sc, sensitivity, specificity, mcc)
    dataset, acc, roc_auc, pr_auc, cohen_kappa_sc, sensitivity, specificity, mcc=get_matrix(indp1_ds,indp1_label,pred_indp1, 'Indp1',dataset, acc, roc_auc, pr_auc, cohen_kappa_sc, sensitivity, specificity, mcc)
    df['Dataset']=dataset
    df['Accuracy']=np.array(acc)*100
    df['AUC-ROC']=np.array(roc_auc)*100
    df['AUC-PR']=np.array(pr_auc)*100
    df['Kappa']=np.array(cohen_kappa_sc)*100
    df['Sn']=np.array(sensitivity)*100
    df['Sp']=np.array(specificity)*100
    df['MCC']=np.array(mcc)*100
    df.to_csv("final_results.csv",index=False)
    print(df)
