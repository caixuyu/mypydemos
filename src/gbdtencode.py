# encoding:utf-8
import sys

import xgboost as xgb
import time
from sklearn.metrics import roc_curve,auc ,roc_auc_score
from sklearn import metrics

def analysis_xgboostModel(model_path):
    space = {}
    tree_index = -1
    leaf_index = 1
    with open(model_path) as treeFile:
        for line in treeFile:
            content = line.strip('\n').strip('\t')
            if content[0:7]== 'booster':
                tree_index += 1
            elif content.split(':')[1][0:4]=='leaf':
                space[str(tree_index)+':'+content.split(':')[0]] = str(leaf_index)
                leaf+=1
    return space

def new_sample_output(output_path, labels, preds, feature_space_dict ):
    with open(output_path,'w') as output_file:
        label_index = 0
        for leaf_list in preds:
            temp_str = str(int(labels[label_index]))+' '
            label_index += 1
            boost_tree_num = 0
            for ele in leaf_list:
                key = str(boost_tree_num) + ':' + str(ele)
                value = feature_space_dict[key]
                temp_str += value + ':' +'1 '
                boost_tree_num += 1
            output_file.write(temp_str[:1]+'\n')

def run(train_path, test_path, mo_txt_path, encoded_train_path, encoded_test_path, bin_model_save):
    dtrain = xgb.DMatrix(train_path)
    dtest = xgb.DMatrix(test_path)
    train_labels = dtrain.get_label()
    test_labels = dtest.get_label()

    param = { 'max_depth':5, 'eta':0.3, 'objective':'binary:logistic', 'nthread':5, 'eval_metric':'auc' }

    watchlist = [(dtrain, 'train'), (dtest,'valid')]
    num_round = 120

    print 'xgboost model training.'
    bst = xgb.train(param, dtrain, num_round, watchlist)
    bst.save_model(bin_model_save)
    print 'model train finish.'

    print 'start predicting.'
    train_preds = bst.predict(dtrain, pred_leaf=True)
    test_preds = bst.predict(dtest, pred_leaf=True)

    bst.dump_model(mo_txt_path)
    print 'reading dump model feature space.'
    new_feature_space = analysis_xgboostModel(mo_txt_path)

    new_sample_output(encoded_train_path, train_labels, train_preds, new_feature_space)
    new_sample_output(encoded_test_path, test_labels, test_preds, new_feature_space)
    print 'finish encoding.'

if __name__ == '__main__':
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    model_txt_path = sys.argv[3]
    encoded_train_path = sys.argv[4]
    encoded_test_path = sys.argv[5]
    bin_model_save = sys.argv[6]
    run(train_data_path, test_data_path, model_txt_path, encoded_train_path, encoded_test_path, bin_model_save)
