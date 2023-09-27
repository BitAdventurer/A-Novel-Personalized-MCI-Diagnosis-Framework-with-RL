import os
import scipy.io 
import numpy as np
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
import torch
import pyximport; pyximport.install()
from pathlib import Path
import seed as sd
import plot_cv
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import config
args = config.parser.parse_args()

cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm

n_splits = args.n_split
batch_size = 32
save_switch = True

sd.seed_everything(args.seed)

# MCI Labeling
path = args.path

########################################### ORIGINAL
os.chdir(path+'/data/multiscanX/MCI')

original_anomy = []
original_anomy_meta = []
# <0,1> 으로 Labeling
for i in range(len(os.listdir())):
    original_mat_mci = scipy.io.loadmat(os.listdir()[i])
    original_anomy_meta.append(os.listdir()[i])
    original_mat_mci = original_mat_mci['ROICorrelation_FisherZ']
    original_anomy.append([[0, 1], original_mat_mci])

# NC Labeling
os.chdir(path+'/data/multiscanX/NC')

original_normal = []
original_normal_meta = []
# <1,0> 으로 Labeling
for i in range(len(os.listdir())):
    original_mat_nc = scipy.io.loadmat(os.listdir()[i])
    original_normal_meta.append(os.listdir()[i])
    original_mat_nc = original_mat_nc['ROICorrelation_FisherZ'] # ROICorrelation_FisherZ
    original_normal.append([[1, 0], original_mat_nc])

# date merge
original_dataset = original_anomy + original_normal
original_dataset_meta = original_anomy_meta + original_normal_meta

# inf -> 1로 전처리
for i in range(len(original_dataset)):
    for j in range(116):
        for k in range(116):
            if original_dataset[i][1][j][k] == float('inf'):
                original_dataset[i][1][j][k] = 1



# Data, Label 분리
original_data, original_label = [], []
for i in range(len(original_dataset)):
    original_data.append(original_dataset[i][1])
    original_label.append(original_dataset[i][0])

########################################### MULTISCAN X
os.chdir(path+'/data/multiscanX/MCI')

anomy = []
anomy_meta = []
# <0,1> 으로 Labeling
for i in range(len(os.listdir())):
    mat_mci = scipy.io.loadmat(os.listdir()[i])
    anomy_meta.append(os.listdir()[i])
    mat_mci = mat_mci['ROICorrelation_FisherZ']
    anomy.append([[0, 1], mat_mci])

# NC Labeling
os.chdir(path+'/data/multiscanX/NC')

normal = []
normal_meta = []
# <1,0> 으로 Labeling
for i in range(len(os.listdir())):
    mat_nc = scipy.io.loadmat(os.listdir()[i])
    normal_meta.append(os.listdir()[i])
    mat_nc = mat_nc['ROICorrelation_FisherZ'] # ROICorrelation_FisherZ
    normal.append([[1, 0], mat_nc])

# date merge
dataset = anomy + normal
dataset_meta = anomy_meta + normal_meta

# inf -> 1로 전처리
for i in range(len(dataset)):
    for j in range(116):
        for k in range(116):
            if dataset[i][1][j][k] == float('inf'):
                dataset[i][1][j][k] = 1


# Data, Label 분리
data, label = [], []
for i in range(len(dataset)):
    data.append(dataset[i][1])
    label.append(dataset[i][0])


########################################### train/test
os.chdir(path)
cv =  StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=args.dataset_seed)  # original_seed = args.seed
train_set, test_set = [], [] 

for i, v in cv.split(data, label):
    train_set.append(i) # (3, 336)
    test_set.append(v) # (3,169)

########################################### PLOT
fig, ax = plt.subplots(figsize=(15, 8))
plot_cv.plot_cv_indices(cv, data, label, ax, n_splits)

ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.2))],
            ['Testing set', 'Training set'])
# Make the legend fit
plt.tight_layout()
plt.savefig('cs.png', dpi=300)
plt.close()

########################################### TEST
for idx, split in enumerate(test_set):  # (3,169)
    test, test_label = [], []
    test_meta = []

    for test_idx in split:
        test.append(data[test_idx])
        test_label.append(label[test_idx])
        test_meta.append(dataset_meta[test_idx])

    cross_test, cross_test_label = [], []
    down_sample_test_nc = 0
    down_sample_test_mci = 0

    for train_idx in split: # 169
        if 'NC' in original_dataset_meta[train_idx] or 'SMC' in original_dataset_meta[train_idx]:
            down_sample_test_nc+=1
        else:
            down_sample_test_mci+=1

    print('Test mci :',down_sample_test_mci,'Test nc :', down_sample_test_nc)

    if save_switch:
        test_dataset = TensorDataset(torch.Tensor(np.array(test)).unsqueeze(1), torch.Tensor(test_label))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

        test_path = path + '/data/fold{}/test_fold{}.history'.format(idx, idx)
        with open(test_path, mode='wb') as f:
            torch.save(test, f)

        test_label_path = path + '/data/fold{}/test_label_fold{}.history'.format(idx, idx)
        with open(test_label_path, mode='wb') as f:
            torch.save(test_label, f)

        test_loader_path = path + '/data/fold{}/test_loader_fold{}.history'.format(idx, idx)
        with open(test_loader_path, mode='wb') as f:
            torch.save(test_loader, f)

########################################### TRAIN
for idx, split in enumerate(train_set): 
    
    train, train_label = [], []
    train_meta = []

    for train_idx in split:
        train.append(data[train_idx])
        train_label.append(label[train_idx])
        train_meta.append(dataset_meta[train_idx])
    
    ########################################### ONE OF TRAIN FOLD
    train_val_set, validation_set = [], []
    
    ########################################### TRAIN / VALIDATION
    cv2 =  StratifiedShuffleSplit(n_splits=n_splits-1, test_size=0.25, random_state=args.dataset_seed)  # original_seed = args.seed

    for ii, vv in cv2.split(train, train_label): # idx
        train_val_set.append(ii)
        validation_set.append(vv)

    ########################################### DATA SAVE
    for split_idx, split in enumerate(train_val_set):

        cross_train, cross_train_label = [], []
        down_sample_train_nc = 0
        down_sample_train_mci = 0
        for train_idx in split:
            for k in range(len(original_dataset_meta)):
                try:
                    if train_meta[train_idx] == original_dataset_meta[k]:

                        if 'NC' in original_dataset_meta[k] or 'SMC' in original_dataset_meta[k]:
                            train_data = scipy.io.loadmat(path + '/data/multiscanX/NC/' + original_dataset_meta[k])['ROICorrelation_FisherZ']
                            down_sample_train_nc+=1
                        else:
                            train_data = scipy.io.loadmat(path + '/data/multiscanX/MCI/' + original_dataset_meta[k])['ROICorrelation_FisherZ']
                            down_sample_train_mci+=1
                        
                        for j in range(116):
                            if train_data[j][j] == float('inf'):
                                train_data[j][j] = 1
                        
                        cross_train.append(train_data)
                        cross_train_label.append(train_label[train_idx])
                except:
                    pass

        print('Train mci :', down_sample_train_mci,'Train nc :', down_sample_train_nc)

        #original_dataset_meta
        if save_switch:
            train_dataset = TensorDataset(torch.Tensor(np.array(cross_train)).unsqueeze(1), torch.Tensor(cross_train_label))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

            train_path = path + '/data/fold{}/train_fold{}_split{}.history'.format(idx, idx,split_idx)
            with open(train_path, mode='wb') as f:
                torch.save(cross_train, f)

            train_label_path = path + '/data/fold{}/train_label_fold{}_split{}.history'.format(idx, idx,split_idx)
            with open(train_label_path, mode='wb') as f:
                torch.save(cross_train_label, f)

            train_loader_path = path + '/data/fold{}/train_loader_fold{}_split{}.history'.format(idx, idx,split_idx)
            with open(train_loader_path, mode='wb') as f:
                torch.save(train_loader, f)

    for split_idx, split in enumerate(validation_set):
        cross_validation, cross_validation_label = [], []
        down_sample_validation_nc = 0
        down_sample_validation_mci = 0
        for val_idx in split:
            cross_validation.append(train[val_idx])
            cross_validation_label.append(train_label[val_idx])
            for k in range(len(original_dataset_meta)):
                try:
                    if train_meta[val_idx][:-4] == original_dataset_meta[k][:-4]:
                        if 'NC' in original_dataset_meta[k] or 'SMC' in original_dataset_meta[k]:
                        #    validation_data = scipy.io.loadmat(path + '/data/original/NC/NC_Z/' + original_dataset_meta[k])['ROICorrelation_FisherZ']
                            down_sample_validation_nc+=1
                        else:
                        #    validation_data = scipy.io.loadmat(path + '/data/original/MCI/MCI_Z/' + original_dataset_meta[k])['ROICorrelation_FisherZ']
                            down_sample_validation_mci+=1

                        #for j in range(116):
                        #    if validation_data[j][j] == float('inf'):
                        #        validation_data[j][j] = 1

                        #cross_validation.append(validation_data)
                        #cross_validation_label.append(train_label[val_idx])
                except:
                    pass
        print('validation mci :',down_sample_validation_mci,'validation nc :', down_sample_validation_nc)

        if save_switch:
            validation_dataset = TensorDataset(torch.Tensor(np.array(cross_validation)).unsqueeze(1), torch.Tensor(cross_validation_label))
            validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

            validation_path = path + '/data/fold{}/validation_fold{}_split{}.history'.format(idx, idx, split_idx)
            with open(validation_path, mode='wb') as f:
                torch.save(cross_validation, f)

            validation_label_path = path + '/data/fold{}/validation_label_fold{}_split{}.history'.format(idx, idx, split_idx)
            with open(validation_label_path, mode='wb') as f:
                torch.save(cross_validation_label, f)
            
            validation_loader_path = path + '/data/fold{}/validation_loader_fold{}_split{}.history'.format(idx, idx, split_idx)
            with open(validation_loader_path, mode='wb') as f:
                torch.save(validation_loader, f)

    print('cross_train : ', len(cross_train), 'cross_validation : ', len(cross_validation), 'test : ', len(test))


os.chdir(path)
if __name__ == "__main__":
    pass
