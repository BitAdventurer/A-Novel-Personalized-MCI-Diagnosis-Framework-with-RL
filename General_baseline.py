import os
import numpy as np
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from early_stop import EarlyStopping
import gcn_util
import config
import util
import seed
import loss_idd_baseline
import madgrad

args = config.parser.parse_args()
use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class chev(nn.Module):
    def __init__(self, in_features, out_features, K=2, bias=True):
        super(chev, self).__init__()
        
        # Seed setup
        seed.seed_everything(args.seed)

        # Layer definitions
        self.weight = nn.Parameter(torch.Tensor(K, in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(1, 1)) if bias else self.register_parameter('bias', None)

        # Graph convolution dropout
        self.gcn_drop = nn.Dropout(0.0)

        # MLP layers
        self.fc1 = self._init_mlp_layer(in_features * out_features, 64)
        self.fc2 = self._init_mlp_layer(64, 64)
        self.fc3 = self._init_mlp_layer(64, 64)
        self.fc4 = self._init_mlp_layer(64, 64)
        self.fc5 = self._init_mlp_layer(64, 64)
        self.fc6 = self._init_mlp_layer(64, 64)
        self.fc = nn.Linear(64, 2)
        torch.nn.init.xavier_normal_(self.fc.weight)

        # Reset parameters for GCN
        gcn_util.reset_parameters(self)

    def _init_mlp_layer(self, in_features, out_features):
        layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Dropout(0.0),
            nn.PReLu(0.5)
        )

        torch.nn.init.xavier_normal_(layer[0].weight)

        return layer

    def forward(self, x, adj_orig):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = gcn_util.laplacian_norm_adj(adj_orig)
        adj = gcn_util.add_self_loop(-adj)
        
        Tx_0, Tx_1 = x[0], x[0]
        #Tx_1 = Tx_1.unsqueeze(0)

        out = torch.matmul(Tx_0, self.weight[0])

        if self.weight.shape[0] > 1:
            Tx_1 = torch.matmul(adj, x)
            out = out + torch.matmul(Tx_1, self.weight[1])
            out = self._apply_gcn_drop_and_clamp(out)

        for k in range(2, self.weight.shape[0]):
            Tx_2 = 2 * torch.matmul(adj, Tx_1) - Tx_0
            out += torch.matmul(Tx_1, self.weight[k])
            out = self._apply_gcn_drop_and_clamp(out)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias:
            out += self.bias

        out = out.reshape(out.size(0), -1)
        out = self._apply_mlp(out)

        return F.softmax(out, dim=1)

    def _apply_gcn_drop_and_clamp(self, tensor):
        tensor = self.gcn_drop(tensor)
        mean, std = tensor.mean(), tensor.std()
        return torch.clamp(tensor, min=-10*(mean + std), max=10*(mean + std))

    def _apply_mlp(self, x):
        for fc_layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]:
            mean, std = x.mean(), x.std()
            x = torch.clamp(x, min=-20*(mean + std), max=20*(mean + std))
            x = fc_layer(x)
        return self.fc(x)


class Trainer:
    def __init__(self, model, args, criterion, optimizer, scheduler, fold, use_cuda=True):
        self.model = model
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.fold = fold
        self.use_cuda = use_cuda
        self.target_names = ['NC', 'EMCI', 'LMCI']


    def train(self, epoch, fold, idx, train_loader):
        self.model.train()
        train_correct = 0
        total = 0
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            self.optimizer.zero_grad()

            adj = torch.ones(116, 116).cuda()
            outputs = self.model(inputs.reshape(-1, 116, 116), adj)
            outputs = outputs.log()
            loss = self.criterion(outputs, torch.argmax(targets, dim=1))
            loss.backward()

            self.optimizer.step()

            running_loss += loss.data.cpu().numpy()
            _, predicted = torch.max(outputs.data, 1)
            _, targets = torch.max(targets, 1)
            total = len(train_loader.dataset)
            train_correct += predicted.eq(targets).cpu().sum()

            acc = 100. * train_correct / total
            if batch_idx % 5 == 1:
                print(f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.data:.6f}")
        self.scheduler.step()
        return running_loss / batch_idx, acc

    def validate(self, epoch, fold, idx, validation_data, validation_label):
        self.model.eval()
        val_correct = 0
        val_total = 0
        running_loss = 0.0
        acc = 0.0
        predicted1 = []
        targets1 = []

        for batch_idx, _ in enumerate(range(len(validation_data))):
            inputs, targets = torch.tensor(validation_data[batch_idx]), torch.tensor([validation_label[batch_idx]])
            
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                inputs, targets = Variable(inputs.reshape(-1,116,116)), Variable(targets)
                adj = torch.ones(116, 116).cuda()
                outputs = self.model(inputs.float(), adj)
                loss = self.criterion(outputs.log(), torch.argmax(targets, dim=1))

            running_loss += loss.data.cpu().numpy()
            _, predicted = torch.max(outputs.data, 1)
            _, targets = torch.max(targets, 1)

            targets1.extend(targets.tolist())  
            predicted1.extend(predicted.tolist())

            val_total = len(validation_data)
            val_correct += predicted.eq(targets).cpu().sum()

        acc = 100. * val_correct / val_total
        print(f'VALIDATION Correct: {val_correct}/{val_total} = {acc}%')
        return running_loss / batch_idx, acc, predicted1, targets1

    def test(self, epoch, fold, idx, test_data, test_label):
        self.model.eval()
        test_correct = 0
        test_total = 0
        running_loss = 0.0
        acc = 0.0
        predicted1 = []
        targets1 = []

        for batch_idx, _ in enumerate(range(len(test_data))):
            inputs, targets = torch.tensor(test_data[batch_idx]), torch.tensor([test_label[batch_idx]])
            
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                inputs, targets = Variable(inputs.reshape(-1,116,116)), Variable(targets)
                adj = torch.ones(116, 116).cuda()
                outputs = self.model(inputs.float(), adj)
                loss = self.criterion(outputs.log(), torch.argmax(targets, dim=1))

            running_loss += loss.data.cpu().numpy()
            _, predicted = torch.max(outputs.data.cuda(), 1)
            _, targets = torch.max(targets, 1)

            targets1.extend(targets.tolist())  
            predicted1.extend(predicted.tolist())

            test_total = len(test_data)
            test_correct += predicted.eq(targets).cpu().sum()

        acc = 100. * test_correct / test_total
        print(f'TEST Correct: {test_correct}/{test_total} = {acc}%')
        return running_loss / batch_idx, acc, predicted1, targets1


    def calculate_metrics(self, matrix):
        num_classes = matrix.shape[0]
        sensitivities = []
        specificities = []
        precisions = []
        f1_scores = []

        for i in range(num_classes):
            TP = matrix[i, i]
            FN = sum(matrix[i, :]) - TP
            FP = sum(matrix[:, i]) - TP
            TN = np.sum(matrix) - TP - FN - FP

            sensitivity = TP / (TP + FN) if TP + FN != 0 else 0
            sensitivities.append(sensitivity)
            specificity = TN / (TN + FP) if TN + FP != 0 else 0
            specificities.append(specificity)
            precision = TP / (TP + FP) if TP + FP != 0 else 0
            precisions.append(precision)
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if precision + sensitivity != 0 else 0
            f1_scores.append(f1)

        total_samples = np.sum(matrix)
        weights = [sum(row) for row in matrix]

        weighted_sensitivity = sum([sens * weight for sens, weight in zip(sensitivities, weights)]) / total_samples
        weighted_specificity = sum([spec * weight for spec, weight in zip(specificities, weights)]) / total_samples
        weighted_precision = sum([prec * weight for prec, weight in zip(precisions, weights)]) / total_samples
        weighted_f1 = sum([f1 * weight for f1, weight in zip(f1_scores, weights)]) / total_samples

        return weighted_sensitivity, weighted_specificity, weighted_precision, weighted_f1




    def train_baseline(self):
        for i in range(args.n_split-1):
            train_loader = util.self_load_data(args.path+f'/data/fold{self.fold}', f'train_loader_fold{self.fold}_split{i}.history')

            # Load Validation Data
            validation_data = util.validation_data(i)
            validation_label = util.validation_label(i)
            
            epochs         = 1000
            early_stopping = EarlyStopping(patience=200, verbose=True, delta=0.01, val_cm1=0, val_cm2=0, test_cm1=0, test_cm2=0, split=i)
            
            for epoch in tqdm(range(epochs)):
                train_loss, train_acc = self.train(epoch, self.fold, i, train_loader)  # Assuming you have a train method for training
                val_loss, val_acc, val_predicted, val_targets = self.validate(epoch, self.fold, i, validation_data, validation_label)
                
                
                # Calculate metrics
                val_matrix = confusion_matrix(val_targets , val_predicted)
                val_sen, val_spec, val_prec, val_f1 = self.calculate_metrics(val_matrix)

                # Print progress
                print(f"Epoch: {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
                print(f"Train Accuracy: {train_acc:.2f}, Validation Accuracy: {val_acc:.2f}")
                print(f"Validation - Sensitivity: {val_sen:.2f}, Specificity: {val_spec:.2f}, F1 Score: {val_f1:.2f}")

                # Save results
                results = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_sen": val_sen,
                    "val_spec": val_spec,
                    "val_f1": val_f1
                }

                with open(self.args.path + f'/baseline/fold{self.fold}/results_{i}.json', 'a') as f:
                    for key, value in results.items():
                        if isinstance(value, torch.Tensor):
                            results[key] = value.tolist()
                    json.dump(results, f)
                    f.write("\n")

                # Save model
                #torch.save(self.model, self.args.path + f'/baseline/fold{self.fold}/model_{i}.pt')

                # Early stopping
                early_stopping(val_acc+val_f1, self.model, val_targets, val_predicted, i)
                if early_stopping.early_stop:
                    loss_idd_baseline.loss_idd(i)
                    self.model = chev(in_features=116, out_features=120, K=5)
                    if use_cuda:
                        self.model = self.model.float().cuda()
                    print("Early stopping")
                    break

