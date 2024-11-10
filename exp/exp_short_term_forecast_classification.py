from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary_class import M4SummaryClassification
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shutil

warnings.filterwarnings('ignore')


class Exp_Short_Term_Forecast_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast_Classification, self).__init__(args)

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='CrossEntropy'): #改 默认 criterion
#-------------------------------------------------
        if loss_name == 'CrossEntropy':
            return nn.CrossEntropyLoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if os.path.exists(path):
            shutil.rmtree(path)  # 删除之前的 checkpoint 文件夹

        # 重新创建清空后的文件夹
        os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        mse = nn.MSELoss()


        CrossEntropy = nn.CrossEntropyLoss() # 要用这个metric
#-------------------------------------------------

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            #---------------------------------debug
            for data in train_loader:
                print("长这个样子：",data)  # 检查返回的数据结构
                break
            #---------------------------------debug
            for i, (batch_x, batch_y, mask) in enumerate(train_loader):
                #----------------------------debug
                print(f"Batch {i} loaded")
                print("batch_x shape:", batch_x.shape)
                print("batch_y shape:", batch_y.shape)
                print("batch_x sample:", batch_x[0])
                print("batch_y sample:", batch_y[0])
                #----------------------------debug
                if batch_x is None or batch_y is None:
                    continue
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                """print("batch_x shape:", batch_x.shape)
                print("batch_x min:", batch_x.min().item(), "max:", batch_x.max().item())
                print("batch_x mean:", batch_x.mean().item(), "std:", batch_x.std().item())

                print("batch_x:", batch_x)
                print("batch_y shape:", batch_y.shape) """
                if hasattr(self, 'batch_x_mark') and hasattr(self, 'batch_y_mark'):
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
#-------------------------------------------------- timestamp feature 可以有可以没有


                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, None, dec_inp, None)
                """print("Model outputs before process:", outputs.shape, outputs)
                print("feature =", self.args.features)"""
                if self.args.features == 'MS':
                    f_dim = -1 
                else:
                    f_dim = 0
                print("f_dim should be 0:", f_dim)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                print("Model outputs (8,10, 2 ):", outputs.shape)
                print("batch y (8, 10 ):", batch_y.shape)
                batch_y = batch_y[:, -self.args.pred_len:].view(-1) 
#-------------------------------------------------- （batch, sl, label dimension）-> (all batches x sl, label dimension)
                """ print("list all y:", batch_y) """
                #batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
#-------------------------------------------------- 计算 MSE 删除，更新计算 cross entropy
                
                outputs = outputs.view(-1, outputs.size(-1))  # [batch_size * pred_len, num_classes]
                """print("outputs finally:", outputs)"""
                loss = criterion(outputs, batch_y.long())  # CrossEntropy需要 long 
                train_loss.append(loss.item())
                print("loss:", loss)
                print("train_loss:", train_loss)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward() # backpropagation - 反向传播 - 用 loss 计算梯度
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) - 防止梯度过大
                model_optim.step() # 更新参数来minimize loss

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint1.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = vali_loader.dataset.last_insample_window()
        print("1. vali 获取的 x shape", x.shape)
        y = vali_loader.dataset.data_y  # y should hold the true labels for validation
        print("1. vali 获取的 y shape", y.shape)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        print("2. squezze 之前的 x shape （bs,sl,2）", x.shape)


        self.model.eval()
        with torch.no_grad():
            print("Shape of x just before unpacking:", x.shape)
            B, _, C = x.shape
            outputs = torch.zeros((B, self.args.num_classes)).float().to(self.device)  # Adjust output shape for classification
            
            # Split the input into manageable batches
            id_list = np.arange(0, B, 500)  # Divide into chunks if B is large
            id_list = np.append(id_list, B)

            for i in range(len(id_list) - 1):
                batch_output = self.model(x[id_list[i]:id_list[i + 1]], None, None, None)
                outputs[id_list[i]:id_list[i + 1], :] = batch_output[:, -1, :].detach()

            # Reshape outputs for CrossEntropyLoss and convert `y` to tensor
            #outputs = outputs.view(-1, outputs.size(-1))  # [batch_size * pred_len, num_classes]
            true = torch.tensor(y, dtype=torch.long).to(self.device).squeeze()
            #Flatten `y` with the same shape as `outputs` [batch_size * pred_len]

            print("outputs shape:", outputs.shape)  # Should be [batch_size * pred_len, num_classes]
            print("true shape:", true.shape)        # Should match [batch_size * pred_len]

            # Calculate the classification loss
            loss = criterion(outputs, true)  # CrossEntropyLoss automatically applies softmax
            print("loss for vali phase:", loss)
            
        self.model.train()
        return loss


    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = test_loader.dataset.last_insample_window()
        print("1. test 获取的 x shape", x.shape)
        print("x from last sample:", x)
        y = test_loader.dataset.data_y
        print("1. test 获取的 y shape", y.shape, "y:",y)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        print("1. vali 获取的 x shape", x.shape)
        #x = x.unsqueeze(-1)

        #------------------------修改
        x = x[:, -1, :]
        print("should be(651, 2):", x.shape, x)
        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        num_classes = self.args.num_classes
        batch_size = 7
        interval = seq_len + pred_len
        num_batches = (x.shape[0] + seq_len - 1) // seq_len  # 向上取整

        x_batches = []
        y_batches = []



        for start_idx in range(0, len(x) - interval + 1, seq_len):
            # 提取 `x_batch` 和 `y_batch`
            x_batch = x[start_idx:start_idx + seq_len].cpu()  # 强制在 CPU 上处理 numpy 转换
            y_batch = y[start_idx + seq_len:start_idx + interval]  # 同上，确保在 CPU 上
            
            # 检查并补齐 x_batch 的长度
            if x_batch.shape[0] < seq_len:
                missing_len = seq_len - x_batch.shape[0]
                borrow_start_idx = max(0, start_idx - missing_len)
                x_batch = torch.cat([x[borrow_start_idx:borrow_start_idx + missing_len].cpu(), x_batch], dim=0)
            
            # 检查并补齐 y_batch 的长度
            if y_batch.shape[0] < pred_len:
                missing_len = pred_len - y_batch.shape[0]
                borrow_start_idx = max(0, start_idx + seq_len - missing_len)
                y_batch = torch.cat([y[borrow_start_idx:borrow_start_idx + missing_len], y_batch], dim=0)
            
            # 检查形状
            assert x_batch.shape[0] == seq_len, f"x_batch shape mismatch: {x_batch.shape}"
            assert y_batch.shape[0] == pred_len, f"y_batch shape mismatch: {y_batch.shape}"
            
            # 将数据添加到批次列表中
            x_batches.append(x_batch.numpy())  # 保持 numpy 格式以便后续 stack
            y_batches.append(y_batch)

        # 将 x_batches 和 y_batches 转换为 GPU 张量
        x_batches = torch.tensor(np.stack(x_batches), dtype=torch.float32).to(self.device)  # (batch_size, 96, num_features)
        y_batches = torch.tensor(np.stack(y_batches), dtype=torch.float32).to(self.device)  # (batch_size, 10, num_features)

        print("x_batches shape:", x_batches.shape)  # (batch_size, 96, num_features)
        print("y_batches shape:", y_batches.shape)  # (batch_size, 10, num_features)

        x = x_batches
        y = y_batches

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            print("shape of x", x.shape, "x content:", x)

            # encoder - decoder
#-------------------------------------------------- 删除 decoder，没用
            
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device) # B = total batch（size x number） BS x sql x 2
            print("shape of output(after decoder)", outputs.shape) # （651， 10， 2）
            id_list = np.arange(0, B, 1) # id list = 0 ～ batch number
            print ("id_list before append:", id_list.shape, id_list)
            id_list = np.append(id_list, B) 
            print ("id_list appended:", id_list.shape, id_list)
            for i in range(len(id_list) - 1):
                batch_output = self.model(x[id_list[i]:id_list[i + 1]], None, None, None) # shape = (batch_size, pred_len, num_classes)
                outputs[id_list[i]:id_list[i + 1], :, :] = batch_output

                if id_list[i] % 1000 == 0:
                    print(id_list[i])
#-------------------------------------------------- 
            print("shape of output 2", outputs.shape, "outputs content:", outputs)
            outputs = outputs.view(-1, outputs.size(-1))
            print("shape of output 3", outputs.shape, "outputs content:", outputs)
            probabilities = torch.softmax(outputs, dim=-1)  # [batch_size * pred_len, num_classes]
            predicted_labels = probabilities.argmax(dim=-1)  # Get predicted class labels
            print(probabilities.shape, "probabilisties:", probabilities)
            print(predicted_labels.shape, "predicted label:", predicted_labels)
            # Saving predictions and probabilities            
            probabilities = probabilities.detach().cpu().numpy()
            predicted_labels = predicted_labels.detach().cpu().numpy()

            trues = y

            print(trues.shape, "trues:", trues)

            for i in range(0, min(predicted_labels.shape[0], x.shape[0], trues.shape[0]), predicted_labels.shape[0] // 10):
                print(f"Current index i: {i}")
                print(f"x.shape: {x.shape}, trues.shape: {trues.shape}, predicted_labels.shape: {predicted_labels.shape}")
                
                # 检查每次拼接前的维度
                print(f"x[i, :, 0] shape: {x[i, :, 0].shape}")
                print(f"x[i, :, 0] content: {x[i, :, 0].cpu().numpy()}")
                print(f"trues[i] shape: {trues[i].cpu().numpy().squeeze().shape}")
                print(f"trues[i] content: {trues[i].cpu().numpy().squeeze()}")
                print(f"predicted_labels[i] shape: {np.array(predicted_labels[i]).shape}")
                print(f"predicted_labels[i] content: {np.array(predicted_labels[i])}")

                # 尝试拼接
                print("Attempting concatenation...")
                gt = np.concatenate((x[i, :, 0].cpu().numpy(), trues[i].cpu().numpy().squeeze()), axis=0)
                pd = np.concatenate((x[i, :, 0].cpu().numpy(), np.array([predicted_labels[i]])), axis=0)

                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))


        print('test shape:', predicted_labels.shape)
#-------------------------------------------------- 存 csv
        print("predicted label type:",  type(predicted_labels))
        print(predicted_labels.shape)
        predicted_labels = predicted_labels.reshape(-1, self.args.pred_len)
        print("Reshaped predicted labels to (6, 10, 1):", predicted_labels.shape)
        forecasts_df = pandas.DataFrame(predicted_labels, columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = range(predicted_labels.shape[0])
        #forecasts_df.index = test_loader.dataset.ids[:predicted_labels.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        # result save
        folder_path = './m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(self.args.model)
        file_path = './m4_results/' + self.args.model + '/'
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            m4_summary = M4SummaryClassification(file_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            accuracy, f1, precision, recall = m4_summary.evaluate_classification()  # Define `evaluate_classification` accordingly
#------------------------------------------------- 符合 classification
            # Print the classification metrics
            print('Accuracy:', accuracy)
            print('F1 Score:', f1)
            print('Precision:', precision)
            print('Recall:', recall)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
            # cross entropy
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(outputs, trues.view(-1).squeeze().long())  # outputs为模型输出，trues为真实标签
            print(f'Cross-Entropy Loss: {loss.item()}')
            #Acc & Prec & Recall & ROC
            # Convert tensors to numpy arrays for metrics calculation
            y_true = trues.cpu().numpy().reshape(-1)  # Flatten to match predicted labels
            y_pred = predicted_labels.reshape(-1)  # Flatten predicted labels

            # For AUC and Brier Score, use probabilities for the positive class
            y_pred_prob = probabilities[:, 1]  # Assuming column 1 is the probability for class 1

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary')
            recall = recall_score(y_true, y_pred, average='binary')
            f1 = f1_score(y_true, y_pred, average='binary')
            auc = roc_auc_score(y_true, y_pred_prob)


            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')
            print(f'ROC-AUC: {auc:.4f}')


        return