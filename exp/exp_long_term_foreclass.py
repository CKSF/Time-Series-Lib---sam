from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
warnings.filterwarnings('ignore')


class Exp_Long_Term_Foreclass(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Foreclass, self).__init__(args)
        print("before define self device:Using device:", self.device)
        self.device = torch.device(f"cuda:{args.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu")
        print("after define self device:Using device:", self.device)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids).to(self.device)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='CrossEntropy'): #改 默认 criterion
        """print("before define weight Using device:", self.device)

        count_0 = 1647
        count_1 = 297
        class_weights = torch.tensor([1.0, count_0 / count_1], dtype=torch.float).to(self.device)
        print("after define weight Using device:", self.device)
        print("Class weights:", class_weights)"""
#-------------------------------------------------
        if loss_name == 'CrossEntropy':
            
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                #batch_x_mark = batch_x_mark.float().to(self.device)
                #batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = self.model(batch_x, None, dec_inp, None)
                else:
                    #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = self.model(batch_x, None, dec_inp, None)
                    print("print from forclass vali - outputshape:", outputs.shape, outputs)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:].view(-1)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu().squeeze()
                pred = pred.reshape(-1, pred.size(-1))
                true = true.long()
                print("pred shape:", pred.shape)
                print("true shape:", true.shape)

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

            

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            for i, (batch_x, batch_y) in enumerate(train_loader):
                if batch_x is None or batch_y is None:
                    continue
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                '''
                if hasattr(self, 'batch_x_mark') and hasattr(self, 'batch_y_mark'):
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                '''
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = self.model(batch_x, None, dec_inp, None)
                        
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].view(-1)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = self.model(batch_x, None, dec_inp, None)
                    print("print from foreclass train - outputs:", outputs.shape, outputs)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    print("outputs 刚读完数据：",outputs.shape)
                    batch_y = batch_y[:, -self.args.pred_len:].view(-1)
                    print("y 变形 - 对应 output变形",batch_y.shape)

                    outputs = outputs.reshape(-1, outputs.size(-1)) # 添加的
                    print("output shape 变形:",outputs.shape, outputs)
                    loss = criterion(outputs, batch_y.long())
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            '''
            if early_stopping.early_stop:
                print("Early stopping")
                break
            '''

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                #batch_x_mark = batch_x_mark.float().to(self.device)
                #batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = self.model(batch_x, None, dec_inp, None)
                else:
                    outputs = self.model(batch_x, None, dec_inp, None)
                    #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:].view(-1)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                print("我看看怎么个事1？", outputs.shape, outputs)
                batch_y = batch_y.reshape(-1, 1)
                batch_y = batch_y[:, :]
                print("我看看怎么个事 batch y？", batch_y.shape, batch_y)

                pred = outputs
                true = batch_y
                pred = pred.reshape(-1, 2)  # 将 pred 转换为 [40, 2]

                preds.append(pred)
                trues.append(true)
                print("shape of pred", pred.shape, pred)
                print("shape of true", true.shape, true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[:, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[:, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        np.set_printoptions(threshold=np.inf)
        print('test shape:', preds.shape, trues.shape, preds, trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape, preds, trues)
        np.set_printoptions(threshold=1000)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'not calculated'
            

        """mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)"""
        preds = np.argmax(preds, axis=-1)  # 将 (5460, 2) 转换为 (5460,)
        preds = preds.squeeze()  # 将 (1, 5460) 转换为 (5460,)
        trues = trues.squeeze() 

        print("finally preds:", preds,"finally trues:", trues)  
        print("Shape of preds:", preds.shape)
        print("Shape of trues:", trues.shape)
        precision = precision_score(trues, preds, average='weighted')
        recall = recall_score(trues, preds, average='weighted')
        f1 = f1_score(trues, preds, average='weighted')
        auc_score = roc_auc_score(trues, preds, average='weighted', multi_class='ovr')  # 若为多分类任务
        fpr, tpr, _ = roc_curve(trues, preds, pos_label=1)  # 若为二分类任务
        ouc_score = auc(fpr, tpr) if len(np.unique(trues)) == 2 else "N/A for multi-class"

        # 打印输出
        print('precision:{}, recall:{}, f1:{}, auc:{}, ouc:{}'.format(precision, recall, f1, auc_score, ouc_score))

        # 写入结果文件
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('precision:{}, recall:{}, f1:{}, auc:{}, ouc:{}'.format(precision, recall, f1, auc_score, ouc_score))
            f.write('\n\n')

        # 保存结果
        np.save(folder_path + 'metrics.npy', np.array([precision, recall, f1, auc_score, ouc_score]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

