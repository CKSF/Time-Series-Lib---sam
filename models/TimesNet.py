import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        #-------------------------------------------------------
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'short_term_forecast_classification':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.clamp(torch.var(x_enc, dim=1, keepdim=True, unbiased=False), min=1e-4))
        '''if torch.isnan(x_enc).any():
            print("NaN detected in x_enc before normalization.")
        if torch.isnan(means).any():
            print("NaN detected in means.")
        if torch.isnan(stdev).any():
            print("NaN detected in stdev.")
'''
        #---------------------------------------------- 防止 divide by 0： var -》 stdev
        near_zero_variance = stdev < 1e-4  # Threshold can be adjusted if necessary
        """if torch.any(near_zero_variance):
            print("Near-zero variance detected. Skipping normalization for affected sequences.")"""

        # Only normalize sequences that do not have near-zero variance
        x_enc = torch.where(near_zero_variance, x_enc, x_enc / stdev)
        print("x_enc shape before embedding:", x_enc.shape)


#------------------------------------debug
        """if torch.isnan(x_enc).any():
            print("NaN detected in x_enc after normalization.")"""


        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C] / 将 x_mark 嵌入 x_enc(就是那俩 channel)
        # 【BS,sl,Channel】 -> [BS, sl, d_model] 将 channel 这个2维的feature 打到 更高的维度
#------------------------------------debug
        print("After embedding:", enc_out.shape, enc_out)
        """if torch.isnan(enc_out).any():
            print("NaN detected in enc_out after embedding.")"""

        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
        #1. 所以 xenc 一开始是 [B, sl, channel_num(2)]
        #2. embed 以后变成 [bs, sl, d_model], 
        #3, 然后 permute 把 array 中的 【0， 1， 2】 变成【0， 2， 1】 = 【bs, d_model, sl】(8, 128, 96), 
        #4. 然后 linear 在 sl = 96 的基础上， 加上了 pred_l = 10; 所以shape = 【8， 128， 106】； 
        #5. 然后再 permute回【8， 106， 128】
#------------------------------------debug
        print("After predict_linear:", enc_out.shape, enc_out)
        """if torch.isnan(enc_out).any():
            print("NaN detected in enc_out after predict_linear.")"""
#------------------------------------debug

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
            # layers
#------------------------------------debug
            print(f"After TimesNet layer {i}:", enc_out.shape, enc_out)
        # Check for NaNs after each layer
            """if torch.isnan(enc_out).any():
                print(f"NaN detected in enc_out after TimesNet layer {i}.")
                break"""
        # porject back
        dec_out = torch.softmax(self.projection(enc_out), dim=-1)
        #projection 把 d_model = 128 维度的 feature 打回 2 维， 
        #然后在 （B，T，C）中的 C 上计算 softmax normalization，也就是这个 2 维的feature的每一维的概率
#------------------------------------debug
        print("After projection:", dec_out.shape, dec_out)
    # Check for NaNs in dec_out
        """if torch.isnan(dec_out).any():
            print("NaN detected in dec_out after projection.")"""
        dec_out = dec_out[:, -self.pred_len:, :]
        print("结果：", dec_out.shape, dec_out)
        # dec_out = 预测结果； 
        #1. 就是 enc_out [8,106,128]
        #2. layers 处理， 每一个layer 都更新 106 time stamp 中最后 10个数值
        #3. projection + softmax norm 打回 [8, 106, 2] = dec_out
        #4. dec_out 提取 106 中的 最后 10（pred_l）个 -> [8,10,2] = 预测结果

        # De-Normalization from Non-stationary Transformer
        '''
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        '''
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]// 
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        print("Main forward Running")
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'short_term_forecast_classification':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if self.task_name == 'short_term_forecast_classification':
                dec_out = torch.softmax(dec_out, dim=-1) 
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        
        print(f"Running task: {self.task_name}")

        return None
