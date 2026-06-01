import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, temperature=0.07, scale_by_temperature=True, scale_by_weight=False):
        super(Loss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.scale_by_weight = scale_by_weight

    def forward(self, out, mask):
        device = out.device
        

        row, col, val = mask.storage.row(), mask.storage.col(), mask.storage.value()
        row, col, val = row.to(device), col.to(device), val.to(device)
        batch_size = out.shape[0]

        # compute logits
        dot = torch.matmul(out, out.T)
        dot = torch.div(dot, self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(dot, dim=1, keepdim=True)
        dot = dot - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones(batch_size, batch_size).to(device),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        exp_logits = torch.exp(dot) * logits_mask
        log_probs = dot - torch.log(exp_logits.sum(1, keepdim=True))

        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        labels = row.view(row.shape[0], 1)
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        log_probs = log_probs[row, col]
        log_probs = log_probs.view(-1, 1)
        loss = torch.zeros_like(unique_labels, dtype=torch.float).to(device)
        loss.scatter_add_(0, labels, log_probs)
        loss = -1 * loss / labels_count.float().unsqueeze(1)

        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss

class ChunkedLoss(nn.Module):
    def __init__(self, temperature=0.07, scale_by_temperature=True, chunk_size=20000):
        super(ChunkedLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.chunk_size = chunk_size  # 每次处理的行数，根据显存调整

    def forward(self, out, mask):
        device = out.device
        batch_size = out.shape[0]
        
        # 1. 获取稀疏 mask 的坐标和值
        row, col = mask.storage.row().to(device), mask.storage.col().to(device)
        
        total_loss = 0.0
        total_count = 0

        # 2. 按行进行分块计算
        for i in range(0, batch_size, self.chunk_size):
            end_i = min(i + self.chunk_size, batch_size)
            chunk_out = out[i:end_i]  # [chunk_size, dim]
            
            # 计算当前块与全局的点积: [chunk_size, batch_size]
            dot_chunk = torch.matmul(chunk_out, out.T) / self.temperature
            
            # 数值稳定性
            logits_max, _ = torch.max(dot_chunk, dim=1, keepdim=True)
            dot_chunk = dot_chunk - logits_max.detach()
            
            # 构造当前块的对角线 mask (去除自对比)
            # 当前块的全局索引是 range(i, end_i)
            chunk_size_actual = end_i - i
            diag_indices = torch.arange(i, end_i, device=device).view(-1, 1)
            logits_mask_chunk = torch.ones(chunk_size_actual, batch_size, device=device)
            logits_mask_chunk.scatter_(1, diag_indices, 0)
            
            # 计算分母的分母项
            exp_logits_chunk = torch.exp(dot_chunk) * logits_mask_chunk
            sum_exp = exp_logits_chunk.sum(1, keepdim=True)
            
            # 当前 chunk 的所有 log_probs: [chunk_size, batch_size]
            log_probs_chunk = dot_chunk - torch.log(sum_exp)
            
            # 3. 筛选出落在当前 chunk 行范围内的 mask 边
            mask_indices = (row >= i) & (row < end_i)
            if not torch.any(mask_indices):
                continue
                
            chunk_row = row[mask_indices] - i  # 映射回当前 chunk 的局部行索引
            chunk_col = col[mask_indices]
            
            # 提取正样本对的 log_probs
            selected_log_probs = log_probs_chunk[chunk_row, chunk_col]
            
            # 4. 局部聚合损失
            unique_labels, labels_count = chunk_row.unique(return_counts=True)
            loss_chunk = torch.zeros(len(unique_labels), device=device)
            loss_chunk.scatter_add_(0, chunk_row, selected_log_probs)
            loss_chunk = -1 * loss_chunk / labels_count.float()
            
            total_loss += loss_chunk.sum()
            total_count += len(unique_labels)
            
        # 5. 最终平均
        final_loss = total_loss / max(total_count, 1)
        if self.scale_by_temperature:
            final_loss *= self.temperature
            
        return final_loss