import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8SegmentationLoss

class TemporalLoss(v8SegmentationLoss):
    """
    时序损失函数：二分类版本
    在标准分割损失基础上添加翻滚二分类损失
    """
    
    def __init__(self, model):
        # 1. 初始化父类 (v8SegmentationLoss)
        super().__init__(model)
        
        # 2. 获取超参数 (roll_weight 控制翻滚损失的总比重)
        self.roll_weight = getattr(model.args, 'roll', 1.0) if hasattr(model, 'args') else 1.0
        
        # 3. 🚀 核心修复：使用带权重的二分类损失
        # pos_weight=5.0 意味着：如果漏掉一个“翻滚”样本，惩罚是普通样本的 5 倍。
        # 如果你的数据中翻滚样本极少，建议将此值加大到 10.0 甚至 20.0
        device = next(model.parameters()).device
        self.roll_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], device=device))
    
    def __call__(self, preds, batch):
        """
        计算总损失
        """
        roll_pred = None
        
        # 1. 安全解包预测结果 (兼容训练与验证模式)
        if isinstance(preds, (list, tuple)):
            if len(preds) == 2 and isinstance(preds[1], (list, tuple)) and isinstance(preds[0], torch.Tensor):
                # 验证模式
                standard_preds = preds[1]
                roll_pred = None 
            elif len(preds) == 2:
                # 训练模式：(standard_preds, roll_pred)
                standard_preds, roll_pred = preds
            else:
                standard_preds = preds[:-1]
                roll_pred = preds[-1]
        else:
            standard_preds = preds

        # 2. 计算标准检测损失 (Box, Seg, Cls, DFL)
        loss, loss_items = super().__call__(standard_preds, batch)
        
        # 3. 计算翻滚二分类损失
        roll_targets = batch.get('roll_gt', None)
        roll_loss_val = torch.tensor(0.0, device=loss.device)
        
        if roll_targets is not None and roll_pred is not None:
            # 🛡️ 形状对齐：确保 roll_pred 和 roll_targets 都是 [B, 1] 且为 float 类型
            if roll_pred.ndim == 4: # 如果是 [B, 1, H, W] 这种未池化的特征
                roll_pred = roll_pred.mean(dim=[-1, -2])
            
            # 强制对齐形状并将标签转为 float
            roll_targets = roll_targets.view_as(roll_pred).float()
            


            # 🚀 升级版探针：全局正样本计数器
            if not hasattr(self, '_pos_counter'):
                self._pos_counter = 0
                self._batch_counter = 0

                target_list = roll_targets.view(-1).tolist()
                self._pos_counter += sum(target_list)
                self._batch_counter += 1

        # 假设 120 张图，batch=5，大概需要 24 个 batch 跑完一个 Epoch
        # 我们在接近 Epoch 末尾的时候打印一次总计
            if self._batch_counter == 24:
                print("\n" + "="*50)
                (f"🕵️ [全局标签探针] 第 1 个 Epoch 数据集扫描完毕！")
                print(f"   ✅ 累计喂给模型的正样本(翻滚)数量: {self._pos_counter}")
                if self._pos_counter == 0:
                    print("   ❌ 严重警告：整个 Epoch 跑完都没看到一个 1，你的标签文件确实出问题了！")
                print("="*50 + "\n")

            
            # 计算损失
            r_loss = self.roll_loss_fn(roll_pred, roll_targets)
            
            # 加权并累加到总损失
            loss += self.roll_weight * r_loss
            roll_loss_val = r_loss.detach()

        # 4. 更新 loss_items (顺序：box, seg, cls, dfl, roll)
        loss_items = torch.cat([loss_items, roll_loss_val.view(1)])
        
        return loss, loss_items