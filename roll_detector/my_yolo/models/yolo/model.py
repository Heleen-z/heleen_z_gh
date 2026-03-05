import torch
import torch.nn as nn
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.tasks import SegmentationModel

class TemporalRollHead(nn.Module):
    def __init__(self, in_channels=256, t_window=5):
        super().__init__()
        self.t = t_window
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_seq = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * t_window, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1) 
        )

    def forward(self, x):
        bt, c, h, w = x.shape
        if bt % self.t != 0:
            # 🚨 增加高亮警告：如果真出维度问题，必须大声喊出来，绝对不能静默 return None!
            print(f"\n🚨 [警告] 时序维度异常: 批次 {bt} 无法被窗口 {self.t} 整除! 强制放行防崩溃。")
            b = max(1, bt // self.t)
            return torch.zeros((b, 1), device=x.device, requires_grad=True)
            
        b = bt // self.t
        x = self.pool(x).view(bt, c)
        x = x.view(b, self.t, c).transpose(1, 2)
        return self.conv_seq(x)

class ROLL_SegmentationModel(SegmentationModel):
    def __init__(self, cfg='yolo11n-seg.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        self.roll_head = TemporalRollHead(in_channels=256, t_window=5) 
        # 彻底删除了脆弱的 PyTorch Hook 代码

    def init_criterion(self):
        from my_yolo.utils.losses import TemporalLoss
        return TemporalLoss(self)
        
    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        if not hasattr(self, 'roll_head'):
            return super()._predict_once(x, profile, visualize, embed)

        # 🚀 修正 1：入口维度处理（解决推理时的 5D 报错）
        # 如果收到 [1, 5, 3, 640, 640]，将其展平为 [5, 3, 640, 640]
        is_5d = (x.ndim == 5)
        if is_5d:
            original_shape = x.shape # 暂存 [B, T, C, H, W]
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])

        y, dt = [], [] 
        roll_feat = None
        t_window = self.roll_head.t
        
        for i, m in enumerate(self.model):
            if m.f != -1: 
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            # 🚀 修正 2：分割头保护逻辑（你之前的切片逻辑）
            if i == len(self.model) - 1:
                # 检查当前特征的 Batch 是否是 5 的倍数
                curr_bt = x[0].shape[0] if isinstance(x, list) else x.shape[0]
                if curr_bt >= t_window:
                    target_indices = torch.arange(t_window - 1, curr_bt, t_window, device=x[0].device if isinstance(x, list) else x.device)
                    if isinstance(x, list):
                        x = [feat[target_indices] for feat in x]
                    else:
                        x = x[target_indices]

            x = m(x) 
            y.append(x if m.i in self.save else None) 
            
            # 截取第 22 层特征
            if i == 22:
                roll_feat = x
                self.roll_feat_debug = x.detach()    #read

        # 🚀 修正 3：调用翻滚头
        # 此时 roll_feat 是 [B*5, C, H, W]，进入你的 TemporalRollHead.forward
        # 你的 forward 内部会自己做 view(b, t, c)，所以这里直接传就行
        roll_pred = self.roll_head(roll_feat)
        
        if self.training:
            return x, roll_pred
        else:
            self.roll_pred = roll_pred
            return x

class ROLL_YOLO(YOLO):
    @property
    def task_map(self):
        from my_yolo.engine.trainer import TemporalSegmentationTrainer
        from my_yolo.engine.validator import TemporalSegmentationValidator
        from ultralytics.models.yolo.segment.predict import SegmentationPredictor
        return {
            'segment': {
                'model': ROLL_SegmentationModel,
                'trainer': TemporalSegmentationTrainer,
                'validator': TemporalSegmentationValidator,
                'predictor': SegmentationPredictor,
            }
        }