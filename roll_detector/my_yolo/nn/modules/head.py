"""
自定义Segment检测头，添加翻滚预测分支
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Union
from ultralytics.nn.modules.head import Segment
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.tal import make_anchors


class CustomSegment(Segment):
    """
    自定义Segment检测头，添加翻滚预测分支
    
    新增属性:
        cv_roll (nn.ModuleList): 翻滚预测的卷积层
    """
    
    def __init__(self, nc=80, nm=32, npr=256, ch=(), **kwargs):
        """
        Args:
            nc: 类别数
            nm: mask原型数量
            npr: mask原型分辨率
            ch: 输入通道数列表
            **kwargs: 额外参数
        """
        super().__init__(nc, nm, npr, ch, **kwargs)
        
        # 基础参数
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # 标准输出数 (cls + box)
        
        # ========== 翻滚预测分支 ==========
        # 计算通道数
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        
        # 翻滚预测卷积层 (每个检测层一个)
        # 输出3类: stable(0), suspect(1), roll(2)
        self.cv_roll = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 3, 1)  # 3类翻滚状态
            ) for x in ch
        )
    
    def forward(self, x: List[torch.Tensor]) -> Union[Tuple, List[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 多尺度特征图列表
            
        Returns:
            训练模式: (检测输出, mask系数, mask原型, 翻滚预测)
            推理模式: (拼接输出, (检测输出, mask系数, mask原型, 翻滚预测))
        """
        # 提取mask原型
        p = self.proto(x[0])
        bs = p.shape[0]
        
        # 计算mask系数
        mc = torch.cat([
            self.cv4[i](x[i]).view(bs, self.nm, -1) 
            for i in range(self.nl)
        ], 2)
        
        # ========== 检测分支 (标准YOLO) ==========
        detection_out = []
        roll_out = []
        
        for i in range(self.nl):
            # 标准检测输出 (box + cls)
            det = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            detection_out.append(det)
            
            # 翻滚预测输出
            roll = self.cv_roll[i](x[i])
            roll_out.append(roll)
        
        # ========== 输出处理 ==========
        if self.training:
            # 训练模式: 返回所有输出
            return detection_out, mc, p, roll_out
        else:
            # 推理模式: 解码并合并输出
            y = self._inference(detection_out)
            roll_pred = self._process_roll_predictions(roll_out)
            
            if self.export:
                return torch.cat([y, mc], 1), roll_pred
            else:
                return (torch.cat([y[0], mc], 1), (y[1], mc, p, roll_pred))
    
    def _inference(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        解码预测结果 (推理模式)
        
        Args:
            x: 检测输出列表
            
        Returns:
            解码后的预测结果
        """
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape
        
        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        
        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)
    
    def _process_roll_predictions(self, roll_out: List[torch.Tensor]) -> torch.Tensor:
        """处理翻滚预测输出"""
        # 合并多尺度翻滚预测
        roll_cat = []
        for roll in roll_out:
            # 变形为 [B, 3, H*W]
            B, _, H, W = roll.shape
            roll_cat.append(roll.view(B, 3, -1))
        
        # 在通道维度拼接
        return torch.cat(roll_cat, dim=2).permute(0, 2, 1)  # [B, N_anchors, 3]
