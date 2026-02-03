class HeadNeck_Swin_Res2Net(nn.Module):
    

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encode, self.filters = get_encoder(self.cfg)

        self.res2net3d = res2net50_3d(num_classes=self.cfg.MODEL.NUM_CLASSES, dim=self.cfg.MODEL.DIM,
                                      expansion=self.cfg.MODEL.EXPANSION)
        # 解码器路径
        self.up1 = UpsamplingBlock(
            in_channels=self.filters[4],
            skip_channels=self.filters[3],  
            out_channels=self.filters[3]
        )
        self.up2 = UpsamplingBlock(
            in_channels=self.filters[3],
            skip_channels=self.filters[2],
            out_channels=self.filters[2]
        )
        self.up3 = UpsamplingBlock(
            in_channels=self.filters[2],
            skip_channels=self.filters[1],
            out_channels=self.filters[1]
        )
        self.up4 = UpsamplingBlock(
            in_channels=self.filters[1],
            skip_channels=self.filters[0],
            out_channels=self.filters[0]
        )
        self.up5 = UpsamplingBlock(
            in_channels=self.filters[0],
            skip_channels=1,
            out_channels=1
        )

        # self.fusion4 = TPAVIModule(self.filters[4], self.text_len)
        # 输出层
        self.final_conv = nn.Conv3d(2, self.cfg.MODEL.OUT_CHANNELS, kernel_size=1)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.InstanceNorm3d) and module.weight is not None:
            nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, text=None, y=None):
        loss = torch.tensor(0.0, device='cuda')
        fs = self.res2net3d(x)
        features = self.encode(x, fs)

        s0, s1, s2, s3, s4 = features

        # 解码器路径（使用跳跃连接）
        d4 = self.up1(s4, s3)
        d3 = self.up2(d4, s2)
        d2 = self.up3(d3, s1)
        d1 = self.up4(d2, s0)
        d0 = self.up5(d1, x)

        # 输出层
        logits = self.final_conv(d0)
        return logits, loss

    def predict(self, x):

        with torch.no_grad():
            logits = self.forward(x)
            if self.final_conv.out_channels > 1:
                return F.softmax(logits, dim=1)
            else:
                return torch.sigmoid(logits)
