import torch
from monai.utils import look_up_option

from encoder.swin_unter import PatchMerging, PatchMergingV2

MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


def get_encoder(cfg):
    # print(cfg.MODEL.ENCODER)
    if cfg.MODEL.ENCODER.lower() == 'swin':
        from encoder.swin_unter import SwinTransformer
        backbone = SwinTransformer(
            in_chans=cfg.MODEL.IN_CHANNEL,
            embed_dim=cfg.MODEL.FEATURES_SIZE,
            window_size=cfg.MODEL.window_size,
            patch_size=cfg.MODEL.patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=cfg.TRAIN.DROP_RATE,
            attn_drop_rate=0.0,
            drop_path_rate=cfg.MODEL.DROPOUT_RATE,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=cfg.MODEL.CHECKPOINT,
            spatial_dims=cfg.MODEL.SPATIAL_DIMS,
            downsample=look_up_option("merging", MERGING_MODE) if isinstance("merging", str) else "merging",
            use_v2=False
        )
        # pretrained_dict = torch.load(cfg.TRAIN.PRETRAINED_SWIN_PATH)["model"]
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict()}
        # backbone.load_state_dict(pretrained_dict)
        channel_list = [48, 96, 192, 384, 768]
    # elif cfg.MODEL.ENCODER.lower() == 'resnet':
    #     from encoder.resnet import ResNet50Backbone
    #     backbone = ResNet50Backbone()
    #     channel_list = [256, 512, 1024, 2048]
    # elif cfg.MODEL.ENCODER.lower() == 'pvt':
    #     from encoder.pvt import pvt_v2_b5
    #     backbone = pvt_v2_b5()
    #     channel_list = [64, 128, 320, 512]
    # pvt_model_dict = backbone.state_dict()
    # pretrained_state_dicts = torch.load(cfg.TRAIN.PRETRAINED_PVTV2_PATH)
    # state_dict = {k : v for k, v in pretrained_state_dicts.items() if k in pvt_model_dict.keys()}
    # pvt_model_dict.update(state_dict)
    # backbone.load_state_dict(pvt_model_dict)

    return backbone, channel_list
