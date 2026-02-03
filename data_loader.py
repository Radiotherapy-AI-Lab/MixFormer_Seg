def get_loader_data_data_en(args,
                            csv_path=r"C:\Users\ta\Desktop\Swin-UNETR-main\data_catulate\mask_DS_stratified_processed.csv",
                            json_path=r"C:\Users\ta\Desktop\Swin-UNETR-main\data_catulate\medical_cases_en.json",
                            save_split_path=None, load_split_path=None, test=None):
    _, val_files = get_site_data_train_val(csv_path, json_path, save_split_path=save_split_path,
                                           load_split_path=load_split_path, random_state=42)
    print(json_path)
    print(f"Total validation samples: {len(val_files)}")

    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        ScaleIntensityRanged, ScaleIntensityd, ToTensord,
        RandFlipd, RandRotated, RandAffined, RandAdjustContrastd,
        RandZoomd
    )
    import numpy as np

    # 基础验证变换（无增强）
    base_val_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"], reader="ITKReader"),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True
            ),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            ToTensord(keys=["image", "mask", "text"])
        ]
    )

    # 增强变换列表（每种增强对应一个变换）
    augmentation_transforms = []

    # 1. 原始图像（无增强）
    augmentation_transforms.append(
        Compose([])  # 空变换，保留原始图像
    )

    # 2. 水平翻转
    augmentation_transforms.append(
        Compose([
            RandFlipd(keys=["image", "mask"], prob=1.0, spatial_axis=0)
        ])
    )

    # 3. 垂直翻转
    augmentation_transforms.append(
        Compose([
            RandFlipd(keys=["image", "mask"], prob=1.0, spatial_axis=1)
        ])
    )

    # 4. 旋转90度
    augmentation_transforms.append(
        Compose([
            RandRotated(
                keys=["image", "mask"],
                range_x=np.pi / 2,  # 90度
                range_y=np.pi / 2,
                range_z=0,
                prob=1.0,
                mode=("bilinear", "nearest")
            )
        ])
    )

    # 5. 旋转180度
    augmentation_transforms.append(
        Compose([
            RandRotated(
                keys=["image", "mask"],
                range_x=np.pi,  # 180度
                range_y=np.pi,
                range_z=0,
                prob=1.0,
                mode=("bilinear", "nearest")
            )
        ])
    )

    # 6. 旋转270度
    augmentation_transforms.append(
        Compose([
            RandRotated(
                keys=["image", "mask"],
                range_x=3 * np.pi / 2,  # 270度
                range_y=3 * np.pi / 2,
                range_z=0,
                prob=1.0,
                mode=("bilinear", "nearest")
            )
        ])
    )

    # 7. 小幅仿射变换
    augmentation_transforms.append(
        Compose([
            RandAffined(
                keys=["image", "mask"],
                rotate_range=(np.pi / 36, np.pi / 36),  # ±5度
                shear_range=(0.05, 0.05),
                translate_range=(0.05, 0.05),
                scale_range=(0.05, 0.05),
                mode=("bilinear", "nearest"),
                prob=1.0
            )
        ])
    )

    # 8. 亮度对比度调整
    augmentation_transforms.append(
        Compose([
            RandAdjustContrastd(keys="image", gamma=(0.8, 1.2), prob=1.0)
        ])
    )

    # 9. 多尺度缩放0.9
    augmentation_transforms.append(
        Compose([
            RandZoomd(
                keys=["image", "mask"],
                min_zoom=0.9,
                max_zoom=0.9,
                prob=1.0,
                mode=("area", "nearest")
            )
        ])
    )

    # 10. 多尺度缩放1.0（等同于原始尺寸）
    augmentation_transforms.append(
        Compose([
            RandZoomd(
                keys=["image", "mask"],
                min_zoom=1.0,
                max_zoom=1.0,
                prob=1.0,
                mode=("area", "nearest")
            )
        ])
    )

    # 11. 多尺度缩放1.1
    augmentation_transforms.append(
        Compose([
            RandZoomd(
                keys=["image", "mask"],
                min_zoom=1.1,
                max_zoom=1.1,
                prob=1.0,
                mode=("area", "nearest")
            )
        ])
    )

    # 创建增强后的验证数据集
    augmented_val_data = []

    for sample in val_files:
        # 应用基础变换
        base_sample = base_val_transforms(sample.copy())

        # 应用每种增强变换
        for i, aug_transform in enumerate(augmentation_transforms):
            # 复制基础样本
            aug_sample = {k: v.clone() if hasattr(v, 'clone') else v.copy()
                          for k, v in base_sample.items()}

            # 应用特定增强
            aug_sample = aug_transform(aug_sample)

            # 添加增强标识
            aug_sample["augmentation_type"] = i
            aug_sample["original_id"] = sample.get("id", "unknown")

            augmented_val_data.append(aug_sample)

    print(f"Total augmented validation samples: {len(augmented_val_data)}")

    # 创建验证数据集
    val_ds = Dataset(data=augmented_val_data)

    # 创建验证数据加载器
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False
    )

    # 只返回验证数据加载器（训练数据设为None）
    return None, val_loader
