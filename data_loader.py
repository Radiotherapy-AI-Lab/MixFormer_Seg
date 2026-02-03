def get_loader_data_622(args,
                        csv_path,
                        json_path,
                        save_split_path=None, load_split_path=None, test=None):
    datalist, val_files, test_files = get_site_data_train_val_test(csv_path, json_path, save_split_path=save_split_path,
                                                                   load_split_path=load_split_path, random_state=42)
    print(json_path)
    print(f"Total training samples: {len(datalist)}")
    print(f"Total validation samples: {len(val_files)}")
    print(f"Total test samples: {len(test_files)}")

    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        ScaleIntensityRanged, ScaleIntensityd, ToTensord,
        RandFlipd, RandRotated, RandZoomd, RandAdjustContrastd,
        RandGaussianNoised, RandGaussianSmoothd, RandShiftIntensityd,
        RandCoarseDropoutd, RandSpatialCropd, Resized
    )



    train_transforms = Compose(
        [
            # 1. 基础加载和预处理
            LoadImaged(keys=["image", "mask"], reader="ITKReader"),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),

            # 2. CT值特定强度调整
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True
            ),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),

            # 3. 几何空间变换 (保持图像-标签一致性)
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),  # 水平翻转
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),  # 垂直翻转
            RandRotated(
                keys=["image", "mask"],
                range_x=aug_params["rotation"],
                range_y=aug_params["rotation"],
                range_z=0,  # Z轴通常不旋转
                prob=0.8,
                mode=("bilinear", "nearest"),  # 图像双线性插值，标签最近邻
            ),

            ToTensord(keys=["image", "mask", "text"])
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            ToTensord(keys=["image", "mask", "text"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True
            ),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            ToTensord(keys=["image", "mask", "text"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True
            ),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)

        ]
    )



    train_ds = Dataset(data=datalist, transform=train_transforms)

    # # 分布式训练设置
    # if args.distributed:
    #     train_sampler = DistributedSampler(
    #         dataset=train_ds,
    #         even_divisible=True,
    #         shuffle=True
    #     )
    # else:
    train_sampler = None


    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=2,
        sampler=train_sampler,
        drop_last=True
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    test_ds = Dataset(data=test_files, transform=test_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
        drop_last=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
        drop_last=True
    )

    return train_loader, val_loader, test_loader
