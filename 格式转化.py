import SimpleITK as sitk


# 原始数据图像
raw_path  = "./DATASET_Acdc/unetr_pp_raw/unetr_pp_raw_data/Task001_ACDC/imagesTr/patient002_frame12_0000.nii.gz"

# mask_path 输出的尺寸有误的结果（建议 pp 即后处理过的）
mask_path = "./chuli/pred_pp/patient002_frame12.nii.gz"

out_path  = "./chuli/pred_pp/patient002_frame12_match_rawHeader.nii.gz"

raw  = sitk.ReadImage(raw_path)  # 读取原图像的信息
mask = sitk.ReadImage(mask_path)

# mask统一成和raw一样的属性
mask.SetSpacing(raw.GetSpacing())  # 间距
mask.SetOrigin(raw.GetOrigin())   # 原点
mask.SetDirection(raw.GetDirection())  # 方向

sitk.WriteImage(mask, out_path)
print("written:", out_path)
print("raw spacing:", raw.GetSpacing())
print("mask spacing(before->after):", sitk.ReadImage(mask_path).GetSpacing(), "->", sitk.ReadImage(out_path).GetSpacing())