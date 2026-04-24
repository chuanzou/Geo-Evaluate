import os
import numpy as np
import rasterio
from tqdm import tqdm
import argparse


def convert_tif_to_npy(input_dir, output_dir):
    """Convert Sentinel-2 GeoTIFF patches to (4, 128, 128) float32 npy files
    in the [0, 1] reflectance range.

    ``download_sentinel2_beijing.py`` already divides by 10000 on the Earth
    Engine side, so the exported TIFs are already in [0, 1].  An older
    version of this script divided by 10000 a *second* time, silently
    collapsing every value to ~1e-5 which became 0.0 after float32
    round-trip + clip — that's what caused "perfect" GAN metrics earlier.
    We now auto-detect the input range:

      * if max > 2   → assume raw DN units, divide by 10000
      * else         → already reflectance, leave alone

    Any NaN/Inf is zeroed out so downstream training never sees non-finite.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    if not files:
        print(f"错误: 在 {input_dir} 中没找到任何 .tif 文件！")
        return

    print(f"正在开始转换 {len(files)} 个文件...")

    scaled_count = 0
    already_ok_count = 0
    zero_samples = []

    for f in tqdm(files):
        try:
            with rasterio.open(os.path.join(input_dir, f)) as src:
                # B4, B3, B2, B8  →  Red, Green, Blue, NIR
                img = src.read([1, 2, 3, 4]).astype(np.float32)

                # 替换可能的 NaN/Inf（EE 导出时被 mask 掉的像素）
                img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

                # 自适应归一化：判断 TIF 是原始 DN 还是已经是反射率
                max_val = float(img.max())
                if max_val > 2.0:
                    img = img / 10000.0
                    scaled_count += 1
                else:
                    already_ok_count += 1

                img = np.clip(img, 0.0, 1.0)
                # 强制裁剪到 128×128（只取左上角）
                img = img[:, :128, :128]

                if img.shape != (4, 128, 128):
                    print(f"警告: {f} 转换后形状为 {img.shape}，跳过")
                    continue

                if img.max() == 0.0:
                    zero_samples.append(f)

                output_filename = f.replace('.tif', '.npy')
                np.save(os.path.join(output_dir, output_filename), img)
        except Exception as e:
            print(f"处理文件 {f} 时出错: {e}")

    print(
        f"\n完成: /10000 归一化 {scaled_count} 张，不需缩放 {already_ok_count} 张，"
        f"全 0 样本 {len(zero_samples)} 张"
    )
    if zero_samples[:5]:
        print("全 0 样本举例（可能是 EE 全 mask 的空 patch）:", zero_samples[:5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 GeoTIFF 转换为 NumPy 格式")
    parser.add_argument("--input-dir", type=str, required=True, help="输入 TIFF 文件的路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出 NPY 文件的路径")

    args = parser.parse_args()
    convert_tif_to_npy(args.input_dir, args.output_dir)
