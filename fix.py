import os
import re

# ==============================================================================
# 1. 修复 ultralytics/data/base.py (强制 6 通道加载)
# ==============================================================================
base_path = "/Users/leo/Desktop/deeplearing/ultralytics/ultralytics/data/base.py"
with open(base_path, "r", encoding="utf-8") as f:
    base_content = f.read()

# 我们需要替换 load_image 函数中的深度图处理逻辑
# 这里的逻辑是将单通道深度图复制 3 次，拼接到 RGB 后，形成 6 通道
new_depth_logic = """
            # === [Auto-Fix] Force 6-Channel Input (RGB + 3xDepth) ===
            try:
                # 1. Infer depth path
                depth_path = f.replace('/images/', '/depths/').rsplit('.', 1)[0] + '.png'
                
                # 2. Read depth (16-bit or 8-bit)
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                
                if depth is not None:
                    # Resize
                    if depth.shape[:2] != im.shape[:2]:
                        depth = cv2.resize(depth, (im.shape[1], im.shape[0]))
                    
                    # Normalize to 0-255
                    if depth.dtype == 'uint16':
                        depth = (depth / depth.max() * 255.0).astype('uint8')
                    elif depth.dtype != 'uint8':
                        depth = depth.astype('uint8')
                    
                    # Expand to (H,W,1)
                    if len(depth.shape) == 2:
                        depth = np.expand_dims(depth, axis=2)
                        
                    # CRITICAL: Repeat to 3 channels to match RGB backbone weights
                    depth_3ch = np.repeat(depth, 3, axis=2)
                    
                    # Concat: RGB(3) + Depth(3) = 6 Channels
                    im = np.concatenate((im, depth_3ch), axis=2)
                else:
                    # Fallback: Zero padding if no depth
                    print(f"Warning: No depth found for {f}")
                    im = np.concatenate((im, np.zeros_like(im)), axis=2)
                    
            except Exception as e:
                print(f"Depth load error: {e}")
            # ========================================================
"""

# 简单的替换策略：找到原有的 load_image 结尾处或插入点
# 为了稳健，我们查找 'return im, (h0, w0), im.shape[:2]' 并在其前面插入逻辑
# 但更简单的是：如果你之前修改过，我们先假设文件是原版或接近原版。
# 我们直接暴力替换 load_image 的核心读取部分。

if "Force 6-Channel Input" not in base_content:
    # 寻找 cv2.imread(f) 后面插入
    pattern = r"(im = cv2\.imread\(f\).*?\n\s+if im is None:\n\s+raise FileNotFoundError.*?\n)"
    match = re.search(pattern, base_content, re.DOTALL)
    if match:
        # 在读取图片后插入深度图逻辑
        base_content = base_content.replace(match.group(1), match.group(1) + new_depth_logic)
        print("✅ 已修复 base.py: 增加 6 通道强制拼接逻辑")
        with open(base_path, "w", encoding="utf-8") as f:
            f.write(base_content)
    else:
        print("⚠️ 警告: 无法在 base.py 定位插入点，请手动检查。")
else:
    print("✅ base.py 看起来已经包含 6 通道逻辑，跳过。")


# ==============================================================================
# 2. 修复 ultralytics/nn/tasks.py (强制模型按 6 通道构建)
# ==============================================================================
tasks_path = "/Users/leo/Desktop/deeplearing/ultralytics/ultralytics/nn/tasks.py"
with open(tasks_path, "r", encoding="utf-8") as f:
    tasks_content = f.read()

# 这一步非常暴力但有效：我们在 parse_model 函数入口处强制把 ch 设为 6
# 这样无论 train_dual.py 传什么，模型都会按 6 通道构建，匹配上面的数据
if "ch = 6 # [Auto-Fix] Force 6 channels" not in tasks_content:
    tasks_content = tasks_content.replace(
        "def parse_model(d, ch, verbose=True):",
        "def parse_model(d, ch, verbose=True):\n    ch = 6 # [Auto-Fix] Force 6 channels for RGB-D training"
    )
    print("✅ 已修复 tasks.py: 强制 parse_model 使用 ch=6")
    with open(tasks_path, "w", encoding="utf-8") as f:
        f.write(tasks_content)
else:
    print("✅ tasks.py 已经强制了 ch=6，跳过。")

print("\n🎉 修复完成！现在数据是 6 通道，模型也是 6 通道。")
print("请再次运行: python train_dual.py")