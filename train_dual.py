from ultralytics import YOLO

# 1. 加载你的双流模型配置文件
model = YOLO("yolo11-fusion.yaml") 

# 2. 启动训练
# 注意：这里我们不能直接传 ch=6，因为 YOLO 的 train() 方法也会检查参数。
# 我们需要用一个小技巧：在加载模型时并不需要指定 ch，
# 而是通过修改底层代码的默认值来确立 6 通道（详见下文“配套修改”）。
model.train(
    data="MUObj.yaml",
    epochs=100,
    imgsz=640,
    batch=16, # 根据显存调整
    hsv_h=0,  # 必须关闭 HSV
    hsv_s=0,
    hsv_v=0,
    pretrained="yolo11n.pt", # 加载预训练权重
    project="MUObj_Dual",
    name="exp1"
)