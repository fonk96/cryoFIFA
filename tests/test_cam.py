# 预处理
from cam_module import *
from torchvision import transforms
import torchvision.models as models

# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                          std=[0.229, 0.224, 0.225])
# ])

# 加载图像
model=models.resnet18()

# print(original_img.shape)

# input_tensor=torch.from_numpy(original_img).view(1,3,256,256)
input_tensor=torch.randn(1,3,256,256)
# print(input_tensor.size())
#input_tensor = preprocess(original_img).unsqueeze(0)

# 生成CAM
with torch.enable_grad():  # 确保梯度计算开启
    cam = generate_cam(model, input_tensor,1)

# 可视化
original_img = np.ones([256,256,3])
result = overlay_heatmap(original_img, cam)
cv2.imwrite("gradcam_result.jpg", result)