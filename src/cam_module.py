import torch
import numpy as np
import cv2


class CAM_Extractor:
    def __init__(self, model,target_layer,input_mask=None):
        self.model = model
        self.activations = []
        #target_layer=self.model.module.feature_extractor.layers.layer1[-1].conv2
        
        # 注册前向钩子
        self.hooks=target_layer.register_forward_hook(self.save_activation)
        
    def __del__(self):
        self.hooks.remove()
        #torch.cuda.empty_cache()

    def save_activation(self, module, input, output):
          self.activations.append(output.detach().clone().cpu())
    def get_feature_map(self,input_tensor):
        with torch.no_grad():
            _=self.model(input_tensor)
        activations=self.activations[0][0].data.numpy()
        cam=np.sum(activations,axis=0)
        del activations
        activations=None
        return cam

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = []
        self.gradients = []
        
        # 注册前向钩子
        target_layer.register_forward_hook(self.save_activation)
        # 注册反向钩子（注意PyTorch>=1.8的语法）
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output.detach())

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())

def generate_cam(model, input_tensor, target_class=None):
    target_layer=model.layer4[-1].conv2
    grad_cam = GradCAM(model, target_layer)
    
    # 前向传播
    output = model(input_tensor)
    
    # 自动获取目标类别（如果未指定）
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # 反向传播梯度计算
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0][target_class] = 1.0
    output.backward(gradient=one_hot)
    
    # 获取激活和梯度
    activations = grad_cam.activations[0].cpu().data.numpy()[0]
    gradients = grad_cam.gradients[0].cpu().data.numpy()[0]
    
    # 计算权重
    weights = np.mean(gradients, axis=(1,2))
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)
    cam = np.maximum(cam, 0)  # ReLU
    
    return cam


def overlay_heatmap(original_img, cam):
    # 归一化处理
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    # 上采样到原图尺寸
    cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    
    # 生成热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + original_img * 0.6
    
    return superimposed_img


def color_heatmap(cam):
    # 归一化处理
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    

    # 生成热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    
    return heatmap