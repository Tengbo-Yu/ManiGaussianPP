import cv2
import numpy as np

# 读取两张图片
image1 = cv2.imread('1.png')  # 假设图片1路径
image2 = cv2.imread('2.png')  # 假设图片2路径

# 确保两张图片的尺寸相同
image1_resized = cv2.resize(image1, (image2.shape[1], image2.shape[0]))


# 对应像素相乘
multiplied_image = cv2.multiply(image1_resized, image2)

# 限制像素值在 [0, 255] 范围内
multiplied_image = np.clip(multiplied_image, 0, 255).astype(np.uint8)

# 显示相乘后的图像
# cv2.imshow('Multiplied Image', multiplied_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 保存结果
cv2.imwrite('multiplied_image.jpg', multiplied_image)