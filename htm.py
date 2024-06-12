import tensorflow as tf
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import time
import os
import torch
import clip
from PIL import Image

# 定义图片加载和保存函数
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    shape = tf.shape(img)[:-1]
    shape = tf.cast(shape, tf.float32)
    long_dim = tf.reduce_max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# 创建 output 文件夹
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# 加载VGG19模型
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# 定义内容和风格层
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# 定义VGG模型
def vgg_layers(layer_names):
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# 提取风格和内容特征
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

# 定义损失函数
def style_content_loss(outputs, style_targets, content_targets):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

# 裁剪图像
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# 定义风格和内容模型
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                          outputs[self.num_style_layers:])
        
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value 
                      for style_name, value 
                      in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

# 加载内容图片
content_image_path = '/Users/a1234/Desktop/final/content_image.jpg'
content_image = load_img(content_image_path)

# 使用 CLIP 模型从风格库中选择最合适的风格图片
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 定义从风格库中选择最合适风格图片的函数
def select_best_style_image(style_images_dir, style_description):
    style_image_paths = [os.path.join(style_images_dir, fname) for fname in os.listdir(style_images_dir) if fname.endswith(('.jpg', '.png'))]
    print(f"Found {len(style_image_paths)} style images.")

    # 计算文本描述的特征
    text_inputs = torch.cat([clip.tokenize(style_description)]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # 计算每张风格图像的特征
    image_features_list = []
    image_list = []
    for image_path in style_image_paths:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        image_features_list.append(image_features)
        image_list.append(image_path)

    # 计算相似度并选择最相似的图像
    similarities = []
    for image_features in image_features_list:
        similarity = torch.cosine_similarity(text_features, image_features)
        similarities.append(similarity.item())

    # 找到最相似的图像
    most_similar_index = np.argmax(similarities)
    most_similar_image_path = image_list[most_similar_index]

    print(f"Most similar image: {most_similar_image_path}")
    return most_similar_image_path

# 选择最合适的风格图片
style_images_dir = '/Users/a1234/Desktop/final/styles'
style_description = "A painting in the style of melting effect"
most_similar_image_path = select_best_style_image(style_images_dir, style_description)

# 使用找到的最合适风格图像进行风格迁移
style_image = load_img(most_similar_image_path)
extractor = StyleContentModel(style_layers, content_layers)
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# 初始化目标图像
image = tf.Variable(content_image)

# 定义优化器
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

# 训练模型
style_weight = 1e-2
content_weight = 1e4

start = time.time()

# 设置较多的epoch和步数以进行充分训练
epochs = 20
steps_per_epoch = 100

step = 0
print("Starting training...")
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        current_loss = style_content_loss(extractor(image), style_targets, content_targets)
        print(f"Train step: {step}, loss: {current_loss.numpy()}")
    print(f"Epoch {n+1} completed")
    # 保存每个epoch的中间结果
    intermediate_img = tensor_to_image(image)
    intermediate_img.save(os.path.join(output_dir, f'output_image_epoch_{n+1}.jpg'))
    print(f"Intermediate image saved for epoch {n+1}")

end = time.time()
print("Total time: {:.1f} seconds".format(end-start))

# 显示并保存最终结果
final_img = tensor_to_image(image)
plt.imshow(final_img)
plt.show()

output_image_path = os.path.join(output_dir, 'output_image.jpg')
final_img.save(output_image_path)
print(f"Output image saved to {output_image_path}")
