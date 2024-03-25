# coding:utf-8

from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import time

###################
# 模型所需库包
import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

# os.chdir("D:/花卉识别系统beta")  # 将当前工作目录更改为包含文件的目录
# 获取当前文件所在目录
current_directory = os.path.dirname(__file__)
os.chdir(current_directory)  # 将当前工作目录更改为包含文件的目录
print("当前工作目录:", os.getcwd())

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = AlexNet(num_classes=5)
# load model weights
model_weight_path = "./AlexNet.pth"
# , map_location='cpu'
model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

# 关闭 Dropout
model.eval()

###################
from datetime import timedelta

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


# 图片装换操作
def tran(img_path):
    print("Image Path:", img_path)  # 调试打印图片路径
    # 预处理
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img = Image.open(img_path)
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    return img


@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    path = ""
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        filename = secure_filename(f.filename)
        path = secure_filename(f.filename)
        upload_path = os.path.join(basepath, 'static/images', filename)  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        print(path)
        # 将上传的文件保存到指定的路径
        f.save(upload_path)
        # img = tran('static/images/' + path)
        img = tran(upload_path)
        # img = tran(os.path.join('static', 'images', path))
        
        ##########################
        # 预测图片
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img))  # 将输出压缩，即压缩掉 batch 这个维度
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            res = class_indict[str(predict_cla)]
            pred = predict[predict_cla].item()
            print(class_indict[str(predict_cla)], predict[predict_cla].item())
        # res_chinese = ""  # 添加这一行，初始化 res_chinese 变量
        # res_english = ""  # 添加这一行，初始化 res_english 变量
        # if res == "daisy":
        #     res_chinese = "雏菊"
        # if res == "dandelion":
        #     res_chinese = "蒲公英"
        #     if res == "roses":
        #         res_chinese = "玫瑰"
        # if res == "sunflowers":
        #     res_chinese = "向日葵"
        # if res == "tulips":
        #     res_chinese = "郁金香"
        # else:
            # 如果没有匹配到特定花卉，使用class_indices.json中的原名
            res_english = class_indict[str(predict_cla)]
        # 如果准确率低于阈值，将结果设置为“未知花卉或不是花卉”
        if pred < 0.85:
            res_english = "识别失败,未知花卉或不是花卉"

        # print('result:', class_indict[str(predict_class)], 'accuracy:', prediction[predict_class])
        ##########################
        f.save(upload_path)
        image_url = '/static/images/' + filename  # 生成图片的URL
        print("Image URL:", image_url)  # 打印出图片URL，确认其正确性
        pred = pred * 100
        return render_template('upload_ok.html', path=path, res_english=res_english, pred=pred, val1=time.time())
        # 使用图片URL、识别结果和准确率作为参数调用模板
        return render_template('upload_ok.html', image_url=image_url, res_english=res_english, pred=pred_percentage)

    return render_template('upload.html')


if __name__ == '__main__':
    # app.debug = True
    app.run(host='127.0.0.1', port=80, debug=True)
