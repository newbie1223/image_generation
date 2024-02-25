# ライブラリーのインポート
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# アクセストークンの設定
access_tokens="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" # @param {type:"string"}
 
# モデルのインスタンス化
model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=access_tokens)
model.to("cuda")

prompt = "Tokyo Sky Tree by Marc Chagall" #@param {type:"string"}

#　画像出力のディレクトリ
import os
outputfile = './outputfile'
os.mkdir(outputfile)

# 画像のファイル名
import re
filename = re.sub(r'[\\/:*?"<>|,]+', '', prompt).replace(' ','_')

# 画像数
num = 1
 
for i in range(num):
  # モデルにpromptを入力し画像生成
  image = model(prompt,num_inference_steps=100)["sample"][0]
  # 保存
  outputfile = f'{filename} _{i:02} .png'
  image.save(f"outputfile/{outputfile}")
 
for i in range(num):
  outputfile = f'{filename} _{i:02} .png'
  plt.imshow(plt.imread(f"outputfile/{outputfile}"))
  plt.axis('off')