# ライブラリーのインポート
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# アクセストークンの設定
access_tokens="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" # @param {type:"string"}
 
# モデルのインスタンス化
model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=access_tokens)
model.to("cuda")

prompt = "Tokyo Sky Tree by Marc Chagall" #@param {type:"string"}

# モデルにpromptを入力し画像生成
image = model(prompt,num_inference_steps=50).images[0]
# 保存
image.save("./sample.png")
 
# outputfile = f'{filename} _{i:02} .png'
# plt.imshow(plt.imread(f"outputfile/{outputfile}"))
# plt.axis('off')