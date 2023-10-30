#!/usr/bin/env python
# coding: utf-8

# # Stable Diffusion WebUI
# ## 阿里云DSW 一键脚本 By bilibili@十字鱼
# ### 参考感谢——秋葉aaaki、minicacas
# - 十字鱼 https://space.bilibili.com/893892
# - 修改git库地址
# - 修改插件

# ## 1.安装并运行Webui
# - 如果启动不成功，请停止后重复运行第1步。

# In[ ]:


import os
import time
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from huggingface_hub import HfApi, list_repo_files
#安装目录
path = "/mnt/workspace"
# path = "/mnt/data"
#安装网址
url = "https://gitcode.net/overbill1683/stable-diffusion-webui"
repo = url.split('/')[-1]
#是否重装
reinstall = False
#修改分支
branch = "master"
#SD模型（先换行，再加链接）
model_urls = """
https://civitai.com/api/download/models/94654?type=Model&format=SafeTensor&size=pruned&fp=fp16
https://civitai.com/api/download/models/119057?type=Model&format=SafeTensor&size=pruned&fp=fp16
https://civitai.com/api/download/models/89247?type=Model&format=SafeTensor&size=pruned&fp=fp16
https://civitai.com/api/download/models/30163?type=Model&format=SafeTensor&size=full&fp=fp16
https://civitai.com/api/download/models/103436?type=Model&format=SafeTensor&size=pruned&fp=fp16
https://civitai.com/api/download/models/89314
https://civitai.com/api/download/models/90879
https://civitai.com/api/download/models/125771
https://civitai.com/api/download/models/98960
https://civitai.com/api/download/models/139204
https://civitai.com/api/download/models/62084
"""
#VAE模型（先换行，再加链接）
VAE_urls = """
"""
#lora模型（先换行，再加链接）
lora_urls = """
https://civitai.com/api/download/models/62833
"""
#ControlNet模型（先换行，再加链接）
ControlNet_urls = """
https://huggingface.co/ioclab/ioc-controlnet/resolve/main/models/control_v1p_sd15_brightness.safetensors
https://huggingface.co/Gluttony10/1/resolve/main/control_v1p_sd15_brightness.yaml
https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/v2/control_v1p_sd15_qrcode_monster_v2.safetensors
https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/v2/control_v1p_sd15_qrcode_monster_v2.yaml
https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors
https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_shuffle_fp16.safetensors
https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1e_sd15_tile_fp16.safetensors
https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors
https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors
https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors
https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors
https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors
https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors
https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors
https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_scribble_fp16.safetensors
https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_seg_fp16.safetensors
https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors
https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors
https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/ip-adapter_sd15_plus.pth
"""
#扩展插件（先换行，再加链接,不要带.git）
extension_urls = """
https://github.com/continue-revolution/sd-webui-animatediff
https://github.com/sdbds/stable-diffusion-webui-wildcards
https://github.com/antfu/sd-webui-qrcode-toolkit
https://github.com/tzwm/sd-webui-model-downloader-cn
https://github.com/OedoSoldier/sd-webui-image-sequence-toolkit
https://gitcode.net/ranting8323/sd-webui-controlnet
https://gitcode.net/overbill1683/stable-diffusion-webui-localization-zh_Hans
https://gitcode.net/ranting8323/openpose-editor
https://gitcode.net/ranting8323/Stable-Diffusion-Webui-Civitai-Helper
https://gitcode.net/ranting8323/multidiffusion-upscaler-for-automatic1111
https://gitcode.net/ranting8323/stable-diffusion-webui-images-browser
https://gitcode.net/ranting8323/adetailer
https://gitcode.net/ranting8323/stable-diffusion-webui-wd14-tagger
https://gitcode.net/ranting8323/sd-webui-prompt-all-in-one
https://gitcode.net/ranting8323/sd-webui-inpaint-anything
https://gitcode.net/ranting8323/stable-diffusion-webui-two-shot
https://gitcode.net/ranting8323/a1111-sd-webui-tagcomplete
https://gitcode.net/ranting8323/ebsynth_utility
"""
#SD安装文件（不要修改）
repositorie_urls = """
https://gitcode.net/overbill1683/k-diffusion
https://gitcode.net/overbill1683/CodeFormer
https://gitcode.net/overbill1683/BLIP
https://gitcode.net/overbill1683/generative-models
"""
#embedding模型（只放带文件名的直链）（先换行，再加链接）
embedding_urls = """
https://github.com/gluttony-10/1/blob/main/badhandv4.pt
https://github.com/gluttony-10/1/blob/main/easynegative.safetensors
https://github.com/gluttony-10/1/blob/main/putnegative.safetensors
https://github.com/gluttony-10/1/blob/main/verybadimagenegative_v1.3.pt
"""
#adetailer模型（先换行，再加链接）
adetailer_urls = """
https://huggingface.co/Bingsu/adetailer/resolve/main/deepfashion2_yolov8s-seg.pt
https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt
https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n_v2.pt
https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt
https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8n-seg.pt
"""
#额外文件（先换行，再加链接）（下载到安装目录，方便剪切）
extra_urls = """
"""
#内存优化（不要修改）
def libtcmalloc():
    print('开始安装内存优化')
    get_ipython().system('git config --global http.postBuffer 2000000000')
    get_ipython().system('apt-get update -qq > /dev/null 2>&1')
    get_ipython().system('apt-get install -qq -y aria2 > /dev/null 2>&1')
    get_ipython().system('apt-get install -qq -y zip > /dev/null 2>&1')
    get_ipython().system('apt-get install -qq -y ffmpeg > /dev/null 2>&1')
    get_ipython().system('apt-get install -qq -y gifsicle > /dev/null 2>&1')
    get_ipython().system('apt-get install -qq -y libimage-exiftool-perl > /dev/null 2>&1')
    if os.path.exists(f'{path}/temp'):
        os.environ["LD_PRELOAD"] = "libtcmalloc.so"
        print('内存优化已安装')
    else:
        os.chdir(f'{path}')
        os.makedirs('temp', exist_ok=True)
        os.chdir('temp')
        os.system('wget -q http://launchpadlibrarian.net/367274644/libgoogle-perftools-dev_2.5-2.2ubuntu3_amd64.deb')
        os.system('wget -q https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/google-perftools_2.5-2.2ubuntu3_all.deb')
        os.system('wget -q https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libtcmalloc-minimal4_2.5-2.2ubuntu3_amd64.deb')
        os.system('wget -q https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libgoogle-perftools4_2.5-2.2ubuntu3_amd64.deb')
        os.system('apt-get install -qq libunwind8-dev -y > /dev/null 2>&1')
        get_ipython().system('dpkg -i *.deb > /dev/null 2>&1')
        os.environ["LD_PRELOAD"] = "libtcmalloc.so"
        get_ipython().system('rm *.deb > /dev/null 2>&1')
        print('内存优化安装完毕')
#安装主体（不要修改）
def install_main():
    print(f'开始安装{repo}')
    get_ipython().system('git -C {path} clone {url} > /dev/null 2>&1')
    get_ipython().system('git -C {path}/{repo} checkout {branch}')
    get_ipython().system('git -C {path}/{repo} pull')
    get_ipython().system('mkdir -p {path}/ebs/img2img_key > /dev/null 2>&1')
    get_ipython().system('mkdir -p {path}/ebs/img2img_upscale_key > /dev/null 2>&1')
    if os.path.exists(f'{path}/{repo}'):
        get_ipython().run_line_magic('cd', '{path}/{repo}')
        print('开始安装依赖，请耐心等待')
        get_ipython().system('pip install -U pip > /dev/null 2>&1')
        get_ipython().system('pip install -U -r requirements_versions.txt > /dev/null 2>&1')
        get_ipython().system('pip install -U torch==2.0.1 torchvision torchaudio > /dev/null 2>&1')
        get_ipython().system('pip install -U xformers==0.0.21 > /dev/null 2>&1')
        get_ipython().system('pip install -U ipywidgets > /dev/null 2>&1')
        get_ipython().system('pip install -U imageio > /dev/null 2>&1')
        get_ipython().system('pip install -U imageio-ffmpeg > /dev/null 2>&1')
        get_ipython().system('pip install -U setuptools > /dev/null 2>&1')
        get_ipython().system('pip install -U opencv-python > /dev/null 2>&1')
        get_ipython().system('mkdir {path}/{repo}/repositories > /dev/null 2>&1')
        if not os.path.exists(f'{path}/{repo}/config.json'):
            get_ipython().system('wget -q -O {path}/{repo}/config.json https://ghproxy.com/https://github.com/gluttony-10/1/blob/main/config.json')
        if not os.path.exists(f'{path}/{repo}/styles.csv'):
            get_ipython().system('wget -q -O {path}/{repo}/styles.csv https://ghproxy.com/https://github.com/gluttony-10/1/blob/main/styles.csv')
        get_ipython().run_line_magic('cd', '{path}')
        print(f'{repo}安装完毕')
#下载文件1（不要修改）
def download_1():
    tasks = []
    tasks.extend(download_git(extension_urls,f'{path}/{repo}/extensions'))
    tasks.append(f'git -C {path}/{repo}/repositories clone https://gitcode.net/overbill1683/stablediffusion stable-diffusion-stability-ai > /dev/null 2>&1')
    tasks.extend(download_git(repositorie_urls,f'{path}/{repo}/repositories'))
    tasks.append(f'git -C {path}/{repo}/repositories/stable-diffusion-stability-ai pull > /dev/null 2>&1')
    with ThreadPoolExecutor() as executor:
        for task in tasks:
            executor.submit(os.system, task)
#下载文件2（不要修改）
def download_2(times):
    tasks = []
    for i in range(times):
        tasks.extend(download_aria('https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/download/v1.0.0-pre/vaeapprox-sdxl.pt',f'{path}/{repo}/models/VAE-approx'))
        tasks.append(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -d {path}/{repo}/models/opencv -o face_detection_yunet.onnx https://ghproxy.com/https://github.com/opencv/opencv_zoo/blob/91fb0290f50896f38a0ab1e558b74b16bc009428/models/face_detection_yunet/face_detection_yunet_2022mar.onnx?raw=true > /dev/null 2>&1')
        tasks.extend(download_aria('https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt',f'{path}/{repo}/models/animatediff'))
        tasks.extend(download_aria(model_urls,f'{path}/{repo}/models/Stable-diffusion'))
        tasks.extend(download_aria(VAE_urls,f'{path}/{repo}/models/VAE'))
        tasks.extend(download_aria(lora_urls,f'{path}/{repo}/models/Lora'))
        tasks.extend(download_aria(ControlNet_urls,f'{path}/{repo}/models/ControlNet'))
        tasks.extend(download_aria(embedding_urls,f'{path}/{repo}/embeddings'))
        tasks.extend(download_aria(adetailer_urls,f'{path}/{repo}/models/adetailer'))
        tasks.extend(download_hf('Gluttony10/SD',f'{path}'))
        tasks.extend(download_aria('https://huggingface.co/Gluttony10/1/resolve/main/config.yaml','/root/.transparent-background'))
        tasks.extend(download_aria('https://huggingface.co/Gluttony10/1/resolve/main/ckpt_base.pth','/root/.transparent-background'))
        tasks.extend(download_aria('https://huggingface.co/Gluttony10/1/resolve/main/frpc_linux_amd64_v0.2','/usr/local/lib/python3.10/dist-packages/gradio'))
        tasks.extend(download_aria(extra_urls,f'{path}'))
    with ThreadPoolExecutor() as executor:
        for task in tasks:
            executor.submit(os.system, task)
#下载git库（不要修改）
def download_git(links,folder):
    tasks = []
    link = links.strip().split("\n")
    for li in link:
        if "https://ghproxy.com/https://github.com" in li:
            li=li.replace("https://ghproxy.com/https://github.com","https://ghproxy.com/https://github.com")
        elif "https://github.com" in li:
            li=li.replace("https://github.com","https://ghproxy.com/https://github.com")
        tasks.append(f'git -C {folder} clone -q {li} > /dev/null 2>&1')
    for li in link:
        name = li.split('/')[-1]
        tasks.append(f'git -C {folder}/{name} pull -q > /dev/null 2>&1')
    return tasks
#下载模块（不要修改）
def download_aria(links,folder):
    global check
    tasks = []
    link = links.strip().split("\n")
    for li in link:
        if "https://ghproxy.com/https://github.com" in li:
            li=li.replace("https://ghproxy.com/https://github.com","https://ghproxy.com/https://github.com")
        elif "https://github.com" in li:
            li=li.replace("https://github.com","https://ghproxy.com/https://github.com")
        elif "huggingface.co" in li:
            li=li.replace("huggingface.co","huggingface.sukaka.top")
        elif "civitai.com" in li:
            li=li.replace("civitai.com","civitai.sukaka.top")
        fi = li.split('/')[-1]
        if "." in fi:
            tasks.append(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -t 10 -d {folder} -o {fi} "{li}" > /dev/null 2>&1')
            check = check & os.path.exists(f'{folder}/{fi}')
        else:
            tasks.append(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -t 10 -d {folder} "{li}" > /dev/null 2>&1')
    return tasks
#下载抱脸库（不要修改）
def download_hf(repo,folder):
    global check
    hf_api = HfApi(endpoint="https://huggingface.sukaka.top",)
    tasks = []
    while True:
        try:
            files = hf_api.list_repo_files(f'{repo}')
            for fil in ['.gitattributes','README.md']:
                while fil in files:
                    files.remove(fil)
            break
        except:
            pass
    for fi in files:
        li = os.path.join("https://huggingface.sukaka.top", f'{repo}', "resolve/main/", f'{fi}')
        tasks.append(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -t 10 -d {folder} -o {fi} "{li}" > /dev/null 2>&1')
        check = check & os.path.exists(f'{folder}/{fi}')
    return tasks
#主进程（不要修改）
def main():
    time_start = time.time()
    global check
    print("运行开始")
    get_ipython().system('df -hl #查看磁盘')
    get_ipython().system('nvidia-smi #查看显卡')
    get_ipython().run_line_magic('cd', '{path}')
    if reinstall:
        print('旧文件删除中')
        get_ipython().system('rm -rf {path}/{repo}')
    with ProcessPoolExecutor() as executor:
        futures = []
        for func in [libtcmalloc,install_main]:
            futures.append(executor.submit(func))
            time.sleep(1)
        for future in futures:
            future.result()
    if os.path.exists(f'{path}/{repo}'):
        print("开始下载，请耐心等待")
        download_1()
        n = 1
        while not check:
            print("第",n,"次下载")
            n = n+1
            check = True
            download_2(1)
            if not check:
                download_2(1)
        time_end = time.time()
        print('\033[32m安装耗时:',int((time_end - time_start)/60),'min\033[0m')
        get_ipython().system('chmod 755 /usr/local/lib/python3.10/dist-packages/gradio/frpc_linux_amd64_v0.2')
        get_ipython().run_line_magic('cd', '{path}/{repo}')
        get_ipython().system('python launch.py --api --no-download-sd-model --opt-sdp-attention --share --listen --enable-insecure-extension-access')
    else:
        print('安装失败请重试')

check = False
if __name__ == '__main__':
    main()


# ## 2.快速启动

# In[ ]:


# get_ipython().run_line_magic('cd', '/mnt/workspace/stable-diffusion-webui')
# get_ipython().system('python launch.py --api --no-download-sd-model --opt-sdp-attention --share --listen --enable-insecure-extension-access')


# ## 3.压缩outputs并清空
# - 停止后再运行

# In[ ]:


# get_ipython().run_line_magic('cd', '/mnt/workspace/stable-diffusion-webui')
# get_ipython().system('zip -r outputs.zip outputs')
# get_ipython().system('mv outputs.zip ..')
# get_ipython().system('rm -rf outputs')


# ## 4.压缩工程目录并清空
# - 停止后再运行

# In[ ]:


# get_ipython().run_line_magic('cd', '/mnt/workspace')
# get_ipython().system('zip -r ebs.zip ebs')
# get_ipython().system('zip -s 500m ebs.zip --out eb.zip')
#!rm -rf ebs
#!mkdir -p ebs/img2img_key
#!mkdir -p ebs/img2img_upscale_key

