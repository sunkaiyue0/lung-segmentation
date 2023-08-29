import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

from src.data import LungDataset, blend, Pad, Crop, Resize
from src.models import UNet, PretrainedUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_folder = Path("input", "dataset")
models_folder = Path("models")

# origin_filename = "input/dataset/images/CHNCXR_0042_0.png"
# origin_filename = r"C:\Users\lungpoint\Desktop\DRR\44_005_AH (2).png"
origin_filename = r"D:/dev/code/PatientRegistration/datasets/covid-19-chest-x-ray-dataset/images/960053.png"
reverse_gray = False

unet = PretrainedUNet(
    in_channels=1,
    out_channels=2, 
    batch_norm=True, 
    upscale_mode="bilinear"
)

model_name = r"D:\dev\code\PatientRegistration\segmentation\lung_region\weights\unet-7v.pt"
unet.load_state_dict(torch.load(models_folder / model_name, map_location=torch.device("cpu")))
unet.to(device)
unet.eval()

origin = Image.open(origin_filename).convert("L")


origin = torchvision.transforms.functional.resize(origin, (512, 512))
origin = torchvision.transforms.functional.to_tensor(origin) - 0.5
if reverse_gray:
    origin = -origin


with torch.no_grad():
    origin = torch.stack([origin])
    origin = origin.to(device)
    out = unet(origin)
    softmax = torch.nn.functional.log_softmax(out, dim=1)
    out = torch.argmax(softmax, dim=1)
    
    origin = origin[0].to("cpu")
    out = out[0].to("cpu")
    

plt.figure(figsize=(16,8))

pil_origin = torchvision.transforms.functional.to_pil_image(origin + 0.5).convert("RGB")

plt.subplot(1, 2, 1)
plt.title("origin image")
plt.imshow(np.array(pil_origin))

plt.subplot(1, 2, 2)
plt.title("blended origin + predict")
plt.imshow(np.array(blend(origin, out)))

plt.show()