from FastSAM.fastsam import FastSAM, FastSAMPrompt
import torch

# Define an inference SOURCE
SOURCE = "FastSAM/examples/dogs.jpg"

# Create a FastSAM model
model = FastSAM('FastSAM-x.pt')  # or FastSAM-x.pt

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Run inference on an image
everything_results = model(SOURCE, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

# Prepare a Prompt Process object
prompt_process = FastSAMPrompt(SOURCE, everything_results, device='cpu')

# Everything prompt
ann = prompt_process.everything_prompt()

prompt_process.plot(annotations=ann, output_path='./output/')