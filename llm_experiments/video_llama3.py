import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
import subprocess
import os
import accelerate

def frames_to_video(frame_dir, output_path, framerate=1):
    # Pattern: assumes frames like frame_000001.jpg
    input_pattern = os.path.join(frame_dir, 'frame_%06d.jpg')

    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-framerate', str(framerate),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    # Run the command
    subprocess.run(cmd, check=True)

model_name = "DAMO-NLP-SG/VideoLLaMA3-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
video_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/original_action_swap/sample_000000_deadlifting_to_juggling soccer ball_f8dp7wR4GWg_000003_000013_bg_MCV13GHst20_000002_000012"
question = "Describe this video in detail."

# Video conversation
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 128}},
            {"type": "text", "text": question},
        ]
    },
]

inputs = processor(conversation=conversation, return_tensors="pt")
inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
output_ids = model.generate(**inputs, max_new_tokens=128)
response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(response)