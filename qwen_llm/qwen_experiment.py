import transformers
import subprocess

print("transformers version", transformers.__version__)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os

os.chdir("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/qwen_llm")
from qwen25_vl.qwen_vl_utils.src.qwen_vl_utils.vision_process import process_vision_info


def frames_to_video(frame_dir, output_path, framerate=1):
    # Pattern: assumes frames like frame_000001.jpg
    input_pattern = os.path.join(frame_dir, '%06d.jpg')

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

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]


video_dir = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/original_action_swap/sample_000000_deadlifting_to_juggling soccer ball_f8dp7wR4GWg_000003_000013_bg_MCV13GHst20_000002_000012"

max_frames = 16
all_frames = sorted([
    os.path.join(video_dir, fname)
    for fname in os.listdir(video_dir)
    if fname.endswith((".jpg", ".jpeg", ".png"))
])
step = max(1, len(all_frames) // max_frames)
frame_paths = all_frames[::step][:max_frames]

# question = f"What is the action being performed? A) {choices[0]} B) {choices[1]} C) {choices[2]} D) {choices[3]} E) {choices[4]}"
question = f"What is the action being performed? Please answer with only the letter. A) deadlifting B) playing violin C) juggling soccer ball D) eating food E) skiing"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": frame_paths},
            {"type": "text", "text": question},
        ]
    },
]

print("Preparing text")
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print("Finished preparing text")

print("Preparing images")
image_inputs, video_inputs = process_vision_info(messages)
print("Finished preparing images")

print("Preparing inputs")
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
print("Finished preparing inputs")
inputs = inputs.to(model.device)

print("Preparing generated_ids")
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
print("Finished preparing generated_ids")
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
print(output_text[0])