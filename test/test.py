from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

model_name = "Qwen/Qwen2-VL-2B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    trust_remote_code=True,  # Add this
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

def load_and_resize(image_path, max_size=512):
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_size, max_size))
    return img
5

def describe_image(image_path):
    image = load_and_resize(image_path, max_size=384)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": """Describe this technical diagram/figure in detail:
- Type of visualization (diagram, chart, graph, table, etc.)
- Key components and their relationships in short.
Be specific and technical."""}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    input_len = inputs['input_ids'].shape[1]
    
    outputs = model.generate(**inputs, max_new_tokens=300)
    return processor.decode(outputs[0][input_len:], skip_special_tokens=True)



print(describe_image("/Users/tusharc/Code/Python/RAG/RAG-Model/extracted_images/attention_p3_img_1.png"))
