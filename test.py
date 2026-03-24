import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen3-VL-8B-Thinking"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
model.eval()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
)

trimmed = generated_ids[:, inputs["input_ids"].shape[-1]:]

raw = processor.batch_decode(
    trimmed,
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False,
)[0]
print("RAW:", repr(raw))

clean = processor.batch_decode(
    trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]
print("CLEAN:", clean)