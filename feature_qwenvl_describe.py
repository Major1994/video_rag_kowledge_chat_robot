from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import json
import os

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen2.5-VL-7B-Instruct")

def extract_video_description(path,query=None):
    type="video" if path.endswith("mp4") else "image"
    if query==None:
        describe="描述这个"+"视频" if path.endswith("mp4") else "图片"
    else:
        describe=query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": type,
                    type: path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                    #"nframes":10
                },
                {"type": "text", "text": describe},
                
            ],
        }
        
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

if __name__ == "__main__":
    video_dir = "/root/autodl-tmp/video"
    output_file = "video_description.jsonl"
    with open(output_file, "w") as outf:
        pass
        
    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        try:
            video_description = extract_video_description(video_path)
            output_string = json.dumps({video_file:video_description} , ensure_ascii = False)
        except Exception as e:
            print(video_path)
            print(e)
        

        with open(output_file, "a") as outf:
            outf.write(output_string + "\n")
