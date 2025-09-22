from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from convert_mp42wav import *
from pydub import AudioSegment

from moviepy import *

def extract_audio_from_mp4(mp4_file, wav_file):
    # 打开MP4文件
    video = VideoFileClip(mp4_file)
    
    try:
        # 读取音频数据
        audio = video.audio
        
        # 将音频数据保存为WAV文件
        audio.write_audiofile(wav_file)
    finally:
        # 关闭MP4文件
        video.close()


inference_pipeline = pipeline(task=Tasks.auto_speech_recognition,model='Whisper-large-v3')

def split_wav(input_file: str,  segment_length: int = 15, overlap: int = 0):
 
    audio = AudioSegment.from_wav(input_file)
    
    # 获取基本文件名（不带扩展名）
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # 计算每个片段的毫秒数
    segment_length_ms = segment_length * 1000
    overlap_ms = overlap * 1000
    step_ms = segment_length_ms - overlap_ms
    
    # 获取音频总时长
    duration_ms = len(audio)
    result=""
    # 切割音频
    segment_count = 0
    for start_ms in range(0, duration_ms, step_ms):
        end_ms = start_ms + segment_length_ms   
        # 确保不超出音频边界
        if end_ms > duration_ms:
            end_ms = duration_ms  
        # 提取片段
        segment = audio[start_ms:end_ms]
        # 保存片段
        segment.export("tmp2.wav", format="wav")
        #提取语音转成的文字
        rec_result = inference_pipeline(input="tmp2.wav", language=None )
        result+=rec_result[0]["text"]
        segment_count += 1
        
    return result
    
def extract_audio_text(mp4_path):
    extract_audio_from_mp4(mp4_path, wav_path="tmp.wav")
    result=split_wav(wav_path,  segment_length= 15, overlap = 0)
    return result

if __name__ == "__main__":
    video_dir = "/root/autodl-tmp/video"
    output_file = "video_audio_text.jsonl"
    with open(output_file, "w") as outf:
        pass
        
    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        try:
            video_audio_text = extract_audio_text(video_path)
            output_string = json.dumps({video_file:video_audio_text} , ensure_ascii = False)
        except Exception as e:
            print(video_path)
            print(e)
        

        with open(output_file, "a") as outf:
            outf.write(output_string + "\n")
