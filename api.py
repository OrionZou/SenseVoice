# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from typing_extensions import Annotated
from typing import List
from enum import Enum
import torchaudio
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
from fastapi.responses import JSONResponse
import torchaudio

class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"

model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=os.getenv("SENSEVOICE_DEVICE", "cuda:0"))
m.eval()

regex = r"<\|.*\|>"

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """

@app.post("/api/v1/asr")
async def turn_audio_to_text(files: Annotated[List[bytes], File(description="wav or mp3 audios in 16KHz")], keys: Annotated[str, Form(description="name of each audio joined with comma")], lang: Annotated[Language, Form(description="language of audio content")] = "auto"):
    audios = []
    audio_fs = 0
    for file in files:
        file_io = BytesIO(file)
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)
        data_or_path_or_list = data_or_path_or_list.mean(0)
        audios.append(data_or_path_or_list)
        file_io.close()
    if lang == "":
        lang = "auto"
    if keys == "":
        key = ["wav_file_tmp_name"]
    else:
        key = keys.split(",")
    res = m.inference(
        data_in=audios,
        language=lang, # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        ban_emo_unk=False,
        key=key,
        fs=audio_fs,
        **kwargs,
    )
    if len(res) == 0:
        return {"result": []}
    for it in res[0]:
        it["raw_text"] = it["text"]
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
        it["text"] = rich_transcription_postprocess(it["text"])
    return {"result": res[0]}

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str = Form("auto"),
    prompt: str = Form(None),
    temperature: float = Form(0.0),
    response_format: str = Form("json")
):
    # 读取上传音频文件
    file_bytes = await file.read()
    file_io = BytesIO(file_bytes)

    # 加载音频，转 mono
    audios = []
    audio_fs = 0
    waveform, audio_fs = torchaudio.load(file_io)
    waveform = waveform.mean(0)  # 转 mono
    audios.append(waveform)
    file_io.close()

    # 推理（使用你原来的 m.inference）
    # res = m.inference(
    #     data_in=[waveform],
    #     language=language,
    #     use_itn=False,
    #     ban_emo_unk=False,
    #     key=["input_audio"],
    #     fs=sample_rate
    # )
    res = m.inference(
        data_in=audios,
        language=language, # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        ban_emo_unk=False,
        key=["input_audio"],
        fs=audio_fs,
        **kwargs,
    )

    if not res or not res[0]:
        return {"text": ""}

    # 后处理（你已有的逻辑）
    result = res[0]
    for it in result:
        it["raw_text"] = it["text"]
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
        it["text"] = rich_transcription_postprocess(it["text"])

    # OpenAI 格式返回（只返回 text）
    return JSONResponse(content={"text": result[0]["text"]})