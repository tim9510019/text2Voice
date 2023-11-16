import os, stat
import subprocess
from zipfile import ZipFile
import uuid
import torch
import torchaudio
import re
import gradio as gr
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir

# By using XTTS you agree to CPML license https://coqui.ai/cpml
os.environ["COQUI_TOS_AGREED"] = "1"

repo_id = "coqui/xtts"
assetPath = "./assets"

# Use never ffmpeg binary for Ubuntu20 to use denoising for microphone input
print("Export newer ffmpeg binary for denoise filter")
os.chdir(assetPath)
ZipFile("ffmpeg.zip").extractall()
os.chdir("..")
print("Make ffmpeg binary executable")
st = os.stat(os.path.join(assetPath, "ffmpeg"))
os.chmod(os.path.join(assetPath, "ffmpeg"), st.st_mode | stat.S_IEXEC)

# This will trigger downloading model
print("Downloading if not downloaded Coqui XTTS V2")
from TTS.utils.manage import ModelManager

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
ModelManager().download_model(model_name)
model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
print("XTTS downloaded")

config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=os.path.join(model_path, "model.pth"),
    vocab_path=os.path.join(model_path, "vocab.json"),
    eval=True,
    use_deepspeed=True,
)
model.cuda()

# supported_languages = config.languages


def predict(
    prompt,
    language,
    audio_file_pth,
):
    speaker_wav = audio_file_pth

    # Clean-up reference voice - Apply all on demand
    lowpassfilter = trim = True

    if lowpassfilter:
        lowpass_highpass = "lowpass=8000,highpass=75,"
    else:
        lowpass_highpass = ""

    if trim:
        # better to remove silence in beginning and end for microphone
        trim_silence = "areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,"
    else:
        trim_silence = ""

    try:
        out_filename = (
            speaker_wav + str(uuid.uuid4()) + ".wav"
        )  # ffmpeg to know output format

        # we will use newer ffmpeg as that has afftn denoise filter
        shell_command = f"./assets/ffmpeg -y -i {speaker_wav} -af {lowpass_highpass}{trim_silence} {out_filename}".split(
            " "
        )

        subprocess.run(
            [item for item in shell_command],
            capture_output=False,
            text=True,
            check=True,
        )
        speaker_wav = out_filename
        print("Filtered microphone input")
    except subprocess.CalledProcessError:
        # There was an error - command exited with non-zero code
        print("Error: failed filtering, use original microphone input")

    try:
        try:
            (
                gpt_cond_latent,
                speaker_embedding,
            ) = model.get_conditioning_latents(
                audio_path=speaker_wav, gpt_cond_len=30, max_ref_length=60
            )
        except Exception as e:
            print("Speaker encoding error", str(e))
            gr.Warning(
                "It appears something wrong with reference, did you unmute your microphone?"
            )
            return (
                None,
                None,
                None,
            )

        # temporary comma fix
        prompt = re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)", r"\1 \2\2", prompt)

        ## Direct mode

        print("I: Generating new audio...")

        out = model.inference(prompt, language, gpt_cond_latent, speaker_embedding)

        torchaudio.save("output.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)

        # print("I: Generating new audio in streaming mode...")
        # wav_chunks = []
        # chunks = model.inference_stream(
        #     prompt,
        #     language,
        #     gpt_cond_latent,
        #     speaker_embedding,
        #     repetition_penalty=7.0,
        #     temperature=0.85,
        # )

        # first_chunk = True
        # for i, chunk in enumerate(chunks):
        #     if first_chunk:
        #         first_chunk_time = time.time() - t0
        #         metrics_text += f"Latency to first audio chunk: {round(first_chunk_time*1000)} milliseconds\n"
        #         first_chunk = False
        #     wav_chunks.append(chunk)
        #     print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
        # torchaudio.save("output.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)

    except RuntimeError as e:
        if "device-side assert" in str(e):
            # cannot do anything on cuda device side error, need tor estart
            print(
                f"Exit due to: Unrecoverable exception caused by language:{language} prompt:{prompt}",
                flush=True,
            )
            gr.Warning("Unhandled Exception encounter, please retry in a minute")
            print("Cuda device-assert Runtime encountered need restart")

        else:
            if "Failed to decode" in str(e):
                print("Speaker encoding error", str(e))
                gr.Warning(
                    "It appears something wrong with reference, did you unmute your microphone?"
                )
            else:
                print("RuntimeError: non device-side assert error:", str(e))
                gr.Warning("Something unexpected happened please retry again.")
            return (
                None,
                None,
                None,
            )

    return (
        gr.make_waveform(
            audio="output.wav",
        ),
        "output.wav",
        speaker_wav,
    )


examples = [
    [
        "Once when I was six years old I saw a magnificent picture",
        "en",
        "./assets/examples/female.wav",
    ],
    [
        "当我还只有六岁的时候， 看到了一副精彩的插画",
        "zh-cn",
        "./assets/examples/male.wav",
    ],
    [
        "한번은 내가 여섯 살이었을 때 멋진 그림을 보았습니다.",
        "ko",
        "./assets/examples/female.wav",
    ],
    [
        "Lorsque j'avais six ans j'ai vu, une fois, une magnifique image",
        "fr",
        "./assets/examples/male.wav",
    ],
    [
        "Un tempo lontano, quando avevo sei anni, vidi un magnifico disegno",
        "it",
        "./assets/examples/female.wav",
    ],
]


with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("CoquiTTS Text to Voice")
            gr.Markdown(
                """Leave a star 🌟 on the Github <a href="https://github.com/coqui-ai/TTS">🐸TTS</a>, where our open-source inference and training code lives.)"""
            )
            gr.Markdown(
                "I agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml"
            )

    with gr.Row():
        with gr.Column():
            input_text_gr = gr.Textbox(
                label="Text Prompt",
                info="One or two sentences at a time is better. Up to 200 text characters.",
                value="Hi there, I'm your new voice clone. Try your best to upload quality audio",
            )
            language_gr = gr.Dropdown(
                label="Language",
                info="Select an output language for the synthesised speech",
                choices=["en", "zh-cn", "ko", "fr", "it"],
                max_choices=1,
                value="en",
            )
            ref_gr = gr.Audio(
                label="Reference Audio",
                type="filepath",
                value=os.path.join(assetPath, "examples/female.wav"),
            )

            tts_button = gr.Button("Send", elem_id="send-btn", visible=True)

        with gr.Column():
            video_gr = gr.Video(label="Waveform Visual")
            audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
            ref_audio_gr = gr.Audio(label="Reference Audio Used")

    with gr.Row():
        gr.Examples(
            examples,
            label="Examples",
            inputs=[
                input_text_gr,
                language_gr,
                ref_gr,
            ],
            outputs=[video_gr, audio_gr, ref_audio_gr],
            fn=predict,
            cache_examples=False,
        )

    tts_button.click(
        predict,
        [
            input_text_gr,
            language_gr,
            ref_gr,
        ],
        outputs=[video_gr, audio_gr, ref_audio_gr],
    )

demo.queue()
demo.launch(share=True)
