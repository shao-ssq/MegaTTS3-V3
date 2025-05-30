import multiprocessing as mp
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from functools import partial
import gradio as gr
import traceback
from tts.infer_cli import MegaTTS3DiTInfer, convert_to_wav, cut_wav


def model_worker(input_queue, output_queue, device_id):
    device = None
    if device_id is not None:
        device = torch.device(f'cuda:{device_id}')
    infer_pipe = MegaTTS3DiTInfer(device=device)

    while True:
        task = input_queue.get()
        inp_audio_path, inp_npy_path, inp_text, infer_timestep, p_w, t_w = task
        try:
            convert_to_wav(inp_audio_path)
            wav_path = os.path.splitext(inp_audio_path)[0] + '.wav'
            cut_wav(wav_path, max_len=28)
            with open(wav_path, 'rb') as file:
                file_content = file.read()
            resource_context = infer_pipe.preprocess(file_content, latent_file=inp_npy_path)
            wav_bytes = infer_pipe.forward(resource_context, inp_text, time_step=infer_timestep, p_w=p_w, t_w=t_w)
            output_queue.put(wav_bytes)
        except Exception as e:
            traceback.print_exc()
            print(task, str(e))
            output_queue.put(None)


def main(inp_audio, inp_npy, inp_text, infer_timestep, p_w, t_w, processes, input_queue, output_queue):
    print("Push task to the inp queue |", inp_audio, inp_npy, inp_text, infer_timestep, p_w, t_w)
    input_queue.put((inp_audio, inp_npy, inp_text, infer_timestep, p_w, t_w))
    res = output_queue.get()
    if res is not None:
        return res
    else:
        print("")
        return None


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    mp_manager = mp.Manager()

    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if devices != '':
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(",")
    else:
        devices = None
    
    num_workers = 1
    input_queue = mp_manager.Queue()
    output_queue = mp_manager.Queue()
    processes = []

    print("Start open workers")
    for i in range(num_workers):
        p = mp.Process(target=model_worker, args=(input_queue, output_queue, i % len(devices) if devices is not None else None))
        p.start()
        processes.append(p)

    api_interface = gr.Interface(fn=
                                partial(main, processes=processes, input_queue=input_queue, 
                                        output_queue=output_queue), 
                                inputs=[gr.Audio(type="filepath", label="上传参考音频"), gr.File(type="filepath", label="Upload .npy", visible=False), gr.Textbox(label="输入合成文本", lines=1, value="轻轻的我走了，正如我轻轻的来。我挥一挥衣袖，不带走一片云彩。"), 
                                        gr.Number(label="推理时间步长", value=32),
                                        gr.Number(label="Intelligibility 权重", value=1.4),
                                        gr.Number(label="相似度权重", value=3.0)], outputs=[gr.Audio(label="生成的语音")],
                                title="MegaTTS3 - 高保真语音克隆、自然真实情感的文本转语音系统",  
                                description="更多前沿的AI技术和应用，访问:https://deepfaces.cc", concurrency_limit=1,
                                submit_btn="生成",
                                clear_btn="清空"
                                )
    api_interface.launch(server_name='127.0.0.1', server_port=7929, debug=True, inbrowser=True)
    for p in processes:
        p.join()
