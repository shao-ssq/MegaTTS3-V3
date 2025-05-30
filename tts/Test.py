import multiprocessing as mp
import torch
import os
import sys
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from tts.infer_cli import MegaTTS3DiTInfer, convert_to_wav, cut_wav


def model_worker(input_queue, output_queue, device_id):
    device = None
    if device_id is not None:
        device = torch.device(f'cuda:{device_id}')
    infer_pipe = MegaTTS3DiTInfer(device=device)

    while True:
        task = input_queue.get()
        if task is None:
            break  # 用于优雅终止进程
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


def synthesize(inp_audio, inp_npy, inp_text, infer_timestep, p_w, t_w, input_queue, output_queue):
    print("Pushing task:", inp_audio, inp_npy, inp_text, infer_timestep, p_w, t_w)
    input_queue.put((inp_audio, inp_npy, inp_text, infer_timestep, p_w, t_w))
    res = output_queue.get()
    if res is not None:
        return res
    else:
        print("Synthesis failed.")
        return None


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    mp_manager = mp.Manager()

    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if devices != '':
        devices = devices.split(",")
    else:
        devices = None

    num_workers = 1
    input_queue = mp_manager.Queue()
    output_queue = mp_manager.Queue()
    processes = []

    print("Starting worker processes...")
    for i in range(num_workers):
        device_id = i % len(devices) if devices else None
        p = mp.Process(target=model_worker, args=(input_queue, output_queue, device_id))
        p.start()
        processes.append(p)

    example_result = synthesize(
        inp_audio="D:\\GitHub\\webank\\2025上半年\\开源项目探索\\TTS\\AOB\\林思朗.wav",
        inp_npy="example_embedding.npy",
        inp_text="你微粒贷已经逾期，经多次催缴仍未还清，你的信用记录将受到严重影响。信用风险已经被提高，请高度重视以免造成更大的负面影响及承担法律责任。你这边先想办法周转一下，赶在今天之前还进来还好吧。因为一直逾期的话是会影响征信的。",
        infer_timestep=32,
        p_w=1.4,
        t_w=3.0,
        input_queue=input_queue,
        output_queue=output_queue
    )

    if example_result:
        with open("output.wav", "wb") as f:
            f.write(example_result)
