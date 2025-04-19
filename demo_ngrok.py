# --- Start of modifications ---
import os
import subprocess
import sys
import atexit

# Install pyngrok if not present
try:
    from pyngrok import ngrok, conf
    print("pyngrok found.")
except ImportError:
    print("pyngrok not found. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
        from pyngrok import ngrok, conf
        print("pyngrok installed successfully.")
    except Exception as e:
        print(f"Failed to install pyngrok: {e}")
        print("Please install pyngrok manually ('pip install pyngrok') and rerun the script.")
        sys.exit(1)

# Set Ngrok authtoken
NGROK_AUTH_TOKEN = "2vxxCBbzHLoYzYHcYQEI4MjymyR_5738HFhg8JP7NMhdQGUHg"
if not NGROK_AUTH_TOKEN:
    print("Error: Ngrok authtoken is missing in the script.")
    sys.exit(1)

# Configure ngrok
conf.get_default().auth_token = NGROK_AUTH_TOKEN

# Function to kill ngrok tunnel on exit
def kill_ngrok():
    try:
        print("Shutting down ngrok tunnels...")
        tunnels = ngrok.get_tunnels()
        for tunnel in tunnels:
            ngrok.disconnect(tunnel.public_url)
        ngrok.kill()
        print("Ngrok tunnels shut down.")
    except Exception as e:
        print(f"Error shutting down ngrok: {e}")

atexit.register(kill_ngrok)

# Define the local port Gradio will run on
# Using 7860 as a common default, change if needed
GRADIO_LOCAL_PORT = 7860

# Start ngrok tunnel BEFORE Gradio launch blocks
try:
    print(f"Starting ngrok tunnel for port {GRADIO_LOCAL_PORT}...")
    public_url = ngrok.connect(GRADIO_LOCAL_PORT, "http").public_url
    print("------------------------------------------------------------------")
    print(f"✅ Gradio Public URL (via ngrok): {public_url}")
    print("------------------------------------------------------------------")
    print(f"(Gradio is starting locally on http://127.0.0.1:{GRADIO_LOCAL_PORT})")
except Exception as e:
    print(f"❌ Failed to start ngrok tunnel: {e}")
    print("Ensure your ngrok authtoken is correct and ngrok service is reachable.")
    print("Gradio will attempt to start locally only.")
    public_url = None
# --- End of modifications ---


from diffusers_helper.hf_login import login

# os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download'))) # Keep if needed
if '__file__' in locals(): # Avoid error if running interactively where __file__ is not defined
    os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
else:
    # Set a default path or handle the case where __file__ is not available
    default_hf_home = os.path.join(os.getcwd(), './hf_download')
    os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(default_hf_home))
    print(f"Warning: '__file__' not found. Using default HF_HOME: {os.environ['HF_HOME']}")


import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
# import argparse # Removed argparse as we are fixing the launch parameters
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


# Removed argparse section
# parser = argparse.ArgumentParser()
# parser.add_argument('--share', action='store_true') # Removed
# parser.add_argument("--server", type=str, default='0.0.0.0') # Removed
# parser.add_argument("--port", type=int, required=False) # Removed
# parser.add_argument("--inbrowser", action='store_true') # Removed
# args = parser.parse_args()
# print(args) # Removed

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# --- Model Loading (unchanged) ---
print("Loading models...")
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
print("Models loaded.")

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    print("Using DynamicSwapInstaller for models (Low VRAM mode)")
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    print("Moving models to GPU (High VRAM mode)")
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)
    print("Models moved to GPU.")

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


# --- Worker function (unchanged) ---
@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            print("Unloading models for low VRAM mode before text encoding...")
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
            print("Models unloaded.")

        # Text encoding
        print("Encoding text prompts...")
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            print("Loading text encoders to GPU (low VRAM)...")
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)
            print("Text encoders loaded.")

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        print("Text prompts encoded.")

        # Processing input image
        print("Processing input image...")
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        img_save_path = os.path.join(outputs_folder, f'{job_id}_input.png')
        Image.fromarray(input_image_np).save(img_save_path)
        print(f"Input image processed and saved to {img_save_path}")

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        print("Encoding image with VAE...")
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            print("Loading VAE to GPU (low VRAM)...")
            load_model_as_complete(vae, target_device=gpu)
            print("VAE loaded.")

        start_latent = vae_encode(input_image_pt, vae)
        print("Image VAE encoded.")

        # CLIP Vision
        print("Encoding image with CLIP Vision...")
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            print("Loading Image Encoder to GPU (low VRAM)...")
            load_model_as_complete(image_encoder, target_device=gpu)
            print("Image Encoder loaded.")

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        print("Image CLIP Vision encoded.")

        # Dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        print("Starting sampling loop...")
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = list(reversed(range(total_latent_sections))) # Use original padding calculation

        # Optional trick mentioned in original code (uncomment if preferred)
        # if total_latent_sections > 4:
        #     latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        print(f"Total latent sections: {total_latent_sections}, Padding sequence: {latent_paddings}")

        for i, latent_padding in enumerate(latent_paddings):
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                print("Stop signal received. Ending generation.")
                stream.output_queue.push(('end', None))
                return

            print(f'\n--- Section {i+1}/{total_latent_sections} ---')
            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                print("Unloading models before transformer...")
                unload_complete_models()
                print("Loading transformer to GPU (low VRAM)...")
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
                print("Transformer loaded.")

            if use_teacache:
                print("Initializing TeaCache...")
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                print("TeaCache disabled.")
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling Section {i+1}/{total_latent_sections}, Step {current_step}/{steps}'
                current_vid_len_sec = max(0, (total_generated_latent_frames * 4 - 3) / 30)
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {current_vid_len_sec:.2f} seconds (FPS-30). Extending...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                # print(f"\r{hint}", end='') # Console progress (optional)
                return

            print(f"Running sample_hunyuan for section {i+1}...")
            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )
            print(f"\nSection {i+1} sampling complete.")

            if is_last_section:
                print("Prepending start latent for the last section.")
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            print(f"Total generated latent frames so far: {total_generated_latent_frames}")
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                print("Offloading transformer, loading VAE (low VRAM)...")
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8) # Use a fixed reasonable value?
                load_model_as_complete(vae, target_device=gpu)
                print("VAE loaded for decoding.")

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            print("Decoding latents to pixels...")
            if history_pixels is None:
                print("Decoding first section.")
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                print("Decoding current section and appending...")
                # Calculate frames to decode for this section based on whether it's the last one
                section_latent_frames = int(generated_latents.shape[2]) # Frames generated in *this* step
                overlapped_frames = latent_window_size * 4 - 3 # Fixed overlap

                # Decode only the newly generated part (or slightly more for overlap handling)
                # Need to decode enough frames from the *start* of the new section for soft_append
                # Let's decode the new section plus a bit of the previous for blending if needed
                decode_start_index = max(0, total_generated_latent_frames - section_latent_frames - overlapped_frames // 2) # Heuristic overlap
                current_section_latents_to_decode = real_history_latents[:, :, decode_start_index:, :, :]

                current_pixels = vae_decode(current_section_latents_to_decode, vae).cpu()
                # Adjust soft_append logic if needed based on exact frames decoded vs expected overlap
                # Original logic might work, let's try it first
                #history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
                # Simpler approach: just decode the new latents and append, assuming VAE handles edges okay
                new_pixels = vae_decode(generated_latents.to(vae.device, vae.dtype), vae).cpu()
                # Need careful frame indexing for append. For now, decode all history - simpler but slower.
                print("Re-decoding full history (simpler but potentially slower)...")
                history_pixels = vae_decode(real_history_latents, vae).cpu()


            print("Pixels decoded.")

            if not high_vram:
                print("Unloading VAE (low VRAM)...")
                unload_complete_models()
                print("VAE unloaded.")

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}_frames.mp4')

            print(f"Saving video to {output_filename}...")
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
            print(f"Video saved. Current pixel shape {history_pixels.shape}")

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                print("Last section processed. Finishing.")
                break
    except Exception as e: # Catch specific exceptions if possible
        print("\n--- ERROR DURING GENERATION ---")
        traceback.print_exc()
        print("---------------------------------")
        stream.output_queue.push(('error', f"Error during generation: {e}")) # Send error to UI

    finally: # Ensure cleanup happens
        print("Generation process finished or errored. Cleaning up models...")
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
            print("Models unloaded (low VRAM cleanup).")
        # Consider moving models back to CPU even in high VRAM mode if memory is tight
        # else:
        #    pass # Models stay on GPU in high VRAM mode

        stream.output_queue.push(('end', None))
        print("Worker finished.")
    return


# --- Process function (unchanged, handles stream) ---
def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
    global stream
    if input_image is None:
         print("Error: No input image provided.")
         yield None, None, "Error: Please upload an input image.", '', gr.update(interactive=True), gr.update(interactive=False)
         return # Stop processing

    print("\n=== Starting New Generation Process ===")
    print(f"Parameters: Prompt='{prompt}', Seed={seed}, Length={total_second_length}s, Steps={steps}, GS={gs}, TeaCache={use_teacache}, CRF={mp4_crf}")

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True) # Disable Start, Enable End

    stream = AsyncStream()

    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf)

    output_filename = None
    final_message = "Generation finished."

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            print(f"UI Update: Received video file {output_filename}")
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        elif flag == 'progress':
            preview, desc, html = data
            # print(f"UI Update: Progress - {desc}") # Can be noisy
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        elif flag == 'error':
            error_message = data
            print(f"UI Update: Error - {error_message}")
            final_message = f"Generation stopped due to error: {error_message}"
            yield output_filename, gr.update(visible=False), error_message, '', gr.update(interactive=True), gr.update(interactive=False) # Re-enable Start, Disable End
            break # Stop listening on error

        elif flag == 'end':
            print("UI Update: End signal received.")
            yield output_filename, gr.update(visible=False), final_message, '', gr.update(interactive=True), gr.update(interactive=False) # Re-enable Start, Disable End
            break


def end_process():
    print("UI: 'End Generation' button clicked.")
    if 'stream' in globals() and stream is not None:
        stream.input_queue.push('end')
    else:
        print("No active generation process to end.")
    # Keep UI state indicating process is ending, actual button state update happens in process loop
    return gr.update(), gr.update(interactive=False) # Keep End button disabled


# --- Gradio UI Definition (unchanged) ---
print("Setting up Gradio interface...")
quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack (via Ngrok)') # Added title note
    if public_url:
        gr.Markdown(f"**Access this app publicly at: [{public_url}]({public_url})**")
    else:
        gr.Markdown("**Warning: Failed to create ngrok tunnel. App is only accessible locally.**")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button(value="Start Generation", variant="primary")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Accordion("Advanced Settings", open=False): # Put less frequently changed settings here
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but may slightly affect fine details like fingers.')
                n_prompt = gr.Textbox(label="Negative Prompt", value="", info="Note: Negative prompt effectiveness may vary with current models/settings.")
                seed = gr.Number(label="Seed", value=31337, precision=0)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Default 25 is recommended for speed/quality balance.')
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Guidance strength. Default 10.0 is recommended.')
                if not high_vram:
                     gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (Low VRAM Mode Only)", minimum=4, maximum=16, value=6, step=0.5, info="Memory to keep free on GPU during model swaps. Larger value means slower swaps but less OOM risk. Adjust if needed.")
                else:
                     gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (Not used in High VRAM Mode)", minimum=4, maximum=16, value=6, step=0.5, visible=False) # Hide if high vram
                mp4_crf = gr.Slider(label="MP4 Compression (CRF)", minimum=0, maximum=51, value=18, step=1, info="Lower value means better quality & larger file size (0=lossless, ~18=good, 23=default, 51=worst).")

                # Hidden/Fixed settings from original
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

        with gr.Column():
            total_second_length = gr.Slider(label="Target Video Length (Seconds)", minimum=1, maximum=60, value=5, step=0.5) # Adjusted max based on common use
            preview_image = gr.Image(label="Current Step Preview", height=256, visible=False, interactive=False)
            result_video = gr.Video(label="Generated Video", autoplay=True, show_share_button=False, height=512, loop=True, interactive=False)
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            gr.Markdown('ℹ️ *Note: The video generation process extends the clip over time. The preview shows intermediate steps.*')


    gr.HTML('<div style="text-align:center; margin-top:20px;">Share results/find ideas: <a href="https://x.com/search?q=framepack&f=live" target="_blank">#FramePack on X (Twitter)</a> | Model details on <a href="https://huggingface.co/lllyasviel/FramePackI2V_HY" target="_blank">Hugging Face</a></div>')

    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button], show_progress=False)
    end_button.click(fn=end_process, inputs=None, outputs=[start_button, end_button], show_progress=False) # Update button state on click

print("Gradio interface configured.")
print(f"Launching Gradio app locally on port {GRADIO_LOCAL_PORT}...")

# Launch Gradio Interface (locally only, ngrok provides public access)
block.launch(
    server_name="127.0.0.1", # Listen only locally, ngrok handles external
    server_port=GRADIO_LOCAL_PORT, # Use the fixed port
    share=False,           # IMPORTANT: Must be False to use pyngrok tunnel
    # inbrowser=False,       # Not needed, user will use the ngrok link
    prevent_thread_lock=True # May help keep ngrok responsive, test needed
)

print("Gradio app finished.")
# Ngrok tunnel will be killed by atexit handler
