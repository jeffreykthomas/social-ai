# Running the Predictive Agent Playground

This guide explains how to stand up the predictive agent simulation with the new local reasoning model and episode logging pipeline.

## 1. Prerequisites
- Python 3.9 or newer (3.10 recommended for recent PyTorch wheels)
- Git LFS/SSH access for any private models you plan to reuse
- GPU with ≥24 GB VRAM for 14B-class models, or plan to run a quantized/CPU build
- Hugging Face access token if you need to pull gated model weights

## 2. Prepare a Virtual Environment
```bash
python3 -m venv myvenv
source myvenv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install Python Dependencies
The legacy requirements cover Django and simulation tooling; you also need the transformers stack for local inference.
```bash
pip install -r requirements.txt
pip install "torch>=2.1" --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft bitsandbytes datasets safetensors
```
Adjust the PyTorch wheel URL for your CUDA/CPU platform as needed. On Apple Silicon, install the nightly metal build instead of the CUDA wheel.

## 4. Download or Mount a Reasoning Model
Pick an instruction-tuned reasoning model (e.g., `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`). Options:
- `transformers` cache: `python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'); AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-14B')"`
- Local directory: place the model under `models/deepseek-r1` and point the config at that path.
- Quantized GGUF: run it via `llama.cpp` or `text-generation-inference` and update the client to call the HTTP endpoint (requires extra wiring).

Ensure the model fits your hardware. For low-memory nodes, consider 4-bit LoRA-ready variants or smaller checkpoints.

## 5. Configure the LLM Client
Edit `reverie/backend_server/config/need_config.yaml`:
```yaml
llm:
  model_name: "/abs/path/to/deepseek-r1"   # or the HF repo ID
  device: "cuda"                            # "cuda", "cuda:0", "cpu", etc.
  dtype: "bfloat16"
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.9
  repetition_penalty: 1.05
```
Set `device` to `cpu` if you cannot load the model on GPU. If you run a quantized model without bfloat16 support, change `dtype` to `float16` or leave it `null`.

## 6. Start the Environment Server
```bash
cd environment/frontend_server
python manage.py runserver
```
Keep the Django server running; it serves the Smallville map and monitoring UI at http://localhost:8000/.

## 7. Launch the Predictive Agent Backend
In a second terminal:
```bash
cd reverie/backend_server
python reverie.py
```
Follow the prompts to fork an existing simulation or start a new one. The backend uses the configured local model for dialogue and writes logs under `reverie/backend_server/logs/episodes`.

To rotate logs per run, call the helper before stepping agents:
```python
from persona.persona_manager import get_manager
manager = get_manager()
manager.start_new_session("session_2024_08_15")
```

## 8. Running Updates
The main update loop lives in `persona_manager.update_agents()`. During development you can drive it manually:
```python
import asyncio
from persona.persona_manager import run_simulation_step

asyncio.run(run_simulation_step())
```
Integrate this into your orchestration or scheduler if you want continuous operation.

## 9. Collecting Data for Fine-Tuning
All speech and high-level actions are buffered to JSONL files in `reverie/backend_server/logs/episodes/SESSION.jsonl`. Each record includes:
- Agent name and utterance/action metadata
- Snapshot of need states
- Recent thoughts and nearby agents (for speech)

Use these logs as supervised fine-tuning data or to compute RL rewards. Extend the logger via `EpisodeLogger.log_reward` once you finalize a reward function.

## 10. Common Troubleshooting
- **Model OOM:** Lower `max_new_tokens`, switch to a smaller model, or quantize weights.
- **Slow inference:** Enable FlashAttention / xFormers via environment flags, or run the model with `accelerate launch --config_file ...` and point `model_name` to a local TCP endpoint.
- **Missing assets:** Re-run the initial setup instructions in `README.md` (copy `utils.py`, download static assets) if the map or personas fail to load.
- **No logs:** Confirm `reverie/backend_server/logs/episodes` is writable and that you have called `manager.start_new_session` if you need distinct files per run.

With these steps complete, the playground will use your local reasoning model for dialogue, and you’ll have a reusable data trail for reinforcement learning or supervised refinement.
