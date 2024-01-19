from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import os
import torch

output_dir = "mistral/results/"

# Where to load model results
final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")

# Load the entire model on the GPU 0
reloaded_model = AutoPeftModelForCausalLM.from_pretrained(
    final_checkpoint_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map='auto',
)
reloaded_tokenizer = AutoTokenizer.from_pretrained(final_checkpoint_dir)

# Merge the LoRA and the base model
merged_model = reloaded_model.merge_and_unload()

model_name = 'Mixtral-8x7B-blended'

merged_model.save_pretrained('/data/models/' + model_name)
reloaded_tokenizer.save_pretrained('/data/models/' + model_name)
# merged_model.push_to_hub("jeffreykthomas/" + model_name)
# reloaded_tokenizer.push_to_hub("jeffreykthomas/" + model_name)

if __name__ == '__main__':
    pass
