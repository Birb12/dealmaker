from cleandataset import DealsDataset
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import gc


torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.resize_token_embeddings(len(tokenizer)) # token embeddings must be resized since we added special tokens
preferred_output = pd.read_csv('datafortrain.csv')
generated_prompts = []

for index, row in preferred_output.iterrows():
    generate = "Product: " + row['PRODUCT'] + ", Discount: " + row['DISCOUNT'] + ", Description: " + row['OUTPUT'] 
    generated_prompts.append(generate)


max_length = max([len(tokenizer.encode(prompt)) for prompt in generated_prompts])
dataset = DealsDataset(generated_prompts, tokenizer, max_length=max_length)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
gc.collect()
training_args = TrainingArguments(output_dir='./results', num_train_epochs=5, logging_steps=500, # Various memory and speed improvements, adjust for your own needs
                                  evaluation_strategy='steps', gradient_accumulation_steps=4, eval_steps=10, save_total_limit=1,
                                  per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_checkpointing=True,
                                  warmup_steps=100, weight_decay=0.01, fp16=True, load_best_model_at_end=True ,flogging_dir='./logs')
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                              'attention_mask': torch.stack([f[1] for f in data]),
                                                              'labels': torch.stack([f[0] for f in data])})

trainer.train()
trainer.save_model()
tokenizer.save_pretrained("./tokenizer")
