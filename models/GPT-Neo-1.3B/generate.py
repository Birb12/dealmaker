from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoConfig, pipeline
import torch
from happytransformer import HappyGeneration, GENSettings
import random
import numpy as np

def generate():
    generator = pipeline('text-generation', model='inputmodelhere') #FIXME import model
    print(generator("Product: Airpods, Discount: 20% Off, Description: ", do_sample=True, min_length=50, num_return_sequences=5))
generate()