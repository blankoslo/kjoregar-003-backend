import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/llama2-13b-orca-8k-3319", use_fast=False)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, decode_kwargs={ "skip_special_tokens": False })
model = AutoModelForCausalLM.from_pretrained("OpenAssistant/llama2-13b-orca-8k-3319", load_in_8bit=True, device_map="auto")

system_message = 'You are an AI assistant. You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. You might need to use additional knowledge to answer the question.'
template = '''{context}<|prompter|>{text}</s><|assistant|>'''

storedContext = ''

while True:
    msg = input("\n>>> ").strip()

    if (msg[0] == '+'):
        storedContext = ''
        print("reset")
        continue

    inputWithContext = template.format(text = msg, context = storedContext)
    fewShotText = f"""<|system|>{system_message}</s>""" + inputWithContext 

    configuration = {
        "do_sample": True,
        "temperature": 0.25,
        "top_p": 0.95,
        "top_k": 0,
        "max_new_tokens": 256
    }

    inputs = tokenizer(fewShotText, return_tensors="pt").to("cuda")
    generation_kwargs = dict(inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=2048, streamer=streamer)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    storedContext += inputWithContext
    for new_text in streamer:
        storedContext += new_text.replace("\n", "")
        print(new_text, end='', flush=True)
