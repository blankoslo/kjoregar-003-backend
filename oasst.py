import stream
from transformers import AutoTokenizer, AutoModelForCausalLM 

model_name = "openassistant/oasst-sft-1-pythia-12b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={'': 0},
    load_in_8bit=True
)

template = '''{context}<|prompter|>{text}<|endoftext|><|assistant|>'''

def generate(input, context, streamCb):
    history = ""

    if (len(context) > 0):
        history = "".join(map(lambda p: "<|prompter|>" +
                          p["human"] + "<|endoftext|><|assistant|>" + p["robot"], context)) + "<|endoftext|>"

    prompt = template.format(context=history, text=input)

    configuration = {
        "do_sample": True,
        "temperature": 0.25,  # 0.15
        "top_k": 80,
        "top_p": 0.90,
        "max_new_tokens": 1500,
        "return_dict_in_generate": True,
        "output_scores": True,
        "repetition_penalty": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
    }

    response = stream.evaluate(
        prompt,
        model,
        tokenizer,
        configuration,
        streamCb
    ).strip()

    response = response[len(prompt):]  # cut question
    # cut continuation

    return response.split("<|prompter|>")[0].replace("<|endoftext|>", "").strip()
