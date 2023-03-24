import torch
from transformers import LogitsWarper

# inspired by comment in https://github.com/tloen/alpaca-lora/issues/51#issuecomment-1474506296
class CallbackLogitsWarper(LogitsWarper):
    def __init__(self, tokenizer, callback):
        self.tokenizer = tokenizer
        self.callback = callback
        self.res_tokens = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        self.res_tokens.append(input_ids[0][-1])
        result = self.tokenizer.decode(input_ids[0][-1])
        self.callback(result)
        return scores


def evaluate(prompt, model, tokenizer, configuration, callback):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    generation_output = model.generate(
        input_ids=input_ids,
        **configuration,
        logits_processor=[CallbackLogitsWarper(tokenizer, callback)],
    )

    output = ''

    for s in generation_output.sequences:
        output += tokenizer.decode(s)

    return output
