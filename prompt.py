from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

model = AutoModelForSeq2SeqLM.from_pretrained("snrspeaks/t5-one-line-summary")
tokenizer = AutoTokenizer.from_pretrained("snrspeaks/t5-one-line-summary")

#magicTokenizer = AutoTokenizer.from_pretrained(
#    "Gustavosta/MagicPrompt-Stable-Diffusion")

#magicModel = AutoModelForCausalLM.from_pretrained(
#    "Gustavosta/MagicPrompt-Stable-Diffusion")

def generate(input):
    # summary
    input_ids = tokenizer.encode(
        "summarize: " + input, return_tensors="pt", add_special_tokens=True)
    generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_length=50,
                                   repetition_penalty=2.5, length_penalty=1, early_stopping=True, num_return_sequences=3)
    preds = [tokenizer.decode(g, skip_special_tokens=True,
                              clean_up_tokenization_spaces=True) for g in generated_ids]

    # add prompt magic to the summary
    #magic_input_ids = magicTokenizer.encode(preds[0], return_tensors="pt")
    #gen_tokens = magicModel.generate(
    #    input_ids=magic_input_ids,
    #    max_new_tokens=55
    #)

    #resp = magicTokenizer.decode(gen_tokens[0]).replace("<|endoftext|>", "")

    return preds[0] + " fine art. trending on artstation. masterpiece."
