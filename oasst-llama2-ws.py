import os
import json
import asyncio
import torch
from pynvml.smi import nvidia_smi
from websockets import connect, exceptions
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/llama2-13b-orca-v2-8k-3166", use_fast=False)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
model = AutoModelForCausalLM.from_pretrained("OpenAssistant/llama2-13b-orca-v2-8k-3166", torch_dtype=torch.float16, device_map="auto")

nvsmi = nvidia_smi.getInstance()
cuda_device = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

system_message = 'You are an AI assistant that follows instruction extremely well. Help as much as you can.'
template = '''{context}<|prompter|>{text}</s><|assistant|>'''

def vram_status():
    nvq = nvsmi.DeviceQuery('memory.used, memory.total')
    usage = nvq['gpu'][cuda_device]['fb_memory_usage']
    used = usage['used']
    total = usage['total']
    unit = usage['unit']

    return 'GPU memory ' + str(round(used / total * 100)) + '% (' + str(round(used)) + ' / ' + str(round(total)) + ' ' + unit + ')'

async def resources(ws):
    while True:
        status = {
            "type": "SystemResourceUpdate",
            "resources": vram_status(),
        }
        await ws.send(json.dumps({"cmd": "pub",
                                  "topic": "gpt-oasst-llama2-resources",
                                  **status,
                                  }))

        await asyncio.sleep(3)

async def wss(uri):
    async with connect(uri) as websocket:
        # subscribe to worker topic
        await websocket.send(json.dumps({
            "cmd": "sub",
            "topic": "gpt-oasst-llama2"
        }))

        # report resources every 3s
        asyncio.create_task(resources(websocket))

        # process subscribed events
        while True:
            try:
                message = await websocket.recv()
                msg = json.loads(message)

                def process():
                    async def streamToken(msg, token):
                        return await websocket.send(json.dumps({
                            "cmd": "pub",
                            "topic": msg["replyTopic"],
                            "type": "ChatMessageRobotStreamResponse",
                            "id": msg["id"],
                            "robot": token,
                        }))

                    if (msg["type"] == 'ChatMessageCreate'):
                        input = msg["human"]

                        previous = "".join(map(lambda p: "<|prompter|>" +
                          p["human"].replace("\n", "") + "</s><|assistant|>" + p["robot"].replace("\n", "") + "</s>", msg["previous"] )) 

                        inputWithContext = template.format(text = input, context = previous)
                        fewShotText = f"""<|system|>{system_message}</s>""" + inputWithContext 

                        #print("### start input ###")
                        #print(fewShotText)
                        #print("### end input ###")

                        inputs = tokenizer(fewShotText, return_tensors="pt").to("cuda")
                        generation_kwargs = dict(inputs, temperature=0.5, do_sample=True, top_p=0.85, top_k=0, max_new_tokens=2048, streamer=streamer)
                        thread = Thread(target=model.generate, kwargs=generation_kwargs)
                        thread.start()

                        response = ''

                        for new_text in streamer:
                            response += new_text
                            asyncio.run(streamToken(msg, new_text.replace("</s>", "")))
                            
                        #print("response: " + json.loads(json.dumps(response)))

                        update = {
                            "cmd": "pub",
                            "topic": msg["replyTopic"],
                            "type": "ChatMessageRobotResponse",
                            "id": msg["id"],
                            "robot": response,
                        }

                        return update

                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(None, process)

                await websocket.send(json.dumps(res, ensure_ascii=True))

            except json.decoder.JSONDecodeError:
                print("message was not valid json")
            except exceptions.ConnectionClosedError:
                break

asyncio.run(wss("ws://127.0.0.1:3030"))
