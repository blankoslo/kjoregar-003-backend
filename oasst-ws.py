import os
import json
import asyncio
from pynvml.smi import nvidia_smi
from websockets import connect, exceptions

from oasst import generate

nvsmi = nvidia_smi.getInstance()
cuda_device = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

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
                                  "topic": "gpt-oasst-sft-1-resources",
                                  **status,
                                  }))

        await asyncio.sleep(3)

async def wss(uri):
    async with connect(uri) as websocket:
        # subscribe to worker topic
        await websocket.send(json.dumps({
            "cmd": "sub",
            "topic": "gpt-oasst-sft-1"
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

                        if (msg["stream"]):
                            response = generate(input, msg["previous"], lambda p: asyncio.run(
                                streamToken(msg, p.replace("<|assistant|>", ""))))
                        else:
                            response = generate(input, msg["previous"], lambda r: print(
                                r.replace("<|assistant|>", "\n"), flush=True, end=''))

                        print("response: " + json.loads(json.dumps(response)))

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
