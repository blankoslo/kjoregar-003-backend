import os
import json
import asyncio
from pynvml.smi import nvidia_smi
from websockets import connect, exceptions

from sd import generate
from sdavatar import generateAvatar

nvsmi = nvidia_smi.getInstance()
cuda_device = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

def vram_status():
    nvq = nvsmi.DeviceQuery('memory.used, memory.total')
    usage = nvq['gpu'][cuda_device]['fb_memory_usage']
    used = usage['used']
    total = usage['total']
    unit = usage['unit']

    return 'GPU memory ' + str(round(used/ total * 100)) + '% (' + str(round(used)) + ' / ' + str(round(total)) + ' ' + unit + ')'

async def resources(ws):
    while True:
        status = {
            "type": "SystemResourceUpdate",
            "resources": vram_status(),
        }
        await ws.send(json.dumps({"cmd": "pub",
            "topic": "sd-2-1-resources",
            **status,
        }))

        await asyncio.sleep(3)

async def wss(uri):
    async with connect(uri) as websocket:
        # subscribe to worker topic
        await websocket.send(json.dumps({
            "cmd": "sub",
            "topic": "sd-2-1"
        }))

        # report resources every 3s
        asyncio.create_task(resources(websocket))

        # process subscribed events
        while True:
            try:
                message = await websocket.recv()
                msg = json.loads(message)

                def process():
                    if (msg["type"] == 'ChatImageCreate'):
                        input = msg["human"]
                        response = generate(input)

                        update = {
                            "cmd": "pub",
                            "topic": msg["replyTopic"],
                            "type": "ChatImageRobotResponse",
                            "id": msg["id"],
                            "robot": response,
                        }

                        return update
                    elif (msg["type"] == 'ChatAvatarCreate'):
                        response = generateAvatar(None)

                        update = {
                            "cmd": "pub",
                            "topic": msg["replyTopic"],
                            "type": "ChatAvatarRobotResponse",
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
