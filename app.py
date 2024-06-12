################################################################
# 修改自 openai_api.py框架，换用 Quart 框架
# 使用前先修改模型参数。
# 支持并发，多会话，动态加载状态，会话保存
# api_url 填写 http://127.0.0.1:8088 即可测试
# ---参数基本符合 OpenAI 的接口，用任意 OpenAI 客户端均可，无需填写 api key 和 model 参数---
# 添加了部分其它 api
###############################################################

# -*- coding: utf-8 -*-
import time, random, re, sys, os, signal, json, tqdm
from utils import prxxx, gen_echo, clean_symbols
from quart import Quart, websocket, request
from hypercorn.config import Config
from hypercorn.asyncio import serve
import asyncio
from server_pipeline import (
    RWKVChater,
    RWKVNicknameGener,
    RWKVGroupChater,
    RWKVInterruptException,
    process_default_state,
)
from typing import Dict, Tuple
from config import MODEL_STATE_NAME, APP_BIND, APP_AUTOSAVE_TIME, APP_TEST_MESSAGE

with open("help.min.html", "r") as f:
    flask_help = f.read()

random.seed(time.time())
chaters: Dict[str, RWKVChater] = {}
group_chaters: Dict[str, RWKVGroupChater] = {}
nicknameGener = RWKVNicknameGener()

app = Quart(__name__)


def restart():
    # app.shutdown()
    python = sys.executable
    prxxx("### Restart ! ###")
    os.execl(python, python, *sys.argv)


def stop(signal=None, frame=None):
    # app.shutdown()
    prxxx("### STOP ! ###")
    sys.exit()


async def save_chaters_state():
    for id in tqdm.tqdm(chaters, desc="Save chater", leave=False, unit="chr"):
        await asyncio.sleep(0)
        await chaters[id].save_state(id, q=True)
    for id in tqdm.tqdm(
        group_chaters, desc="Save grpup chater", leave=False, unit="chr"
    ):
        await asyncio.sleep(0)
        await group_chaters[id].save_state(id, q=False)


def save_chaters_state_sync():
    for id in tqdm.tqdm(chaters, desc="Save chater", leave=False, unit="chr"):
        if chaters[id].need_save:
            chaters[id].state.save_sync(id)
            prxxx(f"Save state   name: {id}")
    for id in tqdm.tqdm(
        group_chaters, desc="Save grpup chater", leave=False, unit="chr"
    ):
        if group_chaters[id].need_save:
            group_chaters[id].state.save_sync(id)
            prxxx(f"Save state   name: {id}")


async def time_to_save():
    while True:
        for i in range(APP_AUTOSAVE_TIME):  # 防止卡服务器关闭
            await asyncio.sleep(1)
        await save_chaters_state()
        prxxx("Autosave all chater")


async def chat(
    message: str,
    id: str = "-b2bi0JgEhJru87HTcRjh9vdT",
    user: str = "木子",
    nickname: str = "墨子",
    state: str = MODEL_STATE_NAME,
    debug=False,
    echo=None,
) -> Tuple[str, bool]:
    id = clean_symbols(id)

    echo = gen_echo()
    if not id in chaters:
        prxxx()
        chaters[id] = RWKVChater(id, state_name=state)
        await chaters[id].init_state()

    prxxx()
    prxxx(f" #    Chat   id: {id} | user: {user} | echo: {echo}")
    prxxx(f" #    -M->[{message}]-{echo}")
    answer, original, is_want_to_say = await chaters[id].chat(
        message=message, chatuser=user, nickname=nickname, debug=debug
    )

    prxxx()
    prxxx(f" #    Chat   id: {id} | nickname: {nickname} | echo: {echo}")
    prxxx(f" #    {echo}-[{original}][{'FY'[is_want_to_say]}]<-O-")
    prxxx(f" #    {echo}-[{answer}]<-A-")

    # 如果接受到的内容为空，则给出相应的回复
    if answer.isspace() or len(answer) == 0:
        answer = "喵喵喵？"
    return answer, is_want_to_say


async def group_chat_send(
    message: str,
    id: str = "-b2bi0JgEhJru87HTcRjh9vdT",
    user: str = "木子",
    state: str = MODEL_STATE_NAME,
    echo=None,
) -> None:  # -> Tuple[str, bool]:
    id = clean_symbols(id)

    if len(message) == 0:
        return

    echo = gen_echo()
    if not id in group_chaters:
        prxxx()
        group_chaters[id] = RWKVGroupChater(id, state_name=state)
        await group_chaters[id].init_state()

    prxxx()
    prxxx(f" #    Send Gchat   id: {id} | user: {user} | echo: {echo}")
    prxxx(f" #    -M->[{message}]-{echo}")
    await group_chaters[id].send_message(message=message, chatuser=user)


async def group_chat_get(
    id: str = "-b2bi0JgEhJru87HTcRjh9vdT",
    nickname: str = "墨子",
    state: str = MODEL_STATE_NAME,
    echo=None,
) -> Tuple[str, bool]:
    id = clean_symbols(id)

    echo = gen_echo()
    if not id in group_chaters:
        prxxx()
        group_chaters[id] = RWKVGroupChater(id, state_name=state)
        await group_chaters[id].init_state()

    answer, original, is_want_to_say = await group_chaters[id].get_answer(
        nickname=nickname
    )

    prxxx()
    prxxx(f" #    Get gchat   id: {id} | nickname: {nickname} | echo: {echo}")
    prxxx(f" #    {echo}-[{original}][{'FY'[is_want_to_say]}]<-O-")
    prxxx(f" #    {echo}-[{answer}]<-A-")

    # 如果接受到的内容为空，则给出相应的回复
    if answer.isspace() or len(answer) == 0:
        answer = "喵喵喵？"
    return answer, is_want_to_say


async def gen_nickname(name: str, echo=None):
    echo = gen_echo()
    prxxx()
    prxxx(f" #    GenNickname   echo: {echo}")
    prxxx(f" #    -N->[{name}]-{echo}")
    nickname, _ = await nicknameGener.gen_nickname(name)

    prxxx()
    prxxx(f" #    GenNickname   echo: {echo}")
    prxxx(f" #  {echo}-[{nickname}]<-N-")

    # 如果接受到的内容为空，则给出相应的回复
    if nickname.isspace() or len(nickname) == 0 or nickname == "None":
        nickname = name
    return nickname


async def reset_state(id: str, echo=None):
    id = clean_symbols(id)
    flag = False
    if id in chaters:
        await chaters[id].reset_state()
        flag = True
    if id in group_chaters:
        await group_chaters[id].reset_state()
        flag = True
    return flag

async def get_history(id: str, echo=None):
    id = clean_symbols(id)
    if id in chaters:
        return await chaters[id].get_history()
    if id in group_chaters:
        return await group_chaters[id].get_history()
    return ""


@app.route("/chat", methods=["POST", "GET"])
async def R_chat():
    if request.method == "GET":
        kwargs = request.args
    elif request.method == "POST":
        kwargs = await request.form
    try:
        answer, is_want_to_say = await chat(**kwargs)
        return {"message": answer, "is_want_to_say": is_want_to_say, "state": "ok"}
    except RWKVInterruptException:
        return {"state": "interrupted"}


@app.route("/group_chat_send", methods=["POST", "GET"])
async def R_group_chat_send():
    if request.method == "GET":
        kwargs = request.args
    elif request.method == "POST":
        kwargs = await request.form
    await group_chat_send(**kwargs)
    return {"state": "ok"}


@app.route("/group_chat_get", methods=["POST", "GET"])
async def R_group_chat_get():
    if request.method == "GET":
        kwargs = request.args
    elif request.method == "POST":
        kwargs = await request.form
    answer, is_want_to_say = await group_chat_get(**kwargs)
    return {"message": answer, "is_want_to_say": is_want_to_say, "state": "ok"}


@app.route("/nickname", methods=["POST", "GET"])
async def R_nickname():
    if request.method == "GET":
        kwargs = request.args
    elif request.method == "POST":
        kwargs = await request.form
    nickname = await gen_nickname(**kwargs)
    return {"nickname": nickname, "state": "ok"}


@app.route("/reset_state", methods=["GET"])
async def R_reset_state():
    if request.method == "GET":
        kwargs = request.args
    elif request.method == "POST":
        kwargs = await request.form
    flag = await reset_state(**kwargs)
    return {"state": "ok" if flag else "a?"}


@app.route("/save_state", methods=["GET"])
async def R_save_state():
    if request.method == "GET":
        kwargs = request.args
    elif request.method == "POST":
        kwargs = await request.form
    flag = await save_chaters_state(**kwargs)
    return {"state": "ok"}

@app.route("/get_history", methods=["GET"])
async def R_get_history():
    if request.method == "GET":
        kwargs = request.args
    elif request.method == "POST":
        kwargs = await request.form
    history = await get_history(**kwargs)
    return {"state": "ok", "history": history}

@app.route("/restart", methods=["GET"])
async def R_restart():
    if request.args["passwd_gkd"] == "ihAVEcODE":
        await app.shutdown()
        restart()
    return {"state": "fuck you!"}


@app.route("/stop", methods=["GET"])
async def R_stop():
    if request.args["passwd_gkd"] == "ihAVEcODE":
        await app.shutdown()
        stop()
    return {"state": "fuck you!"}


@app.route("/", methods=["GET"])
async def R_index():
    return flask_help


@app.websocket("/chat")
async def W_chat():
    while True:
        data = json.loads(await websocket.receive())
        """
        data{
            id
            message
            username
            nickname*
            default_state*
            echo*
            debug*
        }
        """
        try:
            answer, is_want_to_say = await chat(**data)
            await websocket.send(
                json.dumps(
                    {
                        "message": answer,
                        "is_want_to_say": is_want_to_say,
                        "state": "OK",
                        "echo": data.get("echo", ""),
                    }
                )
            )
        except RWKVInterruptException:
            await websocket.send(
                json.dumps({"state": "interrupted", "echo": data.get("echo", "")})
            )


@app.websocket("/group_chat")
async def W_group_chat():
    while True:
        data = json.loads(await websocket.receive())
        """
        data{
            action
            id
            message+
            username+
            nickname*
            default_state*
            echo*
        }
        """
        if data["action"] == "send":
            await group_chat_send(**data)
            await websocket.send(
                json.dumps({"state": "OK", "echo": data.get("echo", "")})
            )
        elif data["action"] == "get":
            answer, is_want_to_say = await group_chat_get(**data)
            await websocket.send(
                json.dumps(
                    {
                        "message": answer,
                        "is_want_to_say": is_want_to_say,
                        "state": "OK",
                        "echo": data.get("echo", ""),
                    }
                )
            )
        else:
            await websocket.send(
                json.dumps({"state": "A?", "echo": data.get("echo", "")})
            )


# @app.before_serving
async def before_serving():
    # app.add_background_task(time_to_save)
    await process_default_state()
    await nicknameGener.init_state()
    init = RWKVChater("init")
    chaters["init"] = init
    await init.init_state()
    prxxx(f"State size: {init.state.size()}")
    await init.reset_state()
    await chat(
        **{
            "id": "init",
            "message": APP_TEST_MESSAGE,
            "user": "测试者",
        }
    )
    prxxx()
    prxxx(" *#*   RWKV！高性能ですから!   *#*")
    prxxx()
    prxxx("Web api server start!\a")
    prxxx(f"API   bind: {APP_BIND}")


@app.after_serving
async def after_serving():
    save_chaters_state_sync()
    global chaters, group_chaters
    del chaters, group_chaters
    prxxx("### STOP ! ###")


async def main():
    await before_serving()  # fix: timeout wen shutup
    config = Config()
    config.bind = APP_BIND
    config.use_reloader = True
    config.loglevel = "debug"
    """
    for i in tqdm.trange(99999):
        await group_chat_send({"id":"ggtgg","message":"uuuu","user":"yyyyy"})
    """
    await serve(app, config)


if __name__ == "__main__":
    asyncio.run(main())
