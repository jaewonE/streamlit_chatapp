import os
import json
from time import time
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from model import GemmaModel, MockModel

st.set_page_config(page_title="ChatApp", page_icon="🤖", layout="wide")


@st.cache_data
def fontRegistered():
    font_dirs = ['./fonts']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:

        fm.fontManager.addfont(font_file)

    fm._load_fontmanager(try_read_cache=False)


if not os.path.exists("history"):
    os.makedirs("history")

with open("css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


able_model_list = ["mock1", "mock2", "gemma"]


def init_app():
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = []
    if 'model' not in st.session_state:
        st.session_state['model'] = load_model(
            able_model_list[0], max_length=30)
        # st.session_state['model'] = None
    if 'last_use_model_name' not in st.session_state:
        st.session_state['last_use_model_name'] = able_model_list[0]
    if 'history' not in st.session_state:
        st.session_state['history'] = None


def init_json() -> dict:
    st.session_state['history'] = {
        'time': time(),
        'model_name': st.session_state['model_name'],
        'messages': []
    }


def load_model(model_name, **kwargs):
    model = None
    with st.spinner("Loading model..."):
        if model_name == able_model_list[0]:
            model = MockModel(model_name="Mock1", delay=1,
                              init_delay=1, **kwargs)
        elif model_name == able_model_list[1]:
            model = MockModel(model_name="Mock2", delay=1,
                              init_delay=1, **kwargs)
        elif model_name == able_model_list[2]:
            model = GemmaModel(**kwargs)
    if model == None:
        st.error("Please select a model.")
        st.stop()
    return model


init_app()


# Side Bar

# New Chat button
clear_button = st.sidebar.button("New Conversation", key="new_chat")

# 새로운 채팅을 시작할 경우 모든 정보를 초기화한다.
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = []
    st.session_state['model_name'] = []
    st.session_state['history'] = None


model_name = st.sidebar.selectbox(
    "Choose a model:", able_model_list, index=0)
max_length = st.sidebar.slider(
    min_value=30, max_value=300, label="Max length")

if model_name:
    if st.session_state['last_use_model_name'] != model_name:
        st.session_state['model'] = load_model(
            model_name, max_length=max_length)
        st.session_state['last_use_model_name'] = model_name

if max_length:
    st.session_state['model'].set_max_length(max_length)

if st.session_state['model'] is None:
    st.error("model is not loaded.")
    st.stop()


def on_click_history(file_path: str):
    with open(file_path, 'r') as f:
        history = json.load(f)
        print("Load history file: " + file_path)
        st.session_state['generated'] = [m['content']
                                         for m in history['messages'] if m['role'] == "assistant"]
        st.session_state['past'] = [m['content']
                                    for m in history['messages'] if m['role'] == "user"]
        st.session_state['messages'] = history['messages']
        st.session_state['model_name'] = history['model_name']
        st.session_state['history'] = history


# 채팅 기록
st.sidebar.title("Chat History")
for record in [os.path.join("history", f) for f in os.listdir("history") if f.endswith(".json")]:
    with open(record, 'r') as f:
        history = json.load(f)
        state = f"{datetime.fromtimestamp(history['time']).strftime('%Y/%m/%d %H:%M:%S')} | {history['model_name'][0]}"
        st.sidebar.markdown(
            f"""<div style="font-size: 12px; font-weight: 400; color: gray; padding-left: 3px; margin-top: 4px; margin-bottom: 2px;">{state}</div>""", unsafe_allow_html=True)
        st.sidebar.button(history['messages'][0]['content'],
                          key=state, on_click=on_click_history, args=(record,))


# 컨테이너 생성
response_container = st.container()
container = st.container()

# 유저 입력란.
user_input = st.chat_input(placeholder="Your Input",
                           key=None, max_chars=None, disabled=False)

# 유저 입력이 있을 경우 past에 유저 입력을, generated에 response를 추가한다.
if user_input:
    st.session_state['messages'].append(
        {"role": "user", "content": user_input})
    print(f'User Input: {user_input}')
    st.session_state['past'].append(user_input)

    output = st.session_state['model'].generate_response(user_input)
    print("Response: ", end="")
    print(output.replace("\n", " "))
    st.session_state['messages'].append(
        {"role": "assistant", "content": output})
    st.session_state['generated'].append(output)

    st.session_state['model_name'].append(model_name)


# 유저 입력을 화면에 보여주고 progress bar를 보여준다.
# if st.session_state['past'] or st.session_state['generated']:
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message2 = st.chat_message("user", avatar="🧑‍💻")
            message2.write(st.session_state["past"][i])
            message3 = st.chat_message("assistant", avatar="🤖")
            message3.write(st.session_state["generated"][i])

    # save message to json
    if st.session_state['history'] == None:
        init_json()
    st.session_state['history']['messages'] = st.session_state['messages']
    with open(f"history/{st.session_state['history']['time']}.json", 'w') as f:
        json.dump(st.session_state['history'], f)
