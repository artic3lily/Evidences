import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/chat"

def chat_fn(message, history):
    # history is a list of [user, bot] pairs; API holds memory server-side
    r = requests.post(API_URL, json={"message": message}, timeout=60)
    r.raise_for_status()
    ans = r.json()["answer"]
    return ans

with gr.Blocks() as demo:
    gr.Markdown("# Chatbot Tutor with Memory")
    chat = gr.ChatInterface(fn=chat_fn)
    demo.launch()

if __name__ == "__main__":
    demo.launch()