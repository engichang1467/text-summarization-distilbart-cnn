import gradio as gr
from transformers import pipeline

get_completion = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize(input):
  output = get_completion(input)
  return output[0]['summary_text']

demo = gr.Interface(fn=summarize, 
                    inputs=[gr.Textbox(label="Text to summarize", lines=6)], 
                    outputs=[gr.Textbox(label="Result", lines=3)],
                    title="Text summarization with distilbart-cnn",
                    description="Summarize any text using the `sshleifer/distilbart-cnn-12-6` model under the hood!")
demo.launch()