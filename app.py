import gradio as gr
from chain import analyze_vacancy


def process_vacancy(vacancy_text):
    if not vacancy_text or len(vacancy_text.strip()) < 20:
        return "Job description (20 symbols min)"
    try:
        return analyze_vacancy(vacancy_text)
    except Exception as e:
        return f"error: {str(e)}"


with gr.Blocks(title="HIRE ME") as demo:
    gr.HTML('<h1 style="text-align:center; font-size:64px;">ARE WE A MATCH?</h1>')
    gr.HTML('<p style="text-align:center; color:#ff2200;">My name is Denis. I have talent. You have a work. Drop the job description and let let the model decide.</p>')

    vacancy_input = gr.Textbox(
        label="job description",
        placeholder="your job description...",
        lines=6
    )

    submit_btn = gr.Button("CHECK SCORE", variant="primary")

    output = gr.Textbox(
        label="result",
        lines=12,
        interactive=False
    )

    submit_btn.click(
        fn=process_vacancy,
        inputs=vacancy_input,
        outputs=output
    )

    gr.HTML('<p style="text-align:center; color:#888; margin-top:40px;">hexiwell [at] gmail [dot] com</p>')

demo.launch()