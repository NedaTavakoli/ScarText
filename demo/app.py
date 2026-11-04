"""Local Gradio demo: classify myocardial scar from a cardiac MRI report.

The built-in reports are synthetic. The app runs the paper's rule-based NLP and, when an Ollama server is
connected, the paper's zero-shot LLM, and shows whether they agree. Few-shot (opt-in, CMR_DEMO_FEWSHOT=1)
adds the paper's few-shot LLM, whose examples are real expert-labeled reports sent only to the Ollama server.
"""

import os
import re
import zipfile

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

import engine
import llm_client
from templates import EXTRA_CASES, TEMPLATES

# Online (Hugging Face Spaces sets SPACE_ID): only the rules run, and visitors are told not to upload patient data.
PUBLIC = bool(os.environ.get("SPACE_ID")) or os.environ.get("CMR_DEMO_PUBLIC") == "1"
NOTICE = (
    '<div style="color:#6B6B6B;font-size:13px;margin-bottom:4px">'
    + ("Research use only. Not for clinical use. The built-in reports are synthetic. Reports you paste or upload are "
       "processed on this server and not stored, but <b>do not upload identifiable patient information</b>."
       if PUBLIC else
       "Research use only. Not for clinical use. The built-in reports are synthetic. "
       "Paste a real report only on a computer approved for patient data.")
    + "</div>"
)
METHOD_NOTE = (
    "The NLP class comes from the study's rule-based code. Highlights show the terms it looks for, and a highlighted term "
    "does not always decide the class. On 264 expert-labeled reports, the NLP scored 90.2% and zero-shot Qwen3-32B 89.8%. "
    "The NLP's score is not independent, because its rules were developed with the expert readers. Where the two agreed "
    "(85.6% of reports), 97.8% were correct, so the demo flags disagreements for expert review."
)
if PUBLIC:
    METHOD_NOTE += (" Online, Qwen3-32B runs through Hugging Face Transformers (FP8 weights) rather than the paper's Ollama "
                    "setup, so its answers can differ slightly from the paper's runs.")
CASES = {**{engine.CLASS_DISPLAY[c]: TEMPLATES[c] for c in engine.CLASSES}, **EXTRA_CASES}
FIRST = engine.CLASS_DISPLAY[engine.CLASSES[0]]

ONLINE_LLM_NOTE = ("The online demo runs the rules only. To add the LLMs, run the quickstart notebook or this demo "
                   "on your own computer with Ollama (see the GitHub repository).")
ONLINE_FS_NOTE = ("Few-shot needs expert-labeled example reports, which are not shared. Run the notebook or this demo "
                  "on your own computer with your own labeled reports.")
if PUBLIC:
    # Online: zero-shot Qwen3-32B on Hugging Face ZeroGPU (CMR_SPACE_LLM=0 turns it off). If it cannot load, the
    # rules run alone, and a placeholder GPU function keeps ZeroGPU from stopping an app that declares none.
    try:
        if os.environ.get("CMR_SPACE_LLM", "1") == "0":
            raise ImportError("turned off")
        import zerogpu_llm
        backend = zerogpu_llm.ZeroGPUBackend()
    except Exception as e:
        print("Online LLM not loaded:", type(e).__name__, str(e)[:200])
        backend = llm_client.NoBackend(ONLINE_LLM_NOTE)
        try:
            import spaces

            @spaces.GPU
            def _zerogpu_placeholder():
                return None
        except ImportError:
            pass
    fewshot = llm_client.NoBackend(ONLINE_FS_NOTE)
else:
    backend = llm_client.get_backend()
    fewshot = llm_client.get_fewshot_backend()   # opt-in: CMR_DEMO_FEWSHOT=1
FEWSHOT_ON = fewshot.configured
FS_SOURCE, FS_NOTE = "Few-shot LLM", ""


def read_uploaded_report(path):
    """Return the text of an uploaded .txt or .docx file, then delete Gradio's temporary copy."""
    try:
        if path.lower().endswith(".docx"):
            with zipfile.ZipFile(path) as z:
                xml = z.read("word/document.xml").decode("utf8")
            paragraphs = re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)
            return "\n".join("".join(re.findall(r"<w:t[^>]*>([^<]*)", p)) for p in paragraphs)
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except (OSError, KeyError, zipfile.BadZipFile):
        return ""
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def classify(text):
    text = text or ""
    result = engine.run_nlp(text)
    nlp_html = engine.nlp_card(result)
    report_html = engine.render_report(text)
    has_text = bool(text.strip())

    waiting = engine.llm_card(llm_client.LLMResult(None, "Waiting for a report."))
    fs_waiting = engine.llm_card(llm_client.LLMResult(None, "Waiting for a report."), source=FS_SOURCE)
    if not has_text:
        yield nlp_html, waiting, fs_waiting, report_html
        return
    zs_html = engine.llm_card(None) if backend.configured else engine.llm_card(backend.classify(text), result["label"])
    fs_html = engine.llm_card(None, source=FS_SOURCE) if FEWSHOT_ON else fs_waiting
    yield nlp_html, zs_html, fs_html, report_html
    if backend.configured:
        zs_html = engine.llm_card(backend.classify(text), result["label"])
        yield nlp_html, zs_html, fs_html, report_html
    if FEWSHOT_ON:
        fs_html = engine.llm_card(fewshot.classify(text), result["label"], source=FS_SOURCE, agree_note=FS_NOTE)
        yield nlp_html, zs_html, fs_html, report_html


def load_case(name):
    return CASES.get(name, ""), "Synthetic report, written for this demo."


def load_upload(path):
    text = read_uploaded_report(path) if path else ""
    note = "" if text.strip() else "No text could be read from this file. Use a .txt or .docx file."
    return text, None, note


def build():
    with gr.Blocks(title="CMR Scar Classifier Demo", delete_cache=(3600, 3600)) as demo:
        gr.HTML(NOTICE)
        gr.Markdown("# Myocardial scar classification from cardiac MRI reports")
        with gr.Row():
            with gr.Column(scale=5):
                case = gr.Dropdown(list(CASES), value=FIRST, label="Synthetic report")
                report = gr.Textbox(label="Report text", lines=14, max_lines=18)
                note = gr.Markdown()
                with gr.Row():
                    run = gr.Button("Classify", variant="primary", min_width=80)
                    upload = gr.UploadButton("Upload report", file_types=[".txt", ".docx"], type="filepath", min_width=80)
                    clear = gr.Button("Clear", min_width=80)
            with gr.Column(scale=6):
                with gr.Row(equal_height=True):
                    nlp_out = gr.HTML()
                    llm_out = gr.HTML()
                    fs_out = gr.HTML(visible=FEWSHOT_ON)
                report_out = gr.HTML()
                gr.Markdown(METHOD_NOTE)

        outputs = [nlp_out, llm_out, fs_out, report_out]
        private = {"api_visibility": "private"}
        run.click(classify, report, outputs, **private)
        case.input(load_case, case, [report, note], **private).then(classify, report, outputs, **private)
        report.input(lambda: (None, ""), None, [case, note], **private)
        upload.upload(load_upload, upload, [report, case, note], **private).then(classify, report, outputs, **private)
        clear.click(lambda: ("", None, ""), None, [report, case, note], **private).then(classify, report, outputs, **private)
        demo.load(load_case, case, [report, note], **private).then(classify, report, outputs, **private)
    return demo


if __name__ == "__main__":
    # Loopback only on a personal computer; all interfaces only when hosted (Hugging Face Spaces). Never a share link.
    build().queue().launch(
        server_name="0.0.0.0" if PUBLIC else "127.0.0.1",
        server_port=int(os.environ.get("CMR_DEMO_PORT", "7860")),
        share=False,
        inbrowser=not PUBLIC and os.environ.get("CMR_DEMO_NO_BROWSER") != "1",
        footer_links=["settings"],
        theme=gr.themes.Default(font=gr.themes.Font("system-ui"), font_mono=gr.themes.Font("monospace")),
    )
