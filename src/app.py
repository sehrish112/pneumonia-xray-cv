import glob
import json
import os
import random

import gradio as gr
import medmnist
from medmnist import INFO

import inference

_, SIZE = inference.get_model()
_test = None

CSS = """
.card {border-radius: 12px; padding: 18px 20px; border: 1px solid #d0d7de; background: #f6f8fa;}
.verdict {font-size: 1.5rem; font-weight: 700; margin: 0 0 4px 0;}
.pneu {color: #b42318;} .norm {color: #067647;}
.bar {height: 14px; border-radius: 7px; background: #e4e7ec; overflow: hidden; margin: 10px 0 4px 0;}
.bar > div {height: 100%; background: #b42318;}
.small {color: #475467; font-size: 0.9rem;}
.warn {border-radius: 10px; padding: 12px 14px; margin-top: 12px; background: #fffaeb;
       border: 1px solid #fec84b; color: #93370d;}
"""


def get_test():
    global _test
    if _test is None:
        cls = getattr(medmnist, INFO["pneumoniamnist"]["python_class"])
        _test = cls(split="test", download=True, size=SIZE, root="data")
    return _test


def analyse(img, threshold):
    if img is None:
        return "<div class='card small'>Upload a chest X-ray, pick an example or load a random test image.</div>", None
    out, overlay = inference.predict(img, with_cam=True)
    p = out["probability_pneumonia"]
    is_pneumonia = p >= threshold
    css = "pneu" if is_pneumonia else "norm"
    verdict = "Pneumonia" if is_pneumonia else "Normal"
    html = (
        f"<div class='card'>"
        f"<p class='small'>Model output</p>"
        f"<p class='verdict {css}'>{verdict}</p>"
        f"<div class='bar'><div style='width:{p * 100:.1f}%'></div></div>"
        f"<p class='small'>Probability of pneumonia: <b>{p * 100:.1f}%</b> "
        f"(decision threshold {threshold:.2f})</p>"
    )
    if out.get("unfamiliar_image"):
        html += ("<div class='warn'><b>Unfamiliar image.</b> This image looks different from the "
                 "training images (distance ratio "
                 f"{out['distance_ratio']}x the 99th-percentile limit). The prediction above is "
                 "probably unreliable.</div>")
    html += "</div>"
    return html, overlay.resize((384, 384))


def random_example():
    ds = get_test()
    img, y = ds[random.randrange(len(ds))]
    label = "Pneumonia" if int(y[0]) == 1 else "Normal"
    return img, f"Dataset label for this image: **{label}**"


def about_text():
    text = ["### About this model",
            "ResNet18 fine-tuned on PneumoniaMNIST (paediatric chest X-rays, 64x64). "
            "A learning project: **not a medical device**."]
    try:
        m = json.load(open("outputs/metrics.json"))
        text.append(f"**Internal test set** ({m['n_test']} images): AUC {m['auc']:.3f}, "
                    f"sensitivity {m['recall_sensitivity']:.3f}, specificity {m['specificity']:.3f}.")
    except OSError:
        pass
    try:
        e = json.load(open("outputs/external_validation.json"))
        text.append(f"**External test** (adult NIH images, {e['n_images']} images): AUC {e['auc']:.3f}, "
                    f"sensitivity {e['sensitivity_at_0.5']:.3f}, specificity {e['specificity_at_0.5']:.3f}. "
                    "The model does not generalise to a different population.")
    except OSError:
        pass
    text.append("**Limitations:** tiny images, one source, children only, over-confident probabilities, "
                "no clinical validation. The heatmap (Grad-CAM) is coarse and only a visual hint.")
    text.append("The *unfamiliar image* warning compares each image's features with those of the training "
                "images; it is a helpful signal, not a guarantee.")
    return "\n\n".join(text)


examples = sorted(glob.glob("docs/examples/*.png"))

with gr.Blocks(title="Chest X-ray Pneumonia Classifier") as demo:
    gr.Markdown(
        "# Chest X-ray Pneumonia Classifier\n"
        "A PyTorch project by **Sehrish Masood**. Upload a chest X-ray and see the model's "
        "prediction and where it looked.\n\n"
        "**Research prototype on a public benchmark. Not for medical use.**"
    )
    with gr.Tabs():
        with gr.Tab("Classifier"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(type="pil", label="Chest X-ray", image_mode="L", height=380)
                    random_btn = gr.Button("Load random test image", variant="primary")
                    truth = gr.Markdown()
                    threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Decision threshold",
                                          info="Lower = fewer missed cases, more false alarms")
                    if examples:
                        gr.Examples(examples=examples, inputs=inp, label="Examples from the test set: four correct, the last one a model mistake")
                with gr.Column(scale=1):
                    card = gr.HTML()
                    cam = gr.Image(label="Grad-CAM: regions that influenced the prediction",
                                   height=380, interactive=False)
        with gr.Tab("About the model"):
            gr.Markdown(about_text())

    inp.upload(lambda: "*Your own image: no dataset label available.*", None, truth)
    inp.clear(lambda: "", None, truth)
    random_btn.click(random_example, None, [inp, truth])
    gr.on(triggers=[inp.change, threshold.release], fn=analyse, inputs=[inp, threshold], outputs=[card, cam])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
