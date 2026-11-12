import random

import gradio as gr
import medmnist
from medmnist import INFO

import inference

_, SIZE = inference.get_model()
_test = None


def get_test():
    global _test
    if _test is None:
        cls = getattr(medmnist, INFO["pneumoniamnist"]["python_class"])
        _test = cls(split="test", download=True, size=SIZE, root="data")
    return _test


def predict(img):
    if img is None:
        return None, None
    out, overlay = inference.predict(img, with_cam=True)
    p = out["probability_pneumonia"]
    return {"Pneumonia": p, "Normal": 1 - p}, overlay


def random_example():
    ds = get_test()
    img, y = ds[random.randrange(len(ds))]
    return img, f"Ground truth label: {'Pneumonia' if int(y[0]) == 1 else 'Normal'}"


with gr.Blocks(title="Chest X-ray Pneumonia Classifier") as demo:
    gr.Markdown("# Chest X-ray Pneumonia Classifier\n"
                f"**{inference.DISCLAIMER}** Works best with images from that dataset; "
                "arbitrary X-rays may give unreliable results.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Chest X-ray", image_mode="L")
            with gr.Row():
                run = gr.Button("Predict", variant="primary")
                rnd = gr.Button("Load random test image")
            truth = gr.Markdown()
        with gr.Column():
            out_label = gr.Label(label="Prediction")
            out_cam = gr.Image(label="Grad-CAM (where the model looks)")
    run.click(predict, inp, [out_label, out_cam])
    rnd.click(random_example, None, [inp, truth]).then(predict, inp, [out_label, out_cam])

if __name__ == "__main__":
    demo.launch()
