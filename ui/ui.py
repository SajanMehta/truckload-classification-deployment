import gradio as gr
import requests
import io

API_URL_IMAGE = "http://truckload-classification-deployment.railway.internal:8080/predict-image"
API_URL_CSV = "http://truckload-classification-deployment.railway.internal:8080/predict-csv"

print("Calling API at: ", API_URL_IMAGE, API_URL_CSV)

"""def predict_image(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    files = {"image": ("image.png", buf, "image/png")}
    r = requests.post(API_URL_IMAGE, files=files, timeout=60)
    r.raise_for_status()
    return r.json()"""

def predict_image(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    files = {"image": ("image.png", buf, "image/png")}
    r = requests.post(API_URL_IMAGE, files=files, timeout=60)
    r.raise_for_status()
    data = r.json()

    prediction = data.get("predicted_class", "Unknown")
    
    label_out = {prediction: 1.0}

    return label_out

"""def predict_csv(file):
    with open(file.name, "rb") as f:
        files = {"file": ("data.csv", f, "text/csv")}
        r = requests.post(API_URL_CSV, files=files, timeout=300)
    r.raise_for_status()
    return r.json()"""

def predict_csv(file):
    with open(file.name, "rb") as f:
        r = requests.post(API_URL_CSV, files={"file": (file.name, f, "text/csv")}, timeout=300)
    r.raise_for_status()
    data = r.json()

    # 1) Output for gr.JSON
    pred_text = data.get("manifest_prediction", "Unknown")

    is_good = "incorrect" in pred_text.lower()
    score = 0.0 if is_good else 1.0

    label_out = {pred_text: score}

    # 2) Output for gr.Gallery
    gallery_out = [(item["url"], item.get("caption", "")) for item in data.get("gallery", [])]

    return label_out, gallery_out

with gr.Blocks() as demo:
    gr.Markdown("## Run an Image request or a CSV manifest request to the API")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Image Prediction")
            img_in = gr.Image(type="pil", label="Upload an image")
            img_btn = gr.Button("Run Image Prediction")
            img_label = gr.Label(label="Image Prediction Result")
            #img_out = gr.Textbox(label="Image Result")
            img_btn.click(fn=predict_image, inputs=img_in, outputs=img_label)

        with gr.Column():
            gr.Markdown("### CSV Manifest Prediction")
            csv_in = gr.File(label="Upload a CSV", file_types=[".csv"])
            csv_btn = gr.Button("Run CSV")
            csv_out = gr.Label(label="CSV result")
            csv_gallery = gr.Gallery(label="CSV Gallery", columns=2, show_label=False)
            csv_btn.click(fn=predict_csv, inputs=csv_in, outputs=[csv_out, csv_gallery])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
