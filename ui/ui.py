import gradio as gr
import requests
import io

API_URL_IMAGE = "http://truckload-classification-deployment.railway.internal:8080/predict-image"
API_URL_CSV = "http://truckload-classification-deployment.railway.internal:8080/predict-csv"

print("Calling API at: ", API_URL_IMAGE, API_URL_CSV)

def predict_image(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    files = {"image": ("image.png", buf, "image/png")}
    r = requests.post(API_URL_IMAGE, files=files, timeout=60)
    r.raise_for_status()
    return r.json()

def predict_csv(file):
    files = {"file": ("data.csv", file, "text/csv")}
    r = requests.post(API_URL_CSV, files = files, timeout = 300)
    r.raise_for_status()
    return r.json

# demo = gr.Interface(
#     fn=predict_image,
#     inputs=gr.Image(type="pil"),
#     outputs=gr.JSON(),
#     title="Image Classifier Demo"
# )

with gr.Blocks() as demo:
    gr.Markdown("## Run an Image request or a CSV manifest request to the API")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Image Prediction")
            img_in = gr.Image(type="pil", label="Upload an image")
            img_btn = gr.Button("Run Image Prediction")
            img_out = gr.Textbox(label="Image Result")
            img_btn.click(fn=predict_image, inputs=img_in, outputs=img_out)

        with gr.Column():
            gr.Markdown("### CSV Manifest Prediction")
            csv_in = gr.File(label="Upload a CSV", file_types=[".csv"])
            csv_btn = gr.Button("Run CSV")
            csv_out = gr.Textbox(label="CSV result")
            csv_btn.click(fn=predict_csv, inputs=csv_in, outputs=csv_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

