import gradio as gr
import requests
import io

API_URL = "http://truckload-classification-deployment.railway.internal:8080/predict-image"

print("Calling API at: ", API_URL)

def predict(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    files = {"image": ("image.png", buf, "image/png")}
    r = requests.post(API_URL, files=files, timeout=60)
    r.raise_for_status()
    return r.json()

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(),
    title="Image Classifier Demo"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

