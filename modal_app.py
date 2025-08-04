from modal import Image, App, asgi_app
import io
import zipfile
import time

app = App("organoid-streamlit-app")

cellpose_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install("cellpose==4.0.6", "opencv-python-headless", "numpy", "pandas", "Pillow", "torch")
    .apt_install("libglib2.0-0", "libsm6", "libxrender1", "libxext6")
)

@app.function(image=cellpose_image, gpu="T4", timeout=600)
@asgi_app()
def run_app():
    from fastapi import FastAPI, UploadFile
    from fastapi.responses import JSONResponse
    from cellpose import models
    import numpy as np
    import pandas as pd
    from PIL import Image as PILImage
    import cv2

    app = FastAPI()


    @app.post("/segment_zip")
    async def segment_zip(file: UploadFile):
        start_time = time.time()
        contents = await file.read()
        zip_file = zipfile.ZipFile(io.BytesIO(contents))

        model = models.Cellpose(gpu=True, model_type="cyto")
        results = []
        annotated_images = []

        for name in zip_file.namelist():
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                with zip_file.open(name) as img_file:
                    img = PILImage.open(img_file).convert("L")
                    img_np = np.array(img)

                    masks, _, _, _ = model.eval(img_np, diameter=None, channels=[0, 0])
                    num_objects = np.max(masks)
                    results.append({"Image": name, "Organoids": int(num_objects)})

                    if num_objects > 0:
                        overlay = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                        overlay[masks > 0] = [0, 255, 0]
                        _, buf = cv2.imencode(".png", overlay)
                        annotated_images.append((name.replace(".png", "_annotated.png"), buf.tobytes()))

        df = pd.DataFrame(results)
        output_zip = io.BytesIO()
        with zipfile.ZipFile(output_zip, "w") as zf:
            zf.writestr("results.csv", df.to_csv(index=False))
            for name, content in annotated_images:
                zf.writestr(name, content)

        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        n_images = len(results)
        n_organoids = sum(r["Organoids"] for r in results)
        n_blank = sum(1 for r in results if r["Organoids"] == 0)

        summary = (
            f"Cellpose processed {n_images} image{'s' if n_images != 1 else ''} and identified {n_organoids} organoid{'s' if n_organoids != 1 else ''}. "
            f"{n_blank} image{'s' if n_blank != 1 else ''} did not have any organoids identified. "
            f"The run completed in {minutes} minute{'s' if minutes != 1 else ''} and {seconds} second{'s' if seconds != 1 else ''}."
        )

        return JSONResponse({
            "zip_bytes": output_zip.getvalue().hex(),
            "summary": summary
        })

    return app

