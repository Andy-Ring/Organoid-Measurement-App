import os
# Set environment variable to resolve OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from cellpose import models
import math
import zipfile
import tempfile
import io
import pandas as pd

st.set_page_config(layout="wide")

st.title("Organoid Measurement Tool")

# Initialize calibration variables
pixel_diameter = None
microns_per_pixel = None

st.header("Step 1: Calibration")
col1, col2 = st.columns(2)
with col1:
    calibration_mode = st.radio("Diameter calibration method", ["Manual", "Measure on image"])
with col2:
    scale_mode = st.radio("Scale (μm/pixel)", ["Manual", "Measure on image"])

if calibration_mode == "Manual":
    pixel_diameter = st.slider("Estimated organoid diameter (pixels)", 10, 300, 100)
else:
    test_image_file = st.file_uploader("Upload test image to measure organoid diameter", key="diam_calib")
    if test_image_file:
        image_pil = Image.open(test_image_file).convert("RGB")
        st.image(image_pil, caption="Select organoid diameter", use_container_width=True)
        from streamlit_cropper import st_cropper
        with st.spinner("Please draw a rectangle to estimate the diameter..."):
            crop = st_cropper(image_pil, aspect_ratio=None, box_color='#FF0000')
        pixel_diameter = crop.size[0]
        st.success(f"Estimated organoid diameter: {pixel_diameter} pixels")

if scale_mode == "Manual":
    microns_per_pixel = st.number_input("μm/pixel", value=1.0, min_value=0.01, step=0.01)
else:
    scale_image_file = st.file_uploader("Upload image with scale bar", key="scale_calib")
    if scale_image_file:
        scale_pil = Image.open(scale_image_file).convert("RGB")
        st.image(scale_pil, caption="Select scale bar", use_container_width=True)
        from streamlit_cropper import st_cropper
        scale_crop = st_cropper(scale_pil, aspect_ratio=None, box_color='#00FF00')
        px_len = scale_crop.size[0]
        known_len = st.number_input("Known scale bar length (μm)", value=100.0)
        microns_per_pixel = known_len / px_len
        st.success(f"Calibrated scale: {microns_per_pixel:.3f} μm/pixel")

# Feedback on calibration
if pixel_diameter:
    st.info(f"Using estimated organoid diameter: {pixel_diameter:.1f} pixels")
if microns_per_pixel:
    st.info(f"Using scale: {microns_per_pixel:.3f} μm/pixel")

st.header("Step 2: Upload Images and Choose Mode")
upload_mode = st.radio("Choose mode", ["Single Image", "Batch (ZIP)"])

# Cellpose model settings
model_type = st.selectbox("Model type", ["cyto", "cyto2", "nuclei", "tissuenet", "livecell", "bact_omni", "omninet"])
use_gpu = st.checkbox("Use GPU (if available)", value=True)
model = models.CellposeModel(gpu=use_gpu, model_type=model_type)

results = []

# SINGLE IMAGE MODE
if upload_mode == "Single Image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image_pil)
        st.image(image_np, caption="Original Image", use_container_width=True)

        if st.button("Run Cellpose"):
            if pixel_diameter is None or microns_per_pixel is None:
                st.error("Please complete the calibration step before running the model.")
                st.stop()

            st.info("Cellpose is running. This may take a few moments...")
            with st.spinner("Segmenting image with Cellpose..."):

                masks, flows, styles = model.eval(
                    image_np, diameter=pixel_diameter, channels=[0, 0], resample=True
                )
                contours, _ = cv2.findContours(masks.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                overlay = image_np.copy()

                for i, cnt in enumerate(contours):
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    x, y, w, h = cv2.boundingRect(cnt)
                    circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter != 0 else 0
                    diameter = math.sqrt(4 * area / math.pi)

                    cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
                    cv2.putText(
                        overlay,
                        str(i + 1),
                        (x + w // 2, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    results.append({
                        "Organoid_ID": i + 1,
                        "Area (μm²)": round(area * microns_per_pixel ** 2, 2),
                        "Perimeter (μm)": round(perimeter * microns_per_pixel, 2),
                        "Diameter (μm)": round(diameter * microns_per_pixel, 2),
                        "Circularity": round(circularity, 3)
                    })

                st.image(overlay, caption="Detected Organoids", use_container_width=True)
                df = pd.DataFrame(results)
                st.dataframe(df)
                csv = df.to_csv(index=False).encode()
                st.download_button("Download CSV", csv, "organoid_measurements.csv", "text/csv")

# BATCH MODE
elif upload_mode == "Batch (ZIP)":
    zip_file = st.file_uploader("Upload a ZIP of images", type=["zip"])
    if zip_file:
        if pixel_diameter is None or microns_per_pixel is None:
            st.error("Please complete the calibration step before running the model.")
            st.stop()

        st.info("Running Cellpose on all images. Please wait...")
        with st.spinner("Segmenting batch with Cellpose..."), tempfile.TemporaryDirectory() as tmpdir:
            z = zipfile.ZipFile(zip_file)
            z.extractall(tmpdir)
            image_paths = [os.path.join(tmpdir, f) for f in z.namelist() if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            annotated_images = []
            preview_image = None

            for i, path in enumerate(image_paths):
                image_np = np.array(Image.open(path).convert("RGB"))
                masks, flows, styles = model.eval(image_np, diameter=pixel_diameter, channels=[0, 0], resample=True)
                contours, _ = cv2.findContours(masks.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                overlay = image_np.copy()

                for j, cnt in enumerate(contours):
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    x, y, w, h = cv2.boundingRect(cnt)
                    circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter != 0 else 0
                    diameter = math.sqrt(4 * area / math.pi)

                    cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
                    cv2.putText(
                        overlay,
                        str(j + 1),
                        (x + w // 2, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    results.append({
                        "Image": os.path.basename(path),
                        "Organoid_ID": j + 1,
                        "Area (μm²)": round(area * microns_per_pixel ** 2, 2),
                        "Perimeter (μm)": round(perimeter * microns_per_pixel, 2),
                        "Diameter (μm)": round(diameter * microns_per_pixel, 2),
                        "Circularity": round(circularity, 3)
                    })

                if i == 0:
                    preview_image = overlay.copy()

                _, img_buf = cv2.imencode(".png", overlay)
                annotated_images.append((os.path.basename(path), img_buf))

            if preview_image is not None:
                st.image(preview_image, caption="Example Annotated Image", use_container_width=True)

            df = pd.DataFrame(results)
            st.dataframe(df)
            csv = df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, "batch_organoid_measurements.csv", "text/csv")

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for name, img_buf in annotated_images:
                    zip_file.writestr(name.replace(".png", "_annotated.png"), img_buf)
            st.download_button("Download Annotated Images (ZIP)", zip_buffer.getvalue(), "annotated_images.zip", "application/zip")
