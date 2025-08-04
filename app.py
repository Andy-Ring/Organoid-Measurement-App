import streamlit as st
import requests
import zipfile
import io
import pandas as pd

st.set_page_config(layout="wide")
st.title("Organoid Measurement App (GPU-powered via Modal)")

uploaded_file = st.file_uploader("Upload a ZIP file of images", type="zip")

if uploaded_file:
    with st.spinner("Uploading and running Cellpose..."):

        files = {"file": uploaded_file.getvalue()}
        response = requests.post("https://andy-ring--organoid-streamlit-app-run-app.modal.run/segment_zip", files=files)

        if response.status_code == 200:
            data = response.json()
            summary = data["summary"]
            zip_bytes = bytes.fromhex(data["zip_bytes"])
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
                file_names = z.namelist()
                csv_name = [f for f in file_names if f.endswith(".csv")][0]
                csv_bytes = z.read(csv_name)
                df = pd.read_csv(io.BytesIO(csv_bytes))

                st.success("Processing complete!")
                st.write("### Summary")
                st.info(summary)

                st.write("### Results Table")
                st.dataframe(df)

                st.write("### Annotated Images")
                cols = st.columns(3)
                image_files = [f for f in file_names if f.endswith("_annotated.png")]
                for i, fname in enumerate(image_files):
                    with cols[i % 3]:
                        st.image(z.read(fname), caption=fname, use_column_width=True)

                zip_download = io.BytesIO(zip_bytes)
                st.download_button(
                    label="Download Annotated Results (ZIP)",
                    data=zip_download,
                    file_name="organoid_results.zip",
                    mime="application/zip"
                )
        else:
            st.error(f"Error from Modal API: {response.status_code}")


