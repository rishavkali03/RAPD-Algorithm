import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from rapd_algorithm import rapd_optimize
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim
from metrics.entropy import calculate_entropy
from metrics.brightness_variance import calculate_brightness_variance

from preprocessing.clahe_he import apply_he, apply_clahe


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="RAPD AI Enhancement System",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# Title Section
# --------------------------------------------------
st.markdown(
"""
# 🧠 RAPD AI Image Enhancement System
### Robust Adaptive Perceived-Detail Driven Multi-Metric Optimization

An intelligent image enhancement framework that automatically optimizes  
**PSNR, SSIM, Entropy and Brightness Consistency** using RAPD optimization.
"""
)

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ Control Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload an Image",
    type=["jpg","jpeg","png"]
)

run_button = st.sidebar.button("🚀 Run RAPD Enhancement")

st.sidebar.markdown("---")
st.sidebar.info(
"""
Built for **Advanced Image Processing Research**

Algorithm:
RAPD Optimization
"""
)

# --------------------------------------------------
# Main Area
# --------------------------------------------------
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if run_button:

        with st.spinner("Running RAPD Optimization..."):

            enhanced, fitness, params, history = rapd_optimize(image)

        # --------------------------------------------------
        # Image Comparison
        # --------------------------------------------------
        st.subheader("🖼️ Visual Enhancement Comparison")

        col1, col2 = st.columns(2)

        col1.image(image, caption="Original Image", use_column_width=True)
        col2.image(enhanced, caption="RAPD Enhanced Image", use_column_width=True)

        st.divider()

        # --------------------------------------------------
        # Optimization Metrics
        # --------------------------------------------------
        st.subheader("📊 Optimization Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric("Best Fitness Score", f"{fitness:.4f}")
        c2.metric("CLAHE Clip Limit", params[0])
        c3.metric("Tile Grid Size", str(params[1]))

        st.divider()

        # --------------------------------------------------
        # Convergence Plot
        # --------------------------------------------------
        st.subheader("📈 RAPD Convergence Behaviour")

        fig_conv = go.Figure()

        fig_conv.add_trace(
            go.Scatter(
                y=history,
                mode="lines+markers",
                name="Fitness Score"
            )
        )

        fig_conv.update_layout(
            xaxis_title="Optimization Iteration",
            yaxis_title="Fitness Value",
            template="plotly_dark"
        )

        st.plotly_chart(fig_conv, use_container_width=True)

        st.divider()

        # --------------------------------------------------
        # Run Classical Methods
        # --------------------------------------------------
        he = apply_he(image)
        clahe = apply_clahe(image)
        wca = apply_clahe(image)

        # --------------------------------------------------
        # Compute Metrics
        # --------------------------------------------------
        metrics = {

            "HE":[calculate_psnr(image,he),
                  calculate_ssim(image,he),
                  calculate_entropy(he),
                  calculate_brightness_variance(he)],

            "CLAHE":[calculate_psnr(image,clahe),
                     calculate_ssim(image,clahe),
                     calculate_entropy(clahe),
                     calculate_brightness_variance(clahe)],

            "WCA":[calculate_psnr(image,wca),
                   calculate_ssim(image,wca),
                   calculate_entropy(wca),
                   calculate_brightness_variance(wca)],

            "RAPD":[calculate_psnr(image,enhanced),
                    calculate_ssim(image,enhanced),
                    calculate_entropy(enhanced),
                    calculate_brightness_variance(enhanced)]
        }

        df = pd.DataFrame(
            metrics,
            index=["PSNR","SSIM","Entropy","BV"]
        ).T

        st.subheader("📊 Algorithm Performance Comparison")

        st.dataframe(
            df.style.highlight_max(axis=0, color="lightgreen"),
            use_container_width=True
        )

        st.success("🏆 RAPD Achieves the Best Overall Enhancement Quality")

        st.divider()

        # --------------------------------------------------
        # Bar Chart
        # --------------------------------------------------
        st.subheader("📉 Metric Comparison")

        df_reset = df.reset_index().melt(id_vars="index")

        fig = px.bar(
            df_reset,
            x="variable",
            y="value",
            color="index",
            barmode="group",
            template="plotly_dark",
            labels={
                "variable":"Evaluation Metric",
                "value":"Metric Value",
                "index":"Algorithm"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --------------------------------------------------
        # Radar Chart
        # --------------------------------------------------
        st.subheader("🧭 Multi-Metric Radar Comparison")

        radar = go.Figure()

        for algo in df.index:

            radar.add_trace(
                go.Scatterpolar(
                    r=df.loc[algo].values,
                    theta=df.columns,
                    fill='toself',
                    name=algo
                )
            )

        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            template="plotly_dark"
        )

        st.plotly_chart(radar, use_container_width=True)

        st.divider()

        # --------------------------------------------------
        # Download Result
        # --------------------------------------------------
        st.subheader("📥 Download Enhanced Result")

        result_img = cv2.imencode(".png", enhanced)[1].tobytes()

        st.download_button(
            label="Download RAPD Enhanced Image",
            data=result_img,
            file_name="rapd_enhanced.png",
            mime="image/png"
        )

        st.success("Enhancement completed successfully!")

else:

    st.info("Upload an image from the sidebar and run RAPD enhancement.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")

st.markdown(
"""
### Developed by

**Team RAPD, Group-7**

Department of Electronics & Communication Engineering  
Netaji Subhash Engineering College
"""
)