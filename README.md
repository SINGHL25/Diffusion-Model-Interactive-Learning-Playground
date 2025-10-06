# Diffusion-Model-Interactive-Learning-Playground
```PLAIN TEXT
ddpm-ddim-playground/
├── src/
│   ├── models/
│   │   ├── ddpm.py         # DDPM model implementation
│   │   └── ddim.py         # DDIM model implementation
│   ├── diffusion_utils.py  # Noise schedules, sampling functions, etc.
│   ├── ui_components.py    # Reusable Streamlit UI functions
│   ├── visualizer.py       # Functions for Plotly/Matplotlib visualizations
│   └── gemini_assistant.py # (Optional) Integration for AI commentary
├── data/                   # Example images, pre-trained weights (if any)
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project description and usage
├── .streamlit/             # Streamlit configuration (optional)
│   └── config.toml

```
# DDPM-DDIM-Playground

An interactive learning playground for Denoising Diffusion Probabilistic Models (DDPM) and Denoising Diffusion Implicit Models (DDIM). This Streamlit application allows users to explore the forward and reverse diffusion processes, tweak parameters, and visualize the impact on image generation.

## 🎯 Features

*   **Model Selection:** Switch between DDPM and DDIM.
*   **Parameter Control:** Adjust `num_steps`, `noise_level`, `beta_schedule`, `guidance_scale`, and `DDIM eta`.
*   **Real-time Visualizations:**
    *   Step-by-step forward (noising) and reverse (denoising) process.
    *
