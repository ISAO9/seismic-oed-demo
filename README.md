# seismic-oed-demo
Interactive simulator demonstrating Physics-Guided AI for high-resolution seismic waveform inversion and Optimal Experimental Design (OED).

Physics-Guided AI for Seismic Inversion & OED Simulator 🌍


Overview
This project is an interactive demonstration application for High-Resolution Multi-Shot Waveform Inversion and Optimal Experimental Design (OED) using Physics-Guided AI (deep learning integrated with physical constraints).

It allows you to experience in real-time, directly in your browser, how AI overcomes the greatest hurdles in seismic exploration and monitoring: "Velocity-Depth Ambiguity" (spatial shifting) and "Shadow Zones" (blind spots of seismic waves).

Key Features
Interactive OED (Optimal Observation Network Design): Users can freely place and configure the positions and number of seismic sources (shots), as well as restrict the number of active sensors (surface and borehole arrays).

Real-time FWI Inference: Based on the configured observation network, a physical waveform simulation using FDTD (Finite-Difference Time-Domain) runs in the background. Using only this limited waveform data along with prior information (Macro-Model), the AI instantly performs an inversion to reconstruct the complex subsurface fault structures.

Ground Truth Comparison: You can compare the AI's predictions against the true geological model (Ground Truth) to clearly see how the resolution of the reconstructed faults changes depending on the density and layout of your observation network (ranging from minimum to optimal setups).

Tech Stack
Machine Learning: PyTorch (Spatial UNet, Prior-Conditioned Architecture)

Physics Simulation: 2D Acoustic Wave FDTD

Web App Framework: Streamlit
