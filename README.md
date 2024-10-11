# Deep Learning 101 for Audio-based Music Information Retrieval

This repository contains the code and the notebooks for the ISMIR-2024 tutorial "Deep Learning 101 for Audio-based Music Information Retrieval".

It illutstrates the use of Deep Learning components through 3 common MIR tasks
- Multi-Pitch-Estimation
- Cover-Song-Identification
- Auto-Tagging

## TUTO-task-Multi-Pitch-Estimation 

illustrates the use of 
- two inputs: CQT and Harmonic-CQT
- different architecture ConvNet and U-Net
- different variants of convolution: dephtwise-separable, residual-conv

## TUTO-task-Conver-Song-Identification

illustrates the use of
- attention over time and auto-pooling [McFee]
- metric learning using triplet loss
- different triplet mining strategy
- performance measure using ranking

## TUTO-task-Auto-Tagging

illustrates the use of
- different inputs:front-end: Waveform (1D) or Log-Mel-Spectrogram (2D)
- different front-end: 1DConv, TCN, SincNet, 2DConv
- different variants of convolution


