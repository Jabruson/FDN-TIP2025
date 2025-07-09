
# Fourier-based Decoupling Network for Joint Low-Light Image Enhancement and Deblurring (FDN)

[![Paper](https://img.shields.io/badge/paper-IEEE-blue)](https://ieeexplore.ieee.org/document/YOUR_PAPER_ID) [![code](https://img.shields.io/badge/code-GitHub-green.svg)](https://github.com/Jabruson/FDN-TIP2025)

This is the official PyTorch implementation of the paper **"Fourier-based Decoupling Network for Joint Low-Light Image Enhancement and Deblurring"**. (IEEE TIP 2025)

This paper has been accepted by IEEE TRANSACTIONS ON IMAGE PROCESSING.

*By Luwei Tu, Jiawei Wu, Chenxi Wang, Deyu Meng, and Zhi Jin.*

---

## Abstract

Nighttime handheld photography is often simultaneously affected by low light and blur degradations due to object motion and camera shake. Previous methods typically design specific modules to restore the degradations in the spatial domain independently. However, the interdependence of low light and blur degradations in the spatial domain makes it difficult for these approaches to effectively decouple the degradations. In this paper, we observe that in the Fourier domain, low light and blur degradations can be represented independently in the amplitude and phase of the image. We discover that low light degradation exhibits distinct characteristics across different frequency bands in amplitude, while blur degradation is characterized by phase correlation. Leveraging these insights, we propose a Fourier-based Decoupling Network (FDN) for joint low-light image enhancement and deblurring. Experimental results demonstrate that our method achieves state-of-the-art performance on both synthetic and real-world datasets and exhibits significantly sharper edges.



