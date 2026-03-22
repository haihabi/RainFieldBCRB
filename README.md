# Estimation Performance Bound For Rain Field Reconstruction
This repository contains research code and materials for two papers on estimation performance bounds for opportunistic rain field reconstruction using Commercial Microwave Links (CMLs) and designated sensors.

## Authors

- Hai Victor Habi (IEEE Student Member)
- Hagit Messer (Life Fellow, IEEE)

Affiliation: School of Electrical and Computer Engineering, Tel Aviv University, Israel.

## Paper 1

**Title:** How does Cramer-Rao Bound Analysis Benefit Opportunistic Rain Field Reconstruction

**Abstract:** In recent years, a novel approach for 2-D near ground rain fields reconstruction based on signal level measurements from Commercial Microwave Links (CMLs) gains increasing interest globally. While opportunistic use of numerous existing communication links from, e.g., backhaul cellular networks, as virtual rain sensors is appealing, it raises several practical and theoretical challenges, mainly in the leading application of 2-D rain field reconstruction based on line-projection sensors. In this paper we show how the use of various CRB-type bounds, and in particular the misspecified CRB, can help in analyzing the cost of the widely used approximation of modeling a CML as a single-point projection sensor (a virtual rain gauge).

**Accepted to:** IEEE ICASSP 2026

## Paper 2

**Title:** Optimal Allocation of Auxiliary Designated Sensors for Opportunistic Rain Field Reconstruction

**Abstract:** Two-dimensional (2-D) field reconstruction presents a challenge in opportunistic Integrated Sensing and Communication (ISAC), where weather phenomena, such as rain fields or humidity, are sensed and mapped through their effect on wireless communication. In this paper, we analyze the inherent limitations in reconstruction accuracy by deriving the Bayesian Cramér-Rao Bound (BCRB) for field mapping modeled by 2-D B-splines for two sensor types: line (e.g., commercial microwave links) and point (e.g., rain gauges) sensors. In particular, we utilize this bound to optimize the allocation of additional designated point sensors within an existing, given opportunistic sensor network, where sensors are randomly located.

**Accepted to:** IEEE Statistical Signal Processing Workshop (SSP), 2025

## How to run

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Create the results directory (required by `main.py`):

```bash
mkdir -p results
```

4. Run the main experiment:

```bash
python3 main.py
```

## Citation (BibTeX)

```bibtex
@inproceedings{habi2026crb,
  title     = {How does Cramer-Rao Bound Analysis Benefit Opportunistic Rain Field Reconstruction},
  author    = {Habi, Hai Victor and Messer, Hagit},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026},
  organization = {IEEE},
  note      = {Accepted}
}

@inproceedings{habi2025optimal,
  title     = {Optimal Allocation of Auxiliary Designated Sensor for Opportunistic Rain Field Reconstruction},
  author    = {Habi, Hai Victor and Messer, Hagit},
  booktitle = {2025 IEEE Statistical Signal Processing Workshop (SSP)},
  pages     = {131--135},
  year      = {2025},
  organization = {IEEE}
}
```
