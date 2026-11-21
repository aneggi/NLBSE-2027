# NLBSE Competition Paper (4 Pages) — Generation & Execution Guide

This document provides the operational instructions, scientific positioning, and methodological constraints to rapidly generate a **4-page IEEE conference paper** for the NLBSE competition track. It is designed to be used alongside the project's source scripts, experimental logs, official challenge guidelines, and a reference paper (e.g., Borsani et al.) for structural and formatting alignment.

---

## 1. Scientific Positioning ("Green AI & Lifecycle Efficiency")

Unlike standard competition submissions that focus solely on maximizing F1-score through heavy data augmentation or large model ensembles, this work evaluates **computational and energy efficiency across the entire model lifecycle (Fine-Tuning + Inference)** using compact encoder architectures.

* **Core Thesis:** Lightweight encoder models (e.g., `ettin-encoder-17m` or decoupled/cascaded setups as implemented in this repository) achieve competitive classification performance while keeping fine-tuning energy consumption and inference GFLOPs remarkably low—making them practical for real-world, continuous CI/CD integration.
* **Reference to Existing Literature & Heavy Baselines:** Do **not** run direct experimental comparisons against heavy monolithic models (such as `CodeBERT` or `RoBERTa-base`). Instead, use heavy baselines and previous competition entries (e.g., Borsani et al.) purely as **literature examples and contextual benchmarks** to illustrate the standard accuracy-first paradigm versus our efficiency-first perspective. Use Borsani et al. strictly as a template for the IEEE layout, section structure, and official dataset/task description.

---

## 2. Available Hardware Data & Methodological Rules

To ensure full methodological rigor and prevent reviewer pushback regarding physical measurements, strictly adhere to the following definitions based on the actual data extracted from the scripts and logs in this repository:

1. **Classification Performance (Official Challenge Metrics):**
   * Report Precision, Recall, and F1-score for each target class/category required by the challenge, as well as the overall average, extracted directly from the evaluation logs.

2. **Fine-Tuning Phase (HPC Cluster — NVIDIA V100):**
   * Since direct hardware power counters (NVML/RAPL) are unavailable on the Slurm nodes, report the exact **wall-clock training time** ($t_{\text{train}}$) on the **NVIDIA V100 GPU** (nominal TDP = **300 W** for PCIe / **250 W** depending on the exact node specification—verify from cluster specs or state the standard 300 W V100 TDP) and compute the **upper-bound energy consumption estimate**:
     $$E_{\text{est}} \text{ (Wh)} = P_{\text{TDP}} \text{ (W)} \times t_{\text{train}} \text{ (h)}$$
   * *Mandatory Methodological Disclaimer:* Explicitly state in the text that this metric represents a theoretical upper bound (*worst-case proxy*, standard in Green AI literature when direct power sensors are unexposed) rather than a direct hardware wattmeter measurement. **Do not include $\text{CO}_2$ emission estimates.**

3. **Inference Phase (Google Colab — NVIDIA T4):**
   * Do **not** convert Colab metrics into Watts or Joules. Extract the **GFLOPs** (hardware-agnostic computational complexity) and **inference execution time / latency** measured on the **Google Colab NVIDIA T4 instance** directly from the repository scripts/logs.
   * Report GFLOPs per sample and average inference latency (ms/sample), highlighting the low computational footprint during runtime execution.

---

## 3. Target Paper Structure (4 Pages, IEEE Format)

* **Title & Abstract (~150 words):** Clearly highlight *lightweight encoders*, *computational and energy efficiency* (V100 fine-tuning + T4 inference GFLOPs), and the specific NLBSE challenge task.
* **Section I — Introduction (~0.75 pages):**
  * Context of automated software engineering classification (e.g., SATD / issue / comment classification).
  * Limitation of accuracy-only competition approaches (e.g., heavy models and complex augmentation pipelines like those seen in prior literature), which introduce high computational overhead in frequent CI/CD runs.
  * Paper contribution: evaluating a compact encoder approach across both fine-tuning energy upper-bound (V100) and runtime computational cost (T4 GFLOPs).
* **Section II — Approach & Experimental Setup (~1 page):**
  * Concise description of the proposed pipeline derived directly from the repository scripts (model architecture, tokenization/input handling, confidence thresholds or multi-stage logic if present, and training hyperparameters).
  * Hardware & Measurement Setup: NVIDIA V100 (HPC/Slurm) for fine-tuning time and TDP-based energy estimation; Google Colab NVIDIA T4 for inference GFLOPs and latency profiling.
* **Section III — Results & Discussion (~1.5 pages):**
  * **Classification Results Table:** Precision, Recall, and F1-score across all challenge categories.
  * **Efficiency Analysis Table:** Training time on V100, estimated upper-bound training energy ($E_{\text{est}}$ in Wh), and inference GFLOPs / latency on T4. Discuss how these numbers compare qualitatively to the resource demands of standard large transformer baselines reported in literature.
  * **Threats to Validity (1 concise paragraph):** Use of nominal TDP as a worst-case proxy for V100 training energy (no direct NVML/RAPL telemetry), potential runtime variance in shared Google Colab T4 environments for latency, and dataset generalizability.
* **Section IV — Related Work & Conclusion (~0.5 pages):**
  * Brief overview of Green AI / Green Software Engineering principles and prior NLBSE competition approaches (citing Borsani et al. and other relevant entries as examples of accuracy-focused/augmentation strategies).
  * Conclusion and future integration into energy-aware CI/CD pipelines.

---

## 4. Instructions for the LLM / Writing Agent

When generating the LaTeX manuscript (`IEEEtran` conference template) use this folder to generate the LaTex for then import in Overaleaf "paper":

1. **Inspect the Repository First:** Parse all Python scripts (training, inference, GFLOPs profiling) and output logs in the project directory to extract exact model names, hyperparameters, V100 training durations, T4 GFLOPs/latencies, and F1-scores. **Never fabricate numbers**; insert `[TBD]` if a specific value is missing from the logs.
2. **Follow Reference Formatting:** Use the attached PDF/LaTeX of Borsani et al. and the official challenge guidelines to strictly enforce the **4-page limit** (including references), dataset naming conventions, and table styles.
3. **Verify All BibTeX References:** Include only real, verifiable citations (no hallucinated papers) covering:
   * Foundational Green AI & TDP/FLOPs estimation methodology (e.g., Schwartz et al., *Green AI*, CACM 2020; Strubell et al., ACL 2019).
   * The exact encoder model(s) used in the scripts (e.g., `ettin` / ModernBERT technical report or relevant architecture paper).
   * Official NLBSE competition overview papers and prior submissions (such as Borsani et al.) cited purely as literature examples.
4. **Writing Style:** Crisp, direct, engineering-focused, and free of filler. Every claim must be anchored to the exact metrics extracted from the repository.