Multitask Code Classification

Multitask Code Classification
Competition Overview

Automatically identifying low-quality and insecure code is an important software-maintenance challenge. Relevant evidence is often distributed across two complementary artefacts: source code and the natural-language comments written by developers.

The NLBSE’27 Code Classification Competition invites participants to build a model that uses both artefacts to address two related tasks jointly:

    vulnerability classification for source code; and
    Self-Admitted Technical Debt (SATD) classification for developer comments.

Models are assessed not only by their predictive performance, but also by their execution time and computational cost in a shared evaluation environment.

Participants must develop one multitask model that receives a composite code-and-comment artefact and jointly predicts two binary labels. Each input contains a normalised source-code function together with its associated developer comments.

The expected output is a two-element binary vector in the following order:
Output 	Meaning
[0, 0] 	No vulnerability and no SATD
[1, 0] 	Vulnerability and no SATD
[0, 1] 	No vulnerability and SATD
[1, 1] 	Vulnerability and SATD

The primary predictive objective is to maximise the arithmetic mean of the vulnerability and SATD F1-scores. Participants are encouraged to investigate representations and integration strategies that exploit the complementary information in code and comments.
Dataset

The competition uses a multi-annotated dataset based on MADE-WIC (https://doi.org/10.1145/3691620.3695348), a collection of functions and comments mined from open-source projects. Each instance contains source code and its associated comments, together with the following annotations:

    a vulnerability label associated with the source-code component; and
    an SATD label associated with the comment component.

The competition dataset is divided into three partitions:
Partition 	Availability 	Intended use
Training set 	Public 	Train the submitted model
Validation set 	Public 	Measure the performance of the trained model
Test set 	Hidden 	Final evaluation by the competition organisers only

The hidden test set will be used to determine the final winner. It is not disclosed to participants and is withheld to measure how well submissions generalise beyond the public training and validation data.
Requirements

You must train, tune, and evaluate your model on the provided data. We look forward to solutions that outperform our baseline model.

Detailed instructions about the competition (data, rules, baseline, results, etc.) can be found in our GitHub Repository (https://github.com/nlbse2027/code-comment-classification) and a Google Colab Notebook (https://colab.research.google.com/drive/1b_EiXx5woyCrDntwbljfi-BwS9Dwvhoa?usp=sharing).
Competition Organizers

The competition is organized by: Moritz Mock (momock@unibz.it), Thomas Borsani (tborsani@unibz.it), and Barbara Russo (brusso@unibz.it)
Participation Requirements

To participate in the competition, you must train, tune, and evaluate your model using the provided training and validation sets.

Additionally, you must write a paper (2-4 pages) describing:

    The architecture and details of the classification model
    The procedure used to pre-process the data
    The procedure used to tune the classifier on the training set
    The results of your classifier on the validation set
    A link to the code/tool with proper documentation on how to run it and replicate the results

    📌 Submission Link

https://forms.gle/yx2CcKLYYBSjWoiJ8

All submissions must conform to the NLBSE 2027 submission requirements and the ICSE 2027 formatting and submission instructions.
Submission Acceptance

Submissions will be evaluated and accepted based on correctness and reproducibility, defined by the following criteria:

    Clarity and detail of the paper content
    Availability of the code/tool, including the training/tuning/evaluation pipeline, released as open-source
    Correct training/tuning/evaluation of your code/tool on the provided data
    Correct report of the metrics and results
    Clarity of the code documentation

We will use a formula to rank the competition submissions and determine a winner. Details will be provided in the Google Colab notebook.

The accepted submissions will be published in the workshop proceedings.
Citing Relevant Work
General Competition Citation

Please cite the following work if participating in either competition:

Mock, M., Borsani, T., Russo, B., Santos, F., Penney, J., Valenzuela-Toledo, P., Kehrer, T., and Panichella, S. (2027). The NLBSE’27 Tool Competition and Challenge. In Proceedings of the 6th International Workshop on Natural Language-based Software Engineering (NLBSE’27).
Multitask Code Classification Citations

Please also cite the work underlying the dataset and baseline:

Mock, M., Melegati, J., Kretschmann, M., Diaz Ferreyra, N. E., and Russo, B. (2024). MADE-WIC: Multiple Annotated Datasets for Exploring Weaknesses in Code. In Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering (ASE ’24), pp. 2346–2349. Association for Computing Machinery. https://doi.org/10.1145/3691620.3695348

Russo, B., Melegati, J., and Mock, M. (2025). Leveraging Multi-Task Learning to Improve the Detection of SATD and Vulnerability. In 2025 IEEE/ACM 33rd International Conference on Program Comprehension (ICPC), pp. 1–12. https://doi.org/10.1109/ICPC66645.2025.00017

Feng, Z., Guo, D., Tang, D., Duan, N., Feng, X., Gong, M., Shou, L., Qin, B., Liu, T., Jiang, D., and Zhou, M. (2020). CodeBERT: A Pre-Trained Model for Programming and Natural Languages. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 1536–1547. Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.findings-emnlp.139