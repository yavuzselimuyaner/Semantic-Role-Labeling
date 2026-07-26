# Semantic Role Labeling: Sequence Labeling and Multi-class Classification

This repository contains the implementation and comparative evaluation of a Semantic Role Labeling (SRL) system. The project aims to automatically identify semantic arguments for predicates in natural language sentences, effectively answering "who did what to whom, where, and when".

## Overview

Semantic Role Labeling provides machines with a structured understanding of sentence meaning beyond surface-level pattern matching, serving as a foundational component for tasks like open-domain question answering and knowledge graph construction. This project compares three distinct NLP paradigms under identical experimental conditions:
* A rule-based baseline utilizing dependency parse heuristics.
* A statistical model employing Conditional Random Fields (CRF) with hand-crafted features.
* A deep learning approach based on BERT fine-tuning for sequence labeling.

Experiments are conducted in a controlled cross-lingual setting using English and Spanish datasets, keeping the role inventory fixed (PropBank scheme).

---

## Datasets

The models are trained and evaluated on two PropBank-style datasets:
* **English (PropBank-EWT)**: Consists of 50,182 BIO-encoded instances with 110 distinct labels.
* **Spanish (PropBank-ES)**: Universal Propositions on the AnCora treebank, consisting of 38,839 instances with 50 distinct labels.

---

## Methodology

### Label Representation
All three models share a common BIO span encoding scheme. The predicate token receives a `B-V` tag, while argument spans are prefixed with `B-ROLE` and `I-ROLE` (e.g., `B-ARG0`, `I-ARG0`).

### 1. Rule-Based Baseline
* Parses sentences using spaCy's English (`en_core_web_sm`) or Spanish (`es_core_news_sm`) dependency models.
* Maps Universal Dependency labels directly to PropBank roles.

### 2. Conditional Random Field (CRF)
* Uses a linear-chain CRF optimized via L-BFGS.
* Incorporates surface features, capitalization, part-of-speech tags, positional features, and available syntactic features.

### 3. BERT Fine-Tuning
* Wraps the target predicate with reserved vocabulary slots (`[unused0]` and `[unused1]`) for sub-word alignment.
* Fine-tunes `bert-base-cased` for English and `bert-base-multilingual-cased` for Spanish with a linear token classification head.

---

## Results and Evaluation

Performance is measured using span-level Precision, Recall, and F1-Score following the CoNLL-2005 evaluation protocol, excluding predicate tags (`B-V`) from the scoring.

| Model | Dataset | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Rule-Based** | PropBank-EWT (English) | 0.394 | 0.299 | 0.340 |
| **CRF** | PropBank-EWT (English) | 0.650 | 0.470 | 0.545 |
| **BERT** | PropBank-EWT (English) | 0.697 | 0.776 | 0.734 |
| **Rule-Based** | PropBank-ES (Spanish) | 0.178 | 0.121 | 0.144 |
| **CRF** | PropBank-ES (Spanish) | 0.425 | 0.226 | 0.295 |
| **BERT** | PropBank-ES (Spanish) | 0.438 | 0.484 | 0.459 |

**Key Findings:**
* The monotonic performance ordering of Rule-based < CRF < BERT is preserved across both languages.
* A consistent 20-27 point cross-lingual F1 gap exists between English and Spanish.
* This gap is attributed to Spanish's richer inflectional morphology, freer constituent order, and the relatively weak pre-training signal for Spanish in multilingual BERT.

---

## Technologies & Environment

* **Language**: Python 3.10
* **Libraries**: spaCy 3.7, sklearn-crfsuite 0.5, Hugging Face Transformers 4.38, PyTorch 2.x, NLTK 3.9
* **Hardware**: Google Colab (Tesla T4 GPU for BERT; 2-vCPU/12 GB RAM CPU for baselines)

---

## Author

**Yavuz Selim Uyaner**
*Department of Computer Engineering, Ankara University*
*Contact: yavuzselimuyaner@gmail.com*# Semantic Role Labeling: Sequence Labeling and Multi-class Classification

This repository contains the implementation and comparative evaluation of a Semantic Role Labeling (SRL) system. The project aims to automatically identify semantic arguments for predicates in natural language sentences, effectively answering "who did what to whom, where, and when".

## Overview

Semantic Role Labeling provides machines with a structured understanding of sentence meaning beyond surface-level pattern matching, serving as a foundational component for tasks like open-domain question answering and knowledge graph construction. This project compares three distinct NLP paradigms under identical experimental conditions:
* A rule-based baseline utilizing dependency parse heuristics.
* A statistical model employing Conditional Random Fields (CRF) with hand-crafted features.
* A deep learning approach based on BERT fine-tuning for sequence labeling.

Experiments are conducted in a controlled cross-lingual setting using English and Spanish datasets, keeping the role inventory fixed (PropBank scheme).

---

## Datasets

The models are trained and evaluated on two PropBank-style datasets:
* **English (PropBank-EWT)**: Consists of 50,182 BIO-encoded instances with 110 distinct labels.
* **Spanish (PropBank-ES)**: Universal Propositions on the AnCora treebank, consisting of 38,839 instances with 50 distinct labels.

---

## Methodology

### Label Representation
All three models share a common BIO span encoding scheme. The predicate token receives a `B-V` tag, while argument spans are prefixed with `B-ROLE` and `I-ROLE` (e.g., `B-ARG0`, `I-ARG0`).

### 1. Rule-Based Baseline
* Parses sentences using spaCy's English (`en_core_web_sm`) or Spanish (`es_core_news_sm`) dependency models.
* Maps Universal Dependency labels directly to PropBank roles.

### 2. Conditional Random Field (CRF)
* Uses a linear-chain CRF optimized via L-BFGS.
* Incorporates surface features, capitalization, part-of-speech tags, positional features, and available syntactic features.

### 3. BERT Fine-Tuning
* Wraps the target predicate with reserved vocabulary slots (`[unused0]` and `[unused1]`) for sub-word alignment.
* Fine-tunes `bert-base-cased` for English and `bert-base-multilingual-cased` for Spanish with a linear token classification head.

---

## Results and Evaluation

Performance is measured using span-level Precision, Recall, and F1-Score following the CoNLL-2005 evaluation protocol, excluding predicate tags (`B-V`) from the scoring.

| Model | Dataset | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Rule-Based** | PropBank-EWT (English) | 0.394 | 0.299 | 0.340 |
| **CRF** | PropBank-EWT (English) | 0.650 | 0.470 | 0.545 |
| **BERT** | PropBank-EWT (English) | 0.697 | 0.776 | 0.734 |
| **Rule-Based** | PropBank-ES (Spanish) | 0.178 | 0.121 | 0.144 |
| **CRF** | PropBank-ES (Spanish) | 0.425 | 0.226 | 0.295 |
| **BERT** | PropBank-ES (Spanish) | 0.438 | 0.484 | 0.459 |

**Key Findings:**
* The monotonic performance ordering of Rule-based < CRF < BERT is preserved across both languages.
* A consistent 20-27 point cross-lingual F1 gap exists between English and Spanish.
* This gap is attributed to Spanish's richer inflectional morphology, freer constituent order, and the relatively weak pre-training signal for Spanish in multilingual BERT.

---

## Technologies & Environment

* **Language**: Python 3.10
* **Libraries**: spaCy 3.7, sklearn-crfsuite 0.5, Hugging Face Transformers 4.38, PyTorch 2.x, NLTK 3.9
* **Hardware**: Google Colab (Tesla T4 GPU for BERT; 2-vCPU/12 GB RAM CPU for baselines)

---

## Author

**Yavuz Selim Uyaner**
*Department of Computer Engineering, Ankara University*
*Contact: yavuzselimuyaner@gmail.com*