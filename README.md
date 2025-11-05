# Generator_with_UNIEVAL
Developed an AI-driven document generation pipeline for R&amp;D and administrative reports using LangGraph. The system automates drafting, validation, and export based on regulation templates, integrating E5 embedding retrieval, SKT A.X-4.0-Light LLM, and NLI-based compliance evaluation to ensure factual and policy alignment.

- **FormSearchNode**: Classifies document type, retrieves regulation templates, maps metadata.  
- **ContextBuilder**: Builds retrieval context using hybrid **BM25 + FAISS** search.  
- **DraftWriter**: Generates document drafts based on prompt roles and retrieved context.  
- **Validator**: Runs rule-based + NLI-based validation for policy compliance and factual accuracy.  
- **Repairer**: Rewrites failed sections to meet compliance threshold.  
- **Exporter**: Outputs validated DOCX/HTML files with evaluation tables.

---

### ⚙️ Core Components
| Component | Description |
|------------|--------------|
| **Embedding Model** | `intfloat/e5-large` (semantic retrieval) |
| **Generation Model** | `SKT A.X-4.0-Light` (LLM generation) |
| **Validation Model** | `XLM-RoBERTa-Large-XNLI` (entailment-based compliance) |
| **Retrieval** | Hybrid BM25 + FAISS |
| **Output Format** | DOCX / HTML (via `python-docx` + `Jinja2`) |

---

### 🧮 Evaluation Metrics (UNIEVAL-based)
| Metric | Description |
|--------|--------------|
| **Accuracy** | Semantic and factual correctness |
| **Relevance** | Alignment with retrieved reference context |
| **Coherence** | Logical flow between document sections |
| **Fluency** | Grammatical and stylistic quality |
| **Consistency** | Internal logical alignment |
| **Redundancy (↓)** | Repetition rate across sentences |
| **Final Score** | Weighted aggregate (0–1 scale) |

---

### 📊 Compliance Evaluation
| Metric | Meaning | Standard |
|--------|----------|----------|
| **Compliance Score** | Regulatory alignment ratio | Core compliance indicator |
| **Similarity (sim_n)** | Semantic distance to each reference | ≥ 0.8 → Semantically aligned |
| **Top References (k)** | Top-k supporting regulation texts | Reviewer-verifiable |
| **Threshold = 0.8** | Compliance judgment baseline | Semantic Paraphrase level |
| **Score Interpretation** | 0–1 scaled | 0.8↑ Excellent / 0.6–0.8 Fair / 0.6↓ Poor |

---

### 💡 Key Features
- **Automated multi-stage generation** (context → draft → validate → repair → export)
- **Reference-aware compliance checking** using semantic entailment
- **Hybrid retrieval** combining keyword and vector-based search
- **Evaluation dashboard** with UNIEVAL-style compliance reports
- **Format-compliant output** for administrative and national R&D documentation

---

### 🧠 Future Work
- Integration with **LangChain/Graph** for multi-agent workflow
- Expansion to **multi-language regulation corpora**
- Enhancement of **semantic redundancy control** and **evaluation visualization**

---

### 👩‍💻 Author
**Sora Yang (양소라)**  
RN, MSN |
TA | CRC | DM | 
AI Developer Trainee @ Alpaco Bootcamp  
Specialized in Clinical Data, R&D Automation, and LLM-driven Compliance Systems.

---

### 📎 References
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* ACL.  
- Wang et al. (2023). *UniEval: A Unified Benchmark for Comprehensive Evaluation of Text Generation.* ACL.  
- Cer et al. (2018). *Universal Sentence Encoder.* arXiv:1803.11175.  
- Goyal et al. (2022). *Evaluating Factual Consistency in Texts via QA-based Metrics.* ACL Findings.  
- Lin (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop.

---

📁 **Project Type:** AI + NLP + Document Automation  
🧱 **Tech Stack:** Python, LangGraph, FAISS, BM25, NLI, DOCX/Jinja2  
🎯 **Goal:** Reduce document creation time by while ensuring format and policy compliance.
