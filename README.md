# Generator_with_UNIEVAL
Developed an advanced AI-driven document generation pipeline capable of automatically drafting, validating, and formatting R&D project plans, administrative reports, and internal documents.
The system integrates a LangGraph-based workflow, E5 embedding retrieval, SKT A.X-4.0-Light LLM, and an XNLI-based compliance validator to ensure both factual correctness and policy alignment.

⸻

## 1. Problem Statement (Practical Pain Points)

Drafting administrative documents and national R&D project plans comes with the following challenges:

  •	Complex rules and templates with high risk of omission
	
  •	Significant quality variance across authors
	
  •	Time-consuming drafting process (often 1–2 days)
	
  •	Manual verification of regulation compliance required
	
  •	Long-document structure alignment is difficult for humans and AI

To solve this, the project aims to deliver:

“Automated drafting → automated compliance validation → automated formatting.”

⸻

## 2. System Overview

### 1) LangGraph-Based End-to-End Automation Pipeline
```python
(Start)
  ↓
FormSearchNode → ContextBuilder → DraftWriter → Validator → Repairer → Exporter
```

### 2) Node Descriptions
```python
| Node            | Role                                                                 |
|-----------------|----------------------------------------------------------------------|
| FormSearchNode  | Classifies document type, retrieves relevant regulations/templates, maps metadata |
| ContextBuilder  | Builds contextual evidence using Hybrid BM25 + FAISS retrieval       |
| DraftWriter     | Generates structure-aware drafts using the SKT A.X-4.0-Light LLM     |
| Validator       | Conducts rule-based + NLI-based compliance and factuality validation |
| Repairer        | Automatically rewrites non-compliant or incoherent sections          |
| Exporter        | Outputs validated DOCX/HTML documents using Jinja2 & python-docx     |
```

⸻

## 3. Core Components
```python
| Component              | Description                                          |
|------------------------|------------------------------------------------------|
| Embedding Model        | intfloat/e5-large for high-precision semantic retrieval |
| Retriever              | Hybrid BM25 + FAISS vector search                    |
| Generation Model       | SKT A.X-4.0-Light local LLM                          |
| Validation Model       | XLM-RoBERTa-Large-XNLI for semantic entailment       |
| Exporter               | Automated DOCX / HTML generator                      |
| Evaluation Metrics     | UNIEVAL-style scoring framework                      |
```

⸻

## 4. UNIEVAL-style Evaluation Metrics
```python
|   Metric         |   Description                                     |
|------------------|---------------------------------------------------|
| Accuracy         | Ratio of claims supported by internal evidence    |
| Relevance        | Semantic alignment with retrieved references      |
| Coherence        | Logical flow between sentences and sections       |
| Fluency          | Grammatical and stylistic clarity                 |
| Consistency      | Internal logical and numerical consistency        |
| Redundancy (↓)   | Repetitive content ratio                          |
| Final Score      | Weighted aggregation on a 0–1 scale               |
```
### 1) Accuracy Formula
```md
$$
Accuracy = \frac{entail}{entail + contra + unknown}
$$
```
  •	entail: Supported claims
	 
  •	contra: Contradicted claims
	 
  •	unknown: Indeterminate claims
	 
  •	tot: Total number of extracted claims

    ➡ Higher accuracy implies stronger internal factual consistency.

⸻

## 5. Compliance Evaluation (Policy Alignment)

```python
|   Metric           |   Meaning                                    |
|--------------------|----------------------------------------------|
| Compliance Score   | Regulatory alignment ratio                   |
| sim_n              | Semantic similarity ≥ 0.8 for alignment      |
| Top-k References   | Key supporting regulation excerpts           |
| Threshold (0.8)    | Baseline for “Excellent” alignment           |
```

⸻

## 6. Key Features

### 1) Fully Automated Multi-Stage Pipeline

Draft → Validate → Repair → Export (all automated)

### 2) Regulation-Aware Document Generation

Structured based on:
  •	Administrative Regulations (Presidential Decree / Enforcement Rules)
	
  •	National R&D Project Plan Template (Annex Form #2)
	
  •	R&D Strategic Plan Guidelines

### 3) Self-Validation with NLI

Automatic detection of inconsistencies, hallucinations, and policy violations.

### 4) Format Compliance Assurance

Detects missing mandatory fields (e.g., objectives, significance, expected outcomes).

### 5) Evaluation Dashboard

Exports a UNIEVAL-style compliance table alongside the final document.

⸻

## 7. Development Journey – Problem → Ideation → Trial & Error → Solution → Impact

### 1) Problem
  •	Slow drafting process

  •	Complex regulations

  •	High inconsistency among authors

  •	Long document generation instability

### 2) Ideation (Technical Challenges)
  •	Beyond auto-completion: requires rule-compliance + quality control

  •	Long-context alignment essential

  •	Redundancy suppression required

### 3) Implementation Efforts
  •	Analyzed administrative regulations & R&D templates

  •	Structured Annex Form #2 (National R&D Plan Format)

  •	Compared E5-Large vs. CDE-v2

  •	Built BM25 + FAISS hybrid retriever

  •	Designed quality metrics: Accuracy, Fluency, Coherence, Redundancy

### 4) Trial & Error
  •	Redundancy in long generation → improved chunking strategy

  •	Missing mandatory fields → Format Validator introduced

  •	Hallucinations → stronger retrieval tuning

### 5) Solution
  •	Completed LangGraph-based conditional workflow

  •	Integrated Local LLM + Hybrid Retrieval

  •	Added NLI-based validator for stable quality control

### 6) Impact
  •	reduction in document creation time
 
  • increased rule/template compliance through auto-validation

  •	Quality standardization → improved review success rate

  •	Serves as an internal “AI Document Assistant” for admin & R&D teams

⸻

## 8. Architecture Visualization

### Layered Architecture
  •	Data Layer: Regulation texts, templates, R&D guidelines
	
  •	Retrieval Layer: BM25 + FAISS hybrid search
	
  •	Generation Layer: E5-Large, CDE-v2, SKT AX 4.0 Light
	
  •	Evaluation Layer: UNIEVAL-based scoring + NLI validator
	
  •	Export Layer: Automated document construction (DOCX/HTML)

⸻

## 9. Future Work

### Technical Enhancements
  •	Multimodal support (tables, images, scanned PDFs)
	
  •	Domain-specific fine-tuning for administrative & R&D LLMs
	
  •	Format Validator 2.0 with stricter structure checking

### User Experience Improvements
  •	Draft → Review → Final workflow UI
	
  •	Real-time Co-pilot mode for live violation checking
	
  •	Template auto-recommendation

### Quality Management Upgrades
  •	Improved redundancy detection
	
  •	Self-feedback LLM processing loop

### Organizational Integration
  •	Connect with internal rulebooks, manuals, and past submissions
	
  •	Department-specific document style optimization
	
  •	On-premises deployment for sensitive materials

### Long-Term Vision

  • Toward a “Document Co-Pilot Platform” that automates:
	
  • Drafting → Review → Collaboration → Approval → Archiving

⸻

## 👩‍💻 Author

#### Sora Yang
    RN, MSN | TA | CRC | DM | AI Developer Trainee @ Alpaco
    Specialized in Clinical Data, R&D Document Automation, and LLM-driven Compliance Systems

⸻

📎 References
	
• Reimers & Gurevych (2019). Sentence-BERT. ACL.
	
• Wang et al. (2023). UniEval. ACL.
	
• Cer et al. (2018). Universal Sentence Encoder.
	
• Goyal et al. (2022). Factual consistency via QA-based Metrics.
	
• Lin (2004). ROUGE. ACL Workshop.
