# 생성문서_근거반영_평가기 & (top_references, sim 값) 의미(공식 문서 평가 기준)

**평가점수(compliance_score)** 와 각 **평가지표(top_references, sim 값)** 의미(**공식 문서 평가 기준)** 정리

: “R&D 행정문서가 근거 문헌(법령, 규정, 가이드라인)을 얼마나 의미적으로 따르고 있는가” 수치화

---

## **1. 평가지표 체계 요약**

| **지표명** | **산출 방식** | **범위** | **의미** |
| --- | --- | --- | --- |
| **compliance_score** | matched_ratio = (sims ≥ threshold) / total | 0~1 | 문서 내용이 근거문헌과 **의미적으로 일치하는 비율** |
| **top_reference_n** | 유사도 상위 N번째 근거문헌 텍스트 | 문자열 | 해당 문서와 **가장 의미적으로 가까운 근거문헌 문장/조항** |
| **sim_n** | cosine_similarity(section, law_corpus[n]) | 0~1 | 각 근거문헌과의 **의미적 유사도 점수** |
| **threshold** | 0.8 (기본값) | — | “의미적 일치(paraphrase-level)”로 간주하는 기준값 |
| **top_k** | 5 (기본값) | — | 상위 5개 근거문헌만 요약 출력 |

---

## **2. 평가지표 해석 기준 (의미적 준수도)**

| **구간** | **등급** | **해석** | **문서 특성** |
| --- | --- | --- | --- |
| **0.90 ~ 1.00** | ★★★★★ (Excellent) | 근거문헌의 조항·내용을 그대로 따름 | 명시적 인용 수준 — 법령조항을 직접 언급하거나 동일한 문체로 서술 |
| **0.80 ~ 0.89** | ★★★★☆ 
(Strong) | 근거문헌의 의미를 충실히 반영 | 준수도 높음 — “관련 법령에 따라 수행함” 등 간접 인용 중심 |
| **0.60 ~ 0.79** | ★★★☆☆ (Moderate) | 일부 근거문헌과 부분 일치 | 관련 조항은 따르나 표현 다양성 높음, 약한 준수 경향 |
| **0.40 ~ 0.59** | ★★☆☆☆ 
(Weak) | 근거문헌 반영이 부족하거나 일부만 반영 | 지침 언급 부족, 
서술은 있으나 법령 근거 불명확 |
| **< 0.40** | ★☆☆☆☆ 
(Poor) | 근거문헌과 거의 무관 | 규정과 무관한 내용 중심, 준수도 낮음 |

---

## **3. 평가지표 산식**

### **(1) 의미유사도 계산**

각 문장 간 의미적 유사도는 **코사인 유사도(Cosine Similarity)** 로 계산됩니다.

$\text{cosine\_similarity}(A,B) = \frac{A \cdot B}{||A|| \times ||B||}$

→ 1에 가까울수록 문장의 **의미적 동일성(paraphrase)** 이 높음.

### **(2) 근거문헌 준수율**

$\text{Compliance Score} = \frac{\text{유사도 ≥ 0.8 인 문헌 수}}{\text{전체 문헌 수}}$

→ 전체 근거문헌 중 의미적으로 일치하는 문헌의 비율(즉, “법령 내용과 얼마나 정합적인가”)

---

## **4. 평가 예시**

| **문서** | **compliance_score** | **해석** | **상위 근거문헌** |
| --- | --- | --- | --- |
| e5l_1017.docx | **0.82** | 법령·지침 내용 충실히 반영 (Strong) | 산업기술보호법 §2 (0.91), 국가R&D 시행규칙 §14 (0.87) |
| 2023_RND계획서.pdf | **0.78** | 부분 준수 (Moderate) | R&D 관리규정 §5 (0.84), 혁신법 시행규칙 §10 (0.81) |
| 기술개발계획_샘플.docx | **0.54** | 근거문헌 반영 미흡 (Weak) | 사회적가치 창출계획 조항 (0.69) |

---

## **5. 지표별 의미 (Interpretation Guide)**

| **지표** | **정의** | **행정/R&D 문서에서의 의미** |
| --- | --- | --- |
| **compliance_score** | 근거문헌 준수율 | 문서가 관련 규정·법령의 의도를 얼마나 반영했는지 |
| **sim_n (유사도)** | 근거문헌과의 의미 일치 정도 | 문장 구조는 달라도 내용적 근거의 유사성을 수치화 |
| **top_reference_n** | 가장 유사한 근거문헌 텍스트 | 어떤 조항/규정이 문서의 논리적 근거로 작용했는가 |
| **threshold (기준값)** | 유사도로 의미적 일치를 구분하는 기준 | 0.8은 실무적으로 “의미적으로 같다”고 보는 최소 기준 |
| **top_k (근거 개수)** | 사람이 검증할 수 있도록 근거문헌 상위 k개만 표시 | 설명 가능성(Explainability) 확보 목적 |

---

## **6. 종합 평가 모델 구조**

```
문서 텍스트(section_text)
       ↓
SentenceTransformer 임베딩
       ↓
Cosine Similarity 계산
       ↓
유사도 ≥ 0.8 → 근거문헌 일치로 판단
       ↓
전체 근거문헌 중 일치 비율 → Compliance Score
       ↓
Top-5 근거문헌 및 유사도 출력
```

---

## **7. 실제 적용 시 판단 가이드**

| **상황** | **권장 조치** |
| --- | --- |
| **점수 ≥ 0.8 (Excellent/Strong)** | 규정 근거 충실 — 인용 표기만 보완 |
| **0.6~0.8 (Moderate)** | 일부 조항 누락 가능 — 문헌 근거 문장 추가 권장 |
| **0.4~0.6 (Weak)** | 법령 근거 불충분 — “○○법 §X에 의거하여” 형태로 보완 필요 |
| **< 0.4 (Poor)** | 문서 내용이 규정과 무관 — 재작성 또는 RFP 요구사항 재검토 필요 |

---

## **8. 결론 요약**

| **평가요소** | **의미** | **비고** |
| --- | --- | --- |
| **Compliance Score** | 근거문헌 반영률 | 문서 전체 준수도 핵심 지표 |
| **Similarity (sim_n)** | 각 근거문헌과의 의미적 거리 | 0.8 이상이면 근거 일치로 간주 |
| **Top References** | 근거문헌 상위 k개 | Reviewer가 직접 검증 가능 |
| **Threshold = 0.8** | 준수 판단 기준선 | Semantic Paraphrase 기준 |
| **Score 구간 해석** | 0~1 사이 점수로 규정 반영 정도 시각화 | 0.8↑ 우수, 0.6~0.8 보통, 0.6↓ 미흡 |

---

- **compliance_score는 “문서가 얼마나 법령의 의도와 의미적으로 일치하느냐”를 정량화한 핵심 평가점수**
- **sim_n**과 **top_reference_n**은 그 점수의 설명 근거(근거문헌과의 실제 매칭 관계)입니다.

---

** Reference List (APA 7th Edition)**

1. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), 3982–3992. Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1410
2. Goyal, T., Durrett, G., & Li, J. (2022). Evaluating factual consistency in text via question answering. Findings of the Association for Computational Linguistics: ACL 2022, 300–314. https://doi.org/10.18653/v1/2022.findings-acl.26
3. Cer, D., Yang, Y., Kong, S.-Y., Hua, N., Limtiaco, N., John, R. S., … & Kurzweil, R. (2018). Universal Sentence Encoder. arXiv preprint arXiv:1803.11175. https://arxiv.org/abs/1803.11175
4. Wang, A., Liang, P., & Zhou, D. (2023). UniEval: A unified benchmark for comprehensive evaluation of text generation. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL). https://doi.org/10.18653/v1/2023.acl-long.522
5. Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. Text Summarization Branches Out: Proceedings of the ACL-04 Workshop, 74–81. Association for Computational Linguistics. https://aclanthology.org/W04-1013

** 참고문헌별 적용 요약**

| **평가요소** | **주요 이론적 근거** | **참고문헌 번호** |
| --- | --- | --- |
| Compliance Score | 의미적 일관성과 규정 일치도 평가를 위한 QA/Entailment 기반 점수화 | (2), (4) |
| Similarity (sim_n) | 문장 임베딩 기반 의미 유사도 (Cosine Similarity ≥ 0.8) | (1), (3) |
| Top References | Reviewer가 근거문헌 직접 검토 가능하도록 상위 k개 문서 제시 | (4) |
| Threshold = 0.8 | 의미적 Paraphrase 기준선 (Sentence-level semantic alignment) | (1), (3) |
| Score 구간 해석 | 품질 등급화를 위한 정규화 및 요약 일치도 기준 (ROUGE 등) | (5) |