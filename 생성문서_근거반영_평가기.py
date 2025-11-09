# -*- coding: utf-8 -*-
import os
import docx
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ─────────────────────────────────────────────
# 1️⃣ 임베더 캐시
# ─────────────────────────────────────────────
_EMBEDDER_CACHE = {"name": None, "model": None}

def _get_embedder(model_name="intfloat/e5-large"):
    global _EMBEDDER_CACHE
    if _EMBEDDER_CACHE["model"] and _EMBEDDER_CACHE["name"] == model_name:
        return _EMBEDDER_CACHE["model"]
    model = SentenceTransformer(model_name)
    _EMBEDDER_CACHE["name"] = model_name
    _EMBEDDER_CACHE["model"] = model
    return model

# ─────────────────────────────────────────────
# 2️⃣ 파일 로드 함수 (DOCX/PDF 자동 판별)
# ─────────────────────────────────────────────
def load_text_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    else:
        return ""

# ─────────────────────────────────────────────
# 3️⃣ 폴더 내 모든 근거문헌 로드
# ─────────────────────────────────────────────
def load_law_corpus_from_dir(dir_path):
    corpus = []
    for file in os.listdir(dir_path):
        path = os.path.join(dir_path, file)
        if not os.path.isfile(path):
            continue
        if path.endswith((".docx", ".pdf")):
            try:
                text = load_text_from_file(path)
                if text.strip():
                    corpus.append(text)
            except Exception as e:
                print(f"[WARN] {file} 불러오기 실패: {e}")
    return corpus

# ─────────────────────────────────────────────
# 4️⃣ 근거문헌 준수도 평가 함수
# ─────────────────────────────────────────────
def reference_compliance_with_sources(
    section_text,
    law_corpus,
    model_name="intfloat/e5-large",
    threshold=0.8,
    top_k=5
):
    if not section_text or not law_corpus:
        return {"compliance_score": 0.0, "top_references": []}

    model = _get_embedder(model_name)

    emb_section = model.encode([section_text], normalize_embeddings=True)
    emb_law = model.encode(law_corpus, normalize_embeddings=True)

    sims = cosine_similarity(emb_section, emb_law)[0]
    matched_ratio = float((sims >= threshold).sum()) / len(law_corpus)
    top_idx = sims.argsort()[::-1][:top_k]
    top_refs = [(law_corpus[i][:200], float(sims[i])) for i in top_idx]

    return {
        "compliance_score": round(float(matched_ratio), 3),
        "top_references": top_refs
    }

# ─────────────────────────────────────────────
# 5️⃣ 여러 파일 일괄 평가 실행
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # (1) 평가 대상 문서 폴더 경로
    target_dir = "/content/drive/MyDrive/1027"  # ← 평가할 문서 폴더 입력 (.docx / .pdf)
    law_dir = "/content/drive/MyDrive/reference_file"        # ← 근거문헌 폴더 입력

    # (2) 근거문헌 로드
    law_corpus = load_law_corpus_from_dir(law_dir)

    # (3) 평가 대상 폴더 순회
    results = []
    for file in os.listdir(target_dir):
        file_path = os.path.join(target_dir, file)
        if not os.path.isfile(file_path):
            continue
        if file_path.endswith((".docx", ".pdf")):
            print(f"\n📄 평가 중: {file}")
            section_text = load_text_from_file(file_path)
            result = reference_compliance_with_sources(section_text, law_corpus)
            results.append({
                "file": file,
                "compliance_score": result["compliance_score"]
            })
            print(f" → 준수도 점수: {result['compliance_score']:.3f}")

    # (4) 전체 요약 출력
    print("\n✅ 평가 완료 결과 요약:")
    for r in results:
        print(f"{r['file']:50s} | Score: {r['compliance_score']:.3f}")
