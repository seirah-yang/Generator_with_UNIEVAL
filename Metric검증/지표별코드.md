# 1. Fluency (유창성)
def _fluency(sents: List[str]) -> float:
    if not sents: return 0.5
    lens = [len(s) for s in sents]
    mean_len = sum(lens)/len(lens)
    punct = sum(ch in ".,;:?!~" for s in sents for ch in s) / (sum(lens)+1e-6)
    score = 0.5 + 0.5 * math.tanh((mean_len-25)/50) - 0.2*abs(punct-0.03)
    return max(0.0, min(1.0, score))
  #문장 길이 평균과 문장 내 구두점 비율을 활용해서 가독성/문법적 유창성을 평가.
	# 문장 길이가 너무 짧거나, 구두점 비율이 이상할 경우 점수 감소.

# 2. Coherence (응집성)
def simple_coherence(sentences: List[str]) -> float:
    if len(sentences) < 2: return 0.5
    scores = []
    for i in range(len(sentences)-1):
        scores.append(keyword_overlap(sentences[i], sentences[i+1]))
    return sum(scores)/len(scores)
  # 인접한 문장 간 키워드 겹침 비율(Jaccard 유사도 기반)로 계산.
	# 문장 흐름이 잘 이어지는지 확인하는 지표.

# 3. Redundancy (중복성)
def ngram_redundancy(sentences: List[str], n: int = 3) -> float:
    grams = []
    for s in sentences:
        toks = re.findall(r"[가-힣A-Za-z0-9]+", (s or "").lower())
        grams += list(zip(*[toks[i:] for i in range(n)]))
    if not grams: return 0.0
    c = Counter(grams)
    dup = sum(v-1 for v in c.values() if v>1)
    return dup / (len(grams) + 1e-6)
  # 문장 안에서 n-gram(예: 3개 단어씩 묶음) 반복 비율을 측정.
	# 값이 높을수록 중복된 표현이 많음 → 바람직하지 않음.

# 4. Accuracy (정확성)
tot = max(1, entail+contra+unknown)
accuracy = entail / tot
  # 섹션 내에서 추출한 주장(claim) 들을 근거 문장과 NLI/QA로 대조해 판정.
	# entail(지지) 비율을 정확성 점수로 환산. 즉, 주장 대비 얼마나 근거로 뒷받침되는지를 본 것.


