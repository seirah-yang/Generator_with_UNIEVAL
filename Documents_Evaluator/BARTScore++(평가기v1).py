# ---------------------------
# 0) 준비: BARTScore 스코어러
# ---------------------------
# 가정: bart_score(h, r) -> float
#  - r(참조)을 조건으로 h(가설)의 토큰 평균 로그확률(또는 그 변형)을 리턴
#  - 길이 정규화(토큰 평균) 권장, 배치/캐시 적용 가능

def bart_score(hypothesis: str, reference: str) -> float:
    # TODO: 실제 구현 - HF transformers로 BART를 로드하여
    #       log p(h | ref)의 토큰별 log prob을 평균해 반환
    raise NotImplementedError

# ------------------------------------
# 1) Non-translation(선별) 휴리스틱(선택)
# ------------------------------------
def overlap_ratio(h, r):
    # 아주 단순한 토큰 겹침 비율(예시)
    hs, rs = set(h.split()), set(r.split())
    if not hs or not rs: return 0.0
    return len(hs & rs) / len(hs | rs)

def is_non_translation(h, r, overlap_thr=0.05, lowprob_ratio_thr=0.5):
    # overlap 기반 간단판 (실전: "낮은 생성확률 토큰 비율" 추정 포함 권장)
    return overlap_ratio(h, r) < overlap_thr

# --------------------------------
# 2) Detect-Correct (Refine 단계)
# --------------------------------
def propose_candidates(h: str, r: str, k: int = 5):
    """
    간단 예시:
    - 실제 구현은 '가장 낮은 확률 토큰 위치'를 찾아
      [치환/삽입/삭제] 후보들을 BART 분포에서 top-k로 뽑아 문장 후보 생성
    - 여기서는 예시로, 토큰 하나를 다른 토큰으로 치환하는 단순 후보를 리턴
    """
    toks = h.split()
    if not toks: 
        return [h]
    # toy: 마지막 토큰을 유사 토큰으로 치환하는 가짜 후보들
    base = toks[:-1]
    return [ " ".join(base + [f"{t}_alt"]) for t in toks[-k:] ] + [h]

def refine_sentence(h: str, r: str, T: int = 3, k: int = 5, min_gain: float = 1e-4) -> str:
    """
    Detect: 확률 낮은 토큰 위치 찾기 (여기선 생략)
    Correct: 후보 생성 -> BARTScore 최대 문장 선택
    반복: T회 or 개선 미미하면 중단
    """
    best = h
    best_score = bart_score(best, r)
    for _ in range(T):
        cands = propose_candidates(best, r, k=k)  # 후보세트
        # 후보 중 BARTScore가 최대인 문장 선택
        scored = [(cand, bart_score(cand, r)) for cand in cands]
        cand_best, cand_s = max(scored, key=lambda x: x[1])
        if cand_s - best_score > min_gain:
            best, best_score = cand_best, cand_s
        else:
            break  # 개선 없으면 조기 종료
    return best

# -------------------------------
# 3) Dist 계산 + 가중 합산 (최종)
# -------------------------------
def bartscore_pp(h: str, r: str, w_exp: float = 1.2, w_imp: float = 1.0,
                 do_refine: bool = True, refine_T: int = 3, refine_k: int = 5) -> float:
    """
    BARTScore++ = - ( w_exp*Dist_exp + w_imp*Dist_imp )
    Dist_exp = BARTS(h*, r) - BARTS(h, r)
    Dist_imp = BARTS(r, r) - BARTS(h*, r)
    """
    # 1) 원 점수
    s_hr = bart_score(h, r)        # BARTS(h, r)
    s_rr = bart_score(r, r)        # BARTS(r, r) - 상수로 캐싱해도 됨

    # 2) refine
    h_star = refine_sentence(h, r, T=refine_T, k=refine_k) if do_refine else h
    s_hstar_r = bart_score(h_star, r)

    # 3) 거리 계산
    dist_exp = (s_hstar_r - s_hr)          # 명시 오류 거리
    dist_imp = (s_rr      - s_hstar_r)     # 암시 오류 거리

    # 4) 가중 합산 (거리 → 점수)
    score_pp = - (w_exp*dist_exp + w_imp*dist_imp)
    return float(score_pp), {
        "BARTS(h,r)": s_hr,
        "BARTS(h*,r)": s_hstar_r,
        "BARTS(r,r)": s_rr,
        "Dist_exp": dist_exp,
        "Dist_imp": dist_imp,
        "h*": h_star
    }

# -------------------------------
# 4) 사용 예시
# -------------------------------
# h = "system output sentence ..."
# r = "reference sentence ..."
# score, detail = bartscore_pp(h, r, w_exp=1.2, w_imp=1.0, do_refine=True, refine_T=3, refine_k=5)
# print(score, detail)
