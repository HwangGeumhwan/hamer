"""
한국어 이름 복원 후처리 모듈

자모(字母) 시퀀스를 입력받아 완성된 한국어 음절 문자열을 반환합니다.

규칙:
  - 각 음절은 완성된 글자 (초성 + 중성 [+ 종성])
  - 겹받침 제외 — 단자음 받침만 허용
  - 쌍자음: 동일 단자음 두 번 연속 입력 (ㄱ+ㄱ→ㄲ, ㄷ+ㄷ→ㄸ, ㅂ+ㅂ→ㅃ, ㅅ+ㅅ→ㅆ, ㅈ+ㅈ→ㅉ)
  - 이중모음: 단모음 조합 가능 (ㅗ+ㅏ→ㅘ, ㅗ+ㅐ→ㅙ, ㅜ+ㅓ→ㅝ, ㅜ+ㅔ→ㅞ 등)
  - ACTIONS에 이미 포함된 ㅢ·ㅚ·ㅟ는 단일 입력 또는 조합(㡈+ㅣ) 모두 허용
"""

# ── 초성 (19개) ────────────────────────────────────────────────────────────────
ONSET_LIST = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
ONSET_IDX  = {c: i for i, c in enumerate(ONSET_LIST)}

# ── 중성 (21개) ────────────────────────────────────────────────────────────────
NUCLEUS_LIST = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
NUCLEUS_IDX  = {v: i for i, v in enumerate(NUCLEUS_LIST)}

# ── 종성 (28개, 0 = 받침 없음) ───────────────────────────────────────────────
CODA_LIST = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ',
             'ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
CODA_IDX   = {c: i for i, c in enumerate(CODA_LIST)}

# ── 쌍자음 조합 ────────────────────────────────────────────────────────────────
DOUBLE_CONSONANT: dict[tuple[str,str], str] = {
    ('ㄱ','ㄱ'): 'ㄲ',
    ('ㄷ','ㄷ'): 'ㄸ',
    ('ㅂ','ㅂ'): 'ㅃ',
    ('ㅅ','ㅅ'): 'ㅆ',
    ('ㅈ','ㅈ'): 'ㅉ',
}

# ── 이중모음 조합 (단모음 두 개 → 이중모음) ──────────────────────────────────
COMPOUND_VOWEL: dict[tuple[str,str], str] = {
    ('ㅗ','ㅏ'): 'ㅘ',
    ('ㅗ','ㅐ'): 'ㅙ',
    ('ㅗ','ㅣ'): 'ㅚ',
    ('ㅜ','ㅓ'): 'ㅝ',
    ('ㅜ','ㅔ'): 'ㅞ',
    ('ㅜ','ㅣ'): 'ㅟ',
    ('ㅡ','ㅣ'): 'ㅢ',
}

# ── 역방향 테이블 ──────────────────────────────────────────────────────────────
# ACTIONS에 이미 있는 이중모음(ㅢ·ㅚ·ㅟ)은 조합 분해 불필요
_ACTIONS_VOWELS = {'ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ',
                   'ㅐ','ㅒ','ㅔ','ㅖ','ㅢ','ㅚ','ㅟ'}
# ACTIONS에 없는 이중모음만 역방향 분해
COMPOUND_TO_PAIR: dict[str, tuple[str,str]] = {
    v: k for k, v in COMPOUND_VOWEL.items() if v not in _ACTIONS_VOWELS
}
DOUBLE_TO_PAIR: dict[str, tuple[str,str]] = {v: k for k, v in DOUBLE_CONSONANT.items()}


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _make_syllable(onset: str, nucleus: str, coda: str = '') -> str:
    o = ONSET_IDX.get(onset, ONSET_IDX['ㅇ'])
    n = NUCLEUS_IDX[nucleus]
    k = CODA_IDX.get(coda, 0)
    return chr(0xAC00 + o * 21 * 28 + n * 28 + k)


# ── 공개 API ──────────────────────────────────────────────────────────────────

def parse_jamo_to_korean(jamo_seq: list[str]) -> str:
    """자모 리스트 → 완성된 한국어 문자열.

    Args:
        jamo_seq: 자모 리스트, 예: ['ㄱ', 'ㅣ', 'ㅁ', 'ㅅ', 'ㅓ', 'ㅇ']

    Returns:
        완성된 문자열, 예: '김성'

    상태 기계 동작 요약:
        자음 입력:
            · 받침 후보 있음 + 쌍자음 가능 → 현 음절 받침 없이 확정, 쌍자음을 새 초성으로
            · 받침 후보 있음 + 쌍자음 불가 → 받침 후보 확정, 새 자음을 초성으로
            · 중성만 있음                   → 자음을 받침 후보로 저장
            · 초성만 있음 + 쌍자음 가능    → 초성을 쌍자음으로 갱신
            · 아무것도 없음                 → 새 초성
        모음 입력:
            · 받침 후보 있음 → 후보를 다음 음절 초성으로, 현 음절 받침 없이 확정
            · 중성 있음 + 이중모음 가능    → 중성을 이중모음으로 갱신
            · 중성 있음 + 이중모음 불가    → 현 음절 확정, 새 음절 시작
            · 초성(또는 없음)만 있음       → 중성 설정
    """
    result: list[str] = []
    onset:     str | None = None
    nucleus:   str | None = None
    coda_cand: str | None = None

    def flush(coda: str = '') -> None:
        nonlocal onset, nucleus, coda_cand
        if nucleus is not None:
            result.append(_make_syllable(onset or 'ㅇ', nucleus, coda))
        elif onset is not None:
            result.append(onset)
        onset = nucleus = coda_cand = None

    for jamo in jamo_seq:
        if jamo in NUCLEUS_IDX:                        # ── 모음 ──
            if coda_cand is not None:
                _carry = coda_cand      # flush()가 coda_cand=None으로 초기화하기 전에 저장
                flush()
                onset   = _carry
                nucleus = jamo
            elif nucleus is not None:
                compound = COMPOUND_VOWEL.get((nucleus, jamo))
                if compound:
                    nucleus = compound
                else:
                    flush()
                    nucleus = jamo
            else:
                nucleus = jamo

        elif jamo in ONSET_IDX:                        # ── 자음 ──
            if coda_cand is not None:
                double = DOUBLE_CONSONANT.get((coda_cand, jamo))
                if double:
                    flush()
                    onset = double
                else:
                    flush(coda_cand)
                    onset = jamo
            elif nucleus is not None:
                coda_cand = jamo
            elif onset is not None:
                double = DOUBLE_CONSONANT.get((onset, jamo))
                if double:
                    onset = double
                else:
                    flush()
                    onset = jamo
            else:
                onset = jamo

        else:                                          # ── 기타 문자 ──
            flush()
            result.append(jamo)

    if coda_cand is not None:
        flush(coda_cand)
    else:
        flush()

    return ''.join(result)


def decompose_korean_to_jamo(text: str) -> list[str]:
    """완성된 한국어 문자열 → 자모 리스트 (parse_jamo_to_korean 의 역함수).

    ACTIONS에 없는 쌍자음은 단자음 두 개로,
    ACTIONS에 없는 이중모음은 단모음 두 개로 분해합니다.

    Args:
        text: 한국어 문자열, 예: '김성'

    Returns:
        자모 리스트, 예: ['ㄱ', 'ㅣ', 'ㅁ', 'ㅅ', 'ㅓ', 'ㅇ']
    """
    result: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            offset    = code - 0xAC00
            onset_ch  = ONSET_LIST[offset // (21 * 28)]
            nucleus_ch = NUCLEUS_LIST[(offset % (21 * 28)) // 28]
            coda_ch   = CODA_LIST[offset % 28]

            # 초성: 쌍자음이면 단자음 두 개로
            if onset_ch in DOUBLE_TO_PAIR:
                result.extend(DOUBLE_TO_PAIR[onset_ch])
            else:
                result.append(onset_ch)

            # 중성: ACTIONS에 없는 이중모음이면 단모음 두 개로
            if nucleus_ch in COMPOUND_TO_PAIR:
                result.extend(COMPOUND_TO_PAIR[nucleus_ch])
            else:
                result.append(nucleus_ch)

            # 종성 (겹받침은 없으므로 단자음만 처리)
            if coda_ch:
                if coda_ch in DOUBLE_TO_PAIR:
                    result.extend(DOUBLE_TO_PAIR[coda_ch])
                else:
                    result.append(coda_ch)
        else:
            result.append(ch)

    return result
