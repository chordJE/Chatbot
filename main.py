import os
import streamlit as st
import anthropic
from dotenv import load_dotenv
from rag import build_index, query_context, is_index_ready

# 로컬: .env 로드 / Streamlit Cloud: st.secrets 사용
load_dotenv()

def get_secret(key: str, default: str = "") -> str:
    """st.secrets(Streamlit Cloud) → os.environ(.env) → default 순으로 조회."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)

# ────────────────────────────────────────────────────────────────────────────
# 박정은 프로필 (시스템 프롬프트에 항상 포함)
# ────────────────────────────────────────────────────────────────────────────
PROFILE = """
당신은 머신러닝, 딥러닝, LLM에 대해 **비유와 쉬운 설명**으로 도와주는 AI/ML 학습 도우미입니다.

[역할]
- AI/ML의 핵심 개념을 일상적인 비유로 쉽게 설명
- 머신러닝 알고리즘의 작동 원리를 직관적으로 전달
- 딥러닝과 신경망의 기초부터 최신 LLM까지 커버
- PDF 자료를 참고하여 심화 내용 제공

[설명 스타일]
- 복잡한 수학보다는 **개념과 비유**를 우선
- 예시: "뉴런은 마치 라이트 스위치처럼 켜고 꺼진다", "경사 하강법은 산에서 내려오는 것" 등
- 어려운 용어는 쉬운 말로 풀어서 설명
- 학습자의 수준을 고려한 점진적 설명

[답변 방침]
- 모든 답변은 한국어로
- 질문에 대해 먼저 핵심을 비유로 설명
- 필요시 구체적인 예제나 코드 스니펫 제공
- PDF 자료가 검색되면 관련 내용을 참고하여 답변하고 출처 명시
- 모르는 내용: "제 학습 자료에서 확인할 수 없는 내용입니다"라고 답변
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Streamlit 설정
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="박정은 Q&A",
    page_icon="💬",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css');

/* 기본 폰트 */
html, body, [class*="css"] { font-family: 'Pretendard Variable', Pretendard, sans-serif; }

/* 페이지 배경 */
[data-testid="stAppViewContainer"] { background-color: #F6E7C6; }
[data-testid="stSidebar"] { background-color: #E3E3E3; }

/* 제목 스타일 */
h1, h2, h3, h4, h5, h6 { color: #807E79; }

/* 텍스트 색상 */
body, p, div, span { color: #807E79; }

/* 입력 필드 */
input, textarea, [class*="stTextInput"], [class*="stChatInput"] {
  background-color: #FFFFFF !important;
  border: 2px solid #BDCED3 !important;
  border-radius: 8px !important;
  color: #807E79 !important;
}

/* 버튼 */
button, [class*="stButton"] {
  background-color: #CFD4AE !important;
  color: #807E79 !important;
  border: none !important;
  border-radius: 6px !important;
  font-weight: 600;
}

button:hover, [class*="stButton"]:hover {
  background-color: #BDCED3 !important;
}

/* 참고 자료 박스 */
.source-box {
  background: #E8CCC5;
  border-left: 4px solid #BDCED3;
  padding: 0.8rem 1rem;
  font-size: 0.82rem;
  color: #807E79;
  border-radius: 4px;
  margin-top: 0.8rem;
  box-shadow: 0 2px 4px rgba(128, 126, 121, 0.1);
}

/* 분할선 */
hr { border-color: #BDCED3 !important; }

/* 캡션 텍스트 */
[class*="stCaption"] { color: #807E79; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI/ML 학습 도우미")
st.caption("머신러닝, 딥러닝, LLM에 대해 쉽고 재미있게 배워보세요!")

# ────────────────────────────────────────────────────────────────────────────
# 사이드바 — API 키 & RAG 인덱스 관리
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = st.text_input(
            "Anthropic API 키",
            type="password",
            placeholder="sk-ant-...",
        )

    st.divider()
    st.subheader("📚 PDF 인덱스")

    ready = is_index_ready()
    if ready:
        st.success("인덱스 준비 완료")
    else:
        st.warning("인덱스 없음 — 관리자 로그인 후 빌드하세요.")

    # 관리자 잠금 영역
    if "admin_unlocked" not in st.session_state:
        st.session_state.admin_unlocked = False

    if not st.session_state.admin_unlocked:
        admin_pw = st.text_input("관리자 비밀번호", type="password", placeholder="비밀번호 입력…")
        if st.button("로그인", use_container_width=True):
            correct = get_secret("ADMIN_PASSWORD", "admin1234")
            if admin_pw == correct:
                st.session_state.admin_unlocked = True
                st.rerun()
            else:
                st.error("비밀번호가 틀렸습니다.")
    else:
        st.success("관리자 모드")
        if st.button("🔄 인덱스 빌드 / 재빌드", use_container_width=True):
            with st.spinner("PDF 파싱 & 임베딩 중… (첫 실행 시 수 분 소요)"):
                try:
                    count = build_index()
                    st.success(f"완료: {count}개 청크 저장")
                    st.rerun()
                except FileNotFoundError as e:
                    st.error(str(e))
        if st.button("잠금", use_container_width=True, type="secondary"):
            st.session_state.admin_unlocked = False
            st.rerun()

    use_rag = st.toggle("RAG 사용", value=ready, disabled=not ready)

    st.divider()
    n_results = st.slider("검색할 청크 수", 1, 10, 5)

if not api_key:
    st.info("사이드바에서 Anthropic API 키를 입력하세요.")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

# ────────────────────────────────────────────────────────────────────────────
# 대화
# ────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("AI/ML에 대해 묻고 싶은 것을 입력하세요…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG 검색
    rag_context = ""
    if use_rag and ready:
        rag_context = query_context(prompt, n_results=n_results)

    # 시스템 프롬프트 구성
    system_prompt = PROFILE
    if rag_context:
        system_prompt += f"\n\n[PDF 자료에서 검색된 관련 내용]\n{rag_context}"

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중…"):
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1500,
                system=system_prompt,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
            )
            answer = response.content[0].text

        st.markdown(answer)

        # 참고 자료 표시
        if rag_context:
            sources = set()
            for line in rag_context.splitlines():
                if line.startswith("[출처:"):
                    src = line.split("|")[0].replace("[출처:", "").strip()
                    sources.add(src)
            if sources:
                st.markdown(
                    "<div class='source-box'>📄 참고 자료: "
                    + ", ".join(sorted(sources))
                    + "</div>",
                    unsafe_allow_html=True,
                )

        st.session_state.messages.append({"role": "assistant", "content": answer})

# 초기화 버튼
if st.session_state.messages:
    if st.button("대화 초기화", type="secondary"):
        st.session_state.messages = []
        st.rerun()
