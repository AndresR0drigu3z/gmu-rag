"""Streamlit chat UI.  Run with:  streamlit run src/app.py"""
import streamlit as st

from config import settings
from rag import answer

st.set_page_config(page_title="GMU Advisor (RAG)", page_icon="🎓")
st.title("🎓 GMU Academic Advisor")
st.caption(f"Answers come only from indexed GMU documents · model: {settings.llm_model}")

with st.sidebar:
    k = st.slider("Chunks to retrieve (top-k)", 1, 10, settings.top_k)

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about courses, requirements, policies..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = answer(question, k)
        st.markdown(result["answer"])
        with st.expander("Sources"):
            for i, s in enumerate(result["sources"], start=1):
                page = f", p.{s['page']}" if s["page"] else ""
                how = "exact course match" if s.get("match") == "exact" else f"distance {s['distance']:.3f}"
                st.markdown(f"**[{i}] {s['source']}{page}** · {how}")
                st.text(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))
    st.session_state.history.append({"role": "assistant", "content": result["answer"]})
