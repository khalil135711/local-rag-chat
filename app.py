import streamlit as st

import rag

st.set_page_config(page_title="Local RAG Chat", page_icon="📚", layout="wide")
st.title("📚 Local RAG Chat")
st.caption(f"100% offline · LLM: `{rag.LLM_MODEL}` · Embeddings: `{rag.EMBED_MODEL}` · no API keys")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Index uploaded files", disabled=not files, use_container_width=True):
        for f in files:
            with st.spinner(f"Indexing {f.name}..."):
                n = rag.ingest_file(f.name, f.getvalue())
            st.success(f"{f.name}: {n} chunks")

    st.subheader("Indexed")
    for src in rag.list_sources():
        c1, c2 = st.columns([4, 1])
        c1.write(f"📄 {src}")
        if c2.button("✕", key=f"del-{src}"):
            rag.delete_source(src)
            st.rerun()

    st.divider()
    agent = st.toggle("Agent mode (router + calculator)", value=False)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            history = st.session_state.messages[:-1]
            stream, sources, tool = rag.answer(question, history, agent=agent)
            reply = st.write_stream(stream)
            if agent:
                st.caption(f"🛠 tool used: {tool}")
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"**{s['source']}** · page {s['page']} · score {s['score']:.2f}")
                        st.text(s["text"][:400])
        except Exception as e:
            reply = f"⚠️ Could not reach Ollama or the model: `{e}`"
            st.error(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
