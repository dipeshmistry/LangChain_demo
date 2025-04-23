import streamlit as st
from agents import get_claude_agent

st.set_page_config(page_title="Claude Agent Assistant", layout="wide")
st.title("🧠 Claude-powered Agent Chat")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

agent = get_claude_agent()

# User input
query = st.chat_input("Ask the Claude agent...")

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# Process new input
if query:
    st.session_state.chat_history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Claude is thinking..."):
            try:
                response = agent.run(query)
                st.markdown(response)
                st.session_state.chat_history.append(("assistant", response))
            except Exception as e:
                st.error(f"Error: {e}")
