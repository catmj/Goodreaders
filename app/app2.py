import streamlit as st

st.title("Streamlit Input Reactivity Test")

st.markdown("""
Please type something into the text box below.
Observe if the 'You typed:' message updates in real-time as you type.
""")

# A simple text input widget
user_input = st.text_input("Type here:", key="debug_input_field")

# Display the value of the text input
st.write(f"You typed: `{user_input}`")

st.markdown("""
---
**What to observe:**
* If the 'You typed:' message **updates immediately** as you type each character, then Streamlit's basic text input reactivity is working correctly in your environment.
* If it **does NOT update** as you type, then there might be an issue with your Streamlit installation, browser, or environment setup that is preventing real-time updates.

Please let me know your observation!
""")