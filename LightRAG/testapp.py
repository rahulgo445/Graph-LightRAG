import streamlit as st
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete

st.title("LLM GenAI Billing Agent")

# Move the API key to the sidebar
with st.sidebar:
    st.header("Configuration")
    # Input for OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    # File uploader for XML files
    uploaded_files = st.file_uploader("Upload XML files", type=["xml"], accept_multiple_files=True)

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Read the agent prompt from prompt.txt
@st.cache_data
def read_agent_prompt():
    with open("./dickens/prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt

agent_prompt = read_agent_prompt()

# Read the business rules from BusinessRules.txt
@st.cache_data
def read_business_rules():
    with open("./dickens/BusinessRules.txt", "r", encoding="utf-8") as f:
        data = f.read()
    return data

business_rules = read_business_rules()

# Read uploaded XML files
def get_xml_contents():
    xml_contents = ""
    if uploaded_files:
        for uploaded_file in uploaded_files:
            content = uploaded_file.read().decode("utf-8")
            xml_contents += f"\n<START_XML_{uploaded_file.name}>\n{content}\n<END_XML_{uploaded_file.name}>\n"
    return xml_contents

xml_contents = get_xml_contents()

# Set OpenAI API key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Set up working directory
    WORKING_DIR = "./dickens"
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    # Initialize LightRAG
    @st.cache_resource(show_spinner=False)
    def get_rag():
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
        )
        # Insert business rules
        rag.insert(business_rules)
        return rag

    rag = get_rag()

    # Display conversation history
    st.subheader("Conversation")
    for i, msg in enumerate(st.session_state['messages']):
        if msg['role'] == 'user':
            st.markdown(f"**User:** {msg['content']}")
        else:
            st.markdown(f"**Assistant ({msg['mode']}):** {msg['content']}")

    # User input within a form
    with st.form(key='chat_form'):
        user_input = st.text_input("Your message:", key='user_input')
        submit = st.form_submit_button('Send')
        if submit and user_input:
            # Append user message to conversation history
            st.session_state['messages'].append({"role": "user", "content": user_input})

            # Construct conversation history for the prompt
            conversation_history = ""
            for msg in st.session_state['messages']:
                if msg['role'] == 'user':
                    conversation_history += f"\nUser: {msg['content']}"
                else:
                    conversation_history += f"\nAssistant: {msg['content']}"

            # Combine all parts into the final prompt
            combined_prompt = f"{agent_prompt}\n\n{xml_contents}\n\nConversation History:{conversation_history}\n\nAssistant:"

            # Perform queries
            with st.spinner('Processing your message...'):
                modes = ['naive', 'local', 'global', 'hybrid']
                results = {}
                for mode in modes:
                    param = QueryParam(mode=mode)
                    response = rag.query(combined_prompt, param=param)
                    results[mode] = response

            # Display results side by side with sectioning
            st.subheader("Results")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Naive")
                st.write(results['naive'])
            with col2:
                st.markdown("### Local")
                st.write(results['local'])

            col3, col4 = st.columns(2)
            with col3:
                st.markdown("### Global")
                st.write(results['global'])
            with col4:
                st.markdown("### Hybrid")
                st.write(results['hybrid'])

            # Append assistant's response to conversation history
            # Let's use 'hybrid' mode's response
            assistant_response = results['hybrid']
            st.session_state['messages'].append({"role": "assistant", "content": assistant_response, "mode": "hybrid"})

            # The form submission resets the input field automatically

else:
    st.write("Please enter your OpenAI API key in the sidebar to start the conversation.")
