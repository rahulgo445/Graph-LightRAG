

import streamlit as st
import os
import networkx as nx
from pyvis.network import Network
import random
import tempfile
import matplotlib.colors as mcolors
import openai
from langsmith import traceable
from langsmith.wrappers import wrap_openai

# Wrap the openai module
#openai = wrap_openai(openai)
os.environ["OPENAI_API_KEY"] = "OPENAI_KEY"
wrap_openai(openai.Client())

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "LLMGenAIBillingAgent"

from lightrag import LightRAG, QueryParam
#from lightrag.llm import gpt_4o_mini_complete
from lightrag.llm import gpt_4o_complete

langsmith_api_key ="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"

additional_prompt = (
    "Do not provide answers that are outside of the provided documents and business rules. "
    "Do not make assumption and be factual in the application of business rules.\n\n"
    "Please figure out the best possible answer to the last user query from the conversation above."
)

st.title("LLM GenAI Billing Agent")

# Move the API key and mode selection to the sidebar
with st.sidebar:
    st.header("Configuration")
    # Input for OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    # Mode selection
    mode = st.radio("Select Mode", ('naive', 'local', 'global', 'hybrid'), key='mode')
    # File uploader for XML files
    uploaded_files = st.file_uploader("Upload XML files", type=["xml"], accept_multiple_files=True)
    # Finish Session button
    if st.button("Finish Session"):
        # Clear the conversation messages but keep other session state variables
        st.session_state['messages'] = []

# Initialize session state for messages if not already done
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Initialize session state for user input
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

# Read the agent prompt from prompt.txt
@traceable
@st.cache_data
def read_agent_prompt():
    with open("./dickens/prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt

agent_prompt = read_agent_prompt()

# Read the business rules from BusinessRules.txt
@traceable
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
    openai.api_key = openai_api_key

    if langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = "LLMGenAIBillingAgent"
    else:
        st.warning("Please enter your LangSmith API key in the sidebar to enable tracing.")

    # Set up working directory
    WORKING_DIR = "./dickens"
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    # Initialize LightRAG
    @traceable
    @st.cache_resource(show_spinner=False)
    def get_rag():
        rag = LightRAG(
            working_dir=WORKING_DIR,
            #llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
            llm_model_func=gpt_4o_complete  # Use gpt_4o_mini_complete LLM model
        )
        # Insert business rules
        rag.insert(business_rules)
        return rag

    rag = get_rag()

    # Generate and display the graph
    st.subheader("Knowledge Graph")

    graphml_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
    if os.path.exists(graphml_file):
        # Load the GraphML file
        G = nx.read_graphml(graphml_file)

        # Extract node types
        node_types = set()
        for node, data in G.nodes(data=True):
            node_type = data.get('type', 'Undefined')
            node_types.add(node_type)

        
        color_list = [
            'green', 'red', 'blue', 'orange', 'purple',
            'cyan', 'magenta', 'yellow', 'black', 'gray', 'brown'
        ]

        # Extract node labels or types
        node_labels = set()
        for node, data in G.nodes(data=True):
            # Try to get 'type' or 'label' attribute
            node_label = data.get('type') or data.get('label') or 'Undefined'
            node_labels.add(node_label)

        # Assign colors to node labels
        type_color_map = {}
        for idx, node_label in enumerate(sorted(node_labels)):
            type_color_map[node_label] = color_list[idx % len(color_list)]

        # Assign colors to nodes based on their label
        for node, data in G.nodes(data=True):
            node_label = data.get('type') or data.get('label') or 'Undefined'
            color = type_color_map.get(node_label, 'green')
            data['color'] = color  # For NetworkX
            data['title'] = f"{node_label}: {node}"  # Tooltip for the node


        # Create a PyVis network
        net = Network(height="600px", width="100%", notebook=False, directed=True)

        # Set options to enable zooming and panning
        net.set_options("""
        var options = {
          "nodes": {
            "shape": "dot",
            "size": 16,
            "font": {
              "size": 14
            }
          },
          "edges": {
            "width": 0.5,
            "arrows": {
              "to": {
                "enabled": true
              }
            },
            "font": {
              "size": 12,
              "align": "middle"
            },
            "smooth": true
          },
          "interaction": {
            "navigationButtons": true,
            "keyboard": true,
            "zoomView": true,
            "dragView": true
          },
          "physics": {
            "stabilization": false
          }
        }
        """)

        # Convert NetworkX graph to PyVis network
        net.from_nx(G)

        # Add edge labels to display relationships
        for edge in net.edges:
            source = edge['from']
            target = edge['to']
            data = G.get_edge_data(source, target)
            # Handle multiple edges between nodes
            if isinstance(data, dict):
                # For NetworkX versions >=2.0, data is a dict of dicts
                # Get the first edge data
                edge_data = data.get(0, data)
            else:
                edge_data = data
            edge_label = edge_data.get('relation', '')
            edge['title'] = edge_label  # Shows up on hover
            edge['label'] = edge_label  # Shows up on the edge

        # Save and display the network
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            html_content = open(tmp_file.name, 'r', encoding='utf-8').read()
            st.components.v1.html(html_content, height=600, scrolling=True)

        # Display the legend
        st.subheader("Legend")
        for node_type, color in type_color_map.items():
            st.markdown(f"<span style='color:{color}; font-size: 20px;'>■</span> {node_type}", unsafe_allow_html=True)
    else:
        st.warning("Knowledge graph file not found. Please ensure that 'graph_chunk_entity_relation.graphml' exists in the './dickens/' directory.")

    # Placeholder for conversation
    st.subheader("Conversation")

    # Display conversation history
    conversation_placeholder = st.container()

    # Define a callback function to process user input
    # 
    
    # Define a callback function to process user input
# def process_input():
#     user_input = st.session_state['user_input']
#     if user_input:
#         # Append user message to conversation history
#         st.session_state['messages'].append({"role": "user", "content": user_input})

#         # Construct conversation history for the prompt, excluding the last user query
#         conversation_history = ""
#         for msg in st.session_state['messages'][:-1]:
#             if msg['role'] == 'user':
#                 conversation_history += f"\nUser: {msg['content']}"
#             else:
#                 conversation_history += f"\nAssistant: {msg['content']}"

#         # Get the last user query
#         last_user_query = st.session_state['messages'][-1]['content'] if st.session_state['messages'] else ""

#         # Combine all parts into the final prompt
#         combined_prompt = (
#             f"{agent_prompt}\n\n{xml_contents}\n\n{additional_prompt}\n\n"
#             f"Conversation History:{conversation_history}\n\n"
#             f"User: {last_user_query}\n\nAssistant:"
#         )

#         # Perform query using selected mode
#         with st.spinner('Processing your message...'):
#             selected_mode = mode  # Use the mode selected in the sidebar
#             param = QueryParam(mode=selected_mode)
#             assistant_response = rag.query(combined_prompt, param=param)

#         # Append assistant's response to conversation history
#         st.session_state['messages'].append({
#             "role": "assistant",
#             "content": assistant_response,
#             "mode": selected_mode
#         })

#         # Clear the input field
#         st.session_state['user_input'] = ''

# Define a callback function to process user input
    @traceable
    def process_input():
        user_input = st.session_state['user_input']
        if user_input:
            # Append user message to conversation history
            st.session_state['messages'].append({"role": "user", "content": user_input})

            # Construct conversation history for the prompt, excluding the last user query
            conversation_history = ""
            for msg in st.session_state['messages'][:-1]:
                if msg['role'] == 'user':
                    conversation_history += f"\nUser: {msg['content']}"
                else:
                    conversation_history += f"\nAssistant: {msg['content']}"

            # Get the last user query
            last_user_query = st.session_state['messages'][-1]['content'] if st.session_state['messages'] else ""

            # Combine all parts into the final prompt
            combined_prompt = (
                f"{agent_prompt}\n\n{xml_contents}\n\n{additional_prompt}\n\n"
                f"Conversation History:{conversation_history}\n\n"
                f"User: {last_user_query}\n\nAssistant:"
            )

            # Perform query using selected mode
            with st.spinner('Processing your message...'):
                selected_mode = mode  # Use the mode selected in the sidebar
                param = QueryParam(mode=selected_mode)
                
                assistant_response = rag.query(combined_prompt, param=param)
                #retrieved_context = rag.get_context()


                #print("." * 80 + "\n" + retrieved_context + "\n" + "." * 80 + "\n" + assistant_response + "\n" + "." * 80)
                

            # Append assistant's response to conversation history
            st.session_state['messages'].append({
                "role": "assistant",
                "content": assistant_response,
                "mode": selected_mode
            })

            # Clear the input field
            st.session_state['user_input'] = ''



    # User input at the bottom with on_change callback
    st.text_input("Your message:", key='user_input', on_change=process_input)

    # Display the conversation history
    with conversation_placeholder:
        for msg in st.session_state['messages']:
            if msg['role'] == 'user':
                st.markdown('<hr style="border:1px dashed #000;">', unsafe_allow_html=True)
                st.markdown(f"**User:** {msg['content']}")
                st.markdown('<hr style="border:1px dashed #000;">', unsafe_allow_html=True)
            else:
                st.markdown(f"**Assistant ({msg['mode']}):** {msg['content']}")

else:
    st.write("Please enter your OpenAI API key in the sidebar to start the conversation.")


