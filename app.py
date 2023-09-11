import streamlit as st
import os

import pandas as pd
import io

from Kitsune import *
from scipy.stats import norm
import openai

openai.api_key = os.environ["FREEEDU_OPENAI_API_KEY"]

project_title = "ChatGPT IDS Interpreter"
project_icon = "Fox512px.png"
st.set_page_config(page_title=project_title, initial_sidebar_state='collapsed', page_icon=project_icon)

def run_kitsune(kitsune, FMgrace, ADgrace, context=1, threshold=-2):
    print("Running Kitsune:")
    packets = deque(maxlen=context * 2 + 1)  # Store the last 'context' packets and the next 'context' packets
    anomalous_packets = []  # Store sequences of packets that include anomalies
    i = 0
    start = time.time()
    RMSEs = []

    while True:
        i += 1
        if i % 1000 == 0:
            print(i)
        result = kitsune.proc_next_packet()
        if isinstance(result, int) and result == -1:  # No packets left
            break

        rmse, packet = result  # Packet still available
        RMSEs.append(rmse)

        if i > FMgrace + ADgrace:  # We start checking for anomalies after FMgrace + ADgrace packets
            packets.append(packet)

            if len(RMSEs) > 100000:
                benignSample = np.log(RMSEs[FMgrace + ADgrace + 1:100000])
            else:
                benignSample = np.log(RMSEs[FMgrace + ADgrace + 1:])
            Mean = np.mean(benignSample)
            Std = np.std(benignSample)

            log_prob = norm.logsf(np.log(rmse), Mean, Std)
            if log_prob < threshold and len(packets) == context * 2 + 1:  # If the packet is anomalous and we have enough context
                anomalous_packets.append(list(packets))  # Add the sequence of packets to our list

    stop = time.time()
    print("Complete. Time elapsed: " + str(stop - start))

    return anomalous_packets, RMSEs

def analyze_with_chatgpt(anomalous_packets):
    context = {"role":"system",
               "content":"""
               You are a packet data interpreter. You will receive multiple messages in sequential order, each containing packet data for you to analyze and interpret.
               The second message contains packet data flagged as an anomaly; your job is to interpret this packet and identify a probable cause, utilizing 
               the packets that came before and after as a comparison point against the anomaly, and then formulate possible advice for the user. 
               Your response should be 50 to 150 words only; be as concise as possible and address the user directly, only mentioning the key issues and cause of the anomaly
               and never mentioning any information before this sentence.
               Assume that the user has no knowledge of networks or cybersecurity, so be very discrete in your descriptions   
               of the intrusion and only start your response with "Based on your packet data," never mentioning the 'middle', 'before', or 'after' packet, or 'packet/s in the list'.
               Try to avoid telling the user 'if the issue persists', as the user might not be able to easily tell if there is an issue with their network.
               Give simple step-by-step advice on what the user can do on their own; or directly inform them 
               Notify the user if the assistance of a professional is necessary for the level of intrusion you found.
               """}
    message_prompt = []
    message_prompt.append(context)

    # Build a sequence of messages, with each message being a single packet up to the last packet flagged as anomalous
    for p in anomalous_packets:
        if isinstance(p, np.ndarray):
            p = p.tolist()  # Convert numpy ndarray to list
        message_format = {"role":"user",
                          "content":str(p)}
        message_prompt.append(message_format)

    # Generate the response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_prompt,
        max_tokens=1008,
        temperature=0.7,
        n=1,
        stop=None,
    )

    # Extract the response text from the API response
    response_text = response['choices'][0]['message']['content']
    # Return the response text and updated chat history
    return response_text

def plot_results(logProbs):
    pass

def main():
    uploaded_files = st.file_uploader("Upload a PCAP file:",
                               type=["pcap","PCAP"],
                               accept_multiple_files=False)

#!!!#
# In the sample Mirai pcap (a recording of the Mirai botnet malware being activated):
# The first 70,000 observations are clean...
#!!!#

    with st.form(key="run_kitsune_form"):
        settings_expander = st.expander("Kitsune Settings")

        packet_limit = settings_expander.number_input("Packet Limit (default: 'no limit')", 70000, 10000000, 10000000,
                                                      10000)
        maxAE = settings_expander.number_input("Max Autoencoder Size", 1, 30, 10, 1)
        FMgrace = settings_expander.number_input("Feature Mapping Grace training period", 1000, 15000, 5000, 1000)
        ADgrace = settings_expander.number_input("Anomaly Detector Grace training period", 10000, 150000, 50000, 1000)
        run_button = st.form_submit_button("Run Kitsune")

    if "anomalous_packets" not in st.session_state:
        st.session_state.anomalous_packets = []
    if "RMSEs" not in st.session_state:
        st.session_state.RMSEs = []

    # button is clicked
    if run_button:
        if uploaded_files is None: # if no files were uploaded:
            st.error("Please upload a pcap file")
        else:
            # Ensure temp directory exists
            if not os.path.exists('temp'):
                os.makedirs('temp')

            filename = os.path.basename(uploaded_files.name)
            file_path = os.path.join("temp", filename)

            # Write out the uploaded file to temp directory
            with open(file_path, 'wb') as f:
                f.write(uploaded_files.getbuffer())

            # Build Kitsune
            K = Kitsune(file_path, packet_limit, maxAE, FMgrace, ADgrace)
            with st.spinner('Running Kitsune...'):
                st.session_state.anomalous_packets, st.session_state.RMSEs = run_kitsune(K, FMgrace, ADgrace)

    st.write("Anomalous packets")
    st.write(len(st.session_state.anomalous_packets))

    if len(st.session_state.anomalous_packets) > 0:
        # Displaying the anomalous packets and sending to ChatGPT
        selected_packet_index = st.selectbox('Select a packet', list(range(len(st.session_state.anomalous_packets))), 0)
        selected_packet = st.session_state.anomalous_packets[selected_packet_index]
        st.write(selected_packet)
        if st.button('Send to ChatGPT'):
            result = analyze_with_chatgpt(selected_packet)
            st.write(result)

if __name__ == '__main__':
    main()