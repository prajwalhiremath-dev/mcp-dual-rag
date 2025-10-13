

import asyncio
import streamlit as st
from chatbot import Chatbot



async def main():
    # st.set_page_config(page_title="MCP SQL Agent", page_icon="🤖")
    # Configuration
    API_URL = "http://localhost:8000/api/v1"  # Updated to include the path pattern
    chatbot = Chatbot(API_URL)
    await chatbot.render()

if __name__ == "__main__":
    asyncio.run(main())

