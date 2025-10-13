

import os
import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import AsyncAnthropic
from anthropic.types import Message, TextBlock, ToolUseBlock

class RAGClient:
    """
    An intelligent client that orchestrates the RAG pipeline by using an LLM
    to call tools and build a complete conversation history.
    """

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = AsyncAnthropic()
        self.tools: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, Any]] = []

    async def connect(self, server_script_path: str = "rag_server.py"):
        """Connects to the RAG MCP server and gets the list of available tools."""
        print("RAG_CLIENT: Connecting to RAG MCP server...")
        server_params = StdioServerParameters(command="python", args=[server_script_path], env=os.environ)
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await self.session.initialize()
            
            mcp_tools = await self.session.list_tools()
            self.tools = [{"name": tool.name, "description": tool.description, "input_schema": tool.inputSchema} for tool in mcp_tools.tools]
            print(f"RAG_CLIENT: Successfully connected. Available tools: {[tool['name'] for tool in self.tools]}")
        except Exception as e:
            print(f"RAG_CLIENT: Failed to connect: {e}")
            raise

    def _serialize_messages(self) -> List[Dict[str, Any]]:
        """
        A robust function to convert complex Anthropic message objects into a simple, 
        JSON-serializable list of dictionaries that the frontend can display.
        """
        serializable_list = []
        for msg in self.messages:
            new_msg = {"role": msg["role"]}
            content = msg["content"]
            
            if isinstance(content, list) and content and hasattr(content[0], 'model_dump'):
                new_msg["content"] = [block.model_dump() for block in content]
            else:
                new_msg["content"] = content
            
            serializable_list.append(new_msg)
        return serializable_list

    async def process_query(self, query: str) -> List[Dict[str, Any]]:
        """Processes a query through the full RAG pipeline, building a detailed history."""
        self.messages = [{"role": "user", "content": query}]

        for _ in range(7):
            print(f"RAG_CLIENT: Calling LLM with {len(self.messages)} messages...")
            response: Message = await self.llm.messages.create(
                model="claude-3-5-sonnet-20241022", max_tokens=4096,
                messages=self.messages, tools=self.tools
            )

            self.messages.append({"role": "assistant", "content": response.content})
            
            has_tool_call = any(block.type == "tool_use" for block in response.content)

            if not has_tool_call:
                print("RAG_CLIENT: LLM provided a direct answer without tools. Ending process.")
                break 

            print("RAG_CLIENT: LLM requested tool calls.")
            tool_results = []
            should_break_loop = False
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_name = content_block.name
                    tool_input = content_block.input
                    tool_use_id = content_block.id
                    
                    print(f"RAG_CLIENT: Executing tool: {tool_name} with args: {tool_input}")
                    result = await self.session.call_tool(tool_name, tool_input)
                    final_content = result.content[0].text
                    
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": tool_use_id,
                        "content": final_content
                    })

            #         # --- THE FIX: Explicitly handle the final answer ---
            #         # If the answer generation agent was just called, we have our final answer.
                    if tool_name == "answer_generation_agent":
                        print("RAG_CLIENT: Final answer generated. Formatting for frontend.")
            #             # We manually create the final assistant message in the correct format.
                        final_assistant_message = {
                            "role": "assistant",
                            "content": [{"type": "text", "text": final_content}]
                        }
            #             # We replace the last message (which was the tool-use request) with this final answer.
            #             self.messages[-1] = final_assistant_message
            #             should_break_loop = True
            #             break # Exit the inner loop over content blocks
                        # We APPEND this final answer to the history.
                        self.messages.append(final_assistant_message)
                        should_break_loop = True
                        break # Exit the inner loop over content blocks
            
            if should_break_loop:
                break # Exit the main processing loop
            
            self.messages.append({"role": "user", "content": tool_results})
        
        return self._serialize_messages()

    async def cleanup(self):
        """Closes the connection."""
        print("RAG_CLIENT: Cleaning up and closing connection...")
        await self.exit_stack.aclose()

