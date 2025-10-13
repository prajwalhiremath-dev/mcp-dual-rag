

from anthropic import Anthropic
import os
import json
from sql_client import SQLClient 
from rag_client import RAGClient   
import asyncio

class MasterOrchestrator:
    """
    A master client that routes questions to the appropriate specialized client (SQL or RAG or hybird).
    """

    def __init__(self):
        self.llm = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY1"))
        print("Master Client initialized.")

    def _route_question(self, question: str) -> str:
        """Uses an LLM to classify the user's question."""
        prompt = f"""
        You are a routing agent. Your job is to classify the user's question into one of two categories based on its content: "sql" or "rag".
        
        - If the question involves county (King, Clark, Kitsap, Pierce, Snohomish, Spokane, Thurstone), classify it as "sql".
        - If the question is a general knowledge query, asks for a definition, an explanation, or information about a company, classify it as "rag".
        - If the question REQUIRES information from BOTH the database and the documents to be fully answered, classify it as "hybrid".

        Respond with only a single word: "sql", "rag", or "hybrid".

        User Question: "{question}"
        
        Examples:
        - "Can you share me the Porsche Annual US sales 2003–2005 report along with respective models?" - rag
        - "How many vehicles are in King county?" -> sql
        - "What is the maximum base MSRP?" -> sql
        - "What is Tesla's mission statement?" -> rag
        - "Who is the CEO of BMW?" -> rag
        - "How many primary facilities operated by Tesla company and what are the industry collaboration from BMW company?" - rag
        - "How many total employees are in BMW groups from 2007 to 2023 ?" - rag
        - "Compare the number of vehicles in King county to Tesla's annual production numbers?" - hybrid
        
        """
        try:
            response = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            decision = response.content[0].text.strip().lower()
            if decision in ["sql", "rag", "hybrid"]:
                return decision
            return "rag"  # Default to RAG if the response is unexpected
        except Exception as e:
            print(f"Error in routing: {e}. Defaulting to RAG.")
            return "rag"
        
    async def _synthesize_hybrid_answer(self, question:str, sql_context:str, rag_context:str) -> str:
        """Takes context from both pipelines and generates a final, synthesized answer."""
        print("MASTER CLIENT : Synthesizing final answer from SQL and RAG contexts...")
        prompt = f"""
        You are a data analyst and expert assistant. Your task is to answer the user's question by synthesizing information from the different sources : a database query result and a document search result.
        
        User Question: "{question}"

        Source 1: Structured Data from Database
        SQL Context: "{sql_context}"

        Source 2: Unstructured Data from Documents  
        RAG Context: "{rag_context}"

        Based on all of this information, provide a single, comprehensive answer to the user's question.
        """
        try:
            response = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            # return response.content[0].text.strip()
            return response.content[0].text
        except Exception as e:
            print(f"Error synthesizing hybrid answer: {e}")
            return "Error synthesizing hybrid answer."

    async def process_query(self, query: str):
        """Routes the query and gets an answer from the appropriate client."""
        
        route = self._route_question(query)
        print(f"MASTER_ORCHESTRATOR : Routing decision for query '{query}': {route}")

        if route == "sql":
            sql_client = SQLClient()
            try:
                # The SQL client connects to server.py
                await sql_client.connect_to_server("server.py")
                response = await sql_client.process_query(query)
                return response
            finally:
                await sql_client.cleanup()
        elif  route == "rag":
            rag_client = RAGClient()
            try:
                # The RAG client connects to rag_server.py
                await rag_client.connect()
                response = await rag_client.process_query(query)
                return response
            finally:
                await rag_client.cleanup()
        elif route == "hybrid":
            print("MASTER CLIENT : Executing hybrid query processing...")
            #running the SQL and RAG sequentially
            #1. Run SQL Pipeline first
            sql_client = SQLClient()
            try:
                # Connect the SQL client to server.py
                await sql_client.connect_to_server("server.py")
                sql_history = await sql_client.process_query(query)
            finally:
                await sql_client.cleanup()
            
            #2. Run RAG Pipeline next
            rag_client = RAGClient()
            try:
                # Connect the RAG client to rag_server.py
                await rag_client.connect()
                rag_history = await rag_client.process_query(query) 
            finally:
                await rag_client.cleanup()

            #3. synthesize the final answer
            #Fix : Added a check to ensure the content list is not empty
            sql_answer = sql_history[-1]['content'][0]['text'] if sql_history and sql_history[-1]['role'] == 'assistant' and sql_history[-1]['content'] else "No SQL answer found."
            rag_answer = rag_history[-1]['content'][0]['text'] if rag_history and rag_history[-1]['role'] == 'assistant' and rag_history[-1]['content'] else "No RAG answer found."

            #3. Synthesize the final answer using both contexts
            # sql_answer = sql_history[-1]['content'][0]['text'] if sql_history and sql_history[-1]['role'] == 'assistant' else "No SQL answer found."
            # rag_answer = rag_history[-1]['content'][0]['text'] if rag_history and rag_history[-1]['role'] == 'assistant' else "No RAG answer found."

            # Synthesize the final answer
            final_answer = await self._synthesize_hybrid_answer(query, sql_answer, rag_answer)

            # Combine the histories for a complete view on the front end
            combined_history = [sql_history[0] if sql_history else []]
            if len(sql_history)>1:
                combined_history.extend(sql_history[1:-1])
            if len(rag_history)>1:
                combined_history.extend(rag_history[1:-1])
            combined_history.append(                
                 {"role": "assistant", "content": [{"type": "text", "text": final_answer}]}  # final synthesized answer
            )
            return combined_history

            # sql_client = SQLClient()
            # rag_client = RAGClient()
            # try:
            #     # Connect both clients
            #     await sql_client.connect_to_server("server.py")
            #     await rag_client.connect()

                # Execute both pipelines in parallel
                # sql_task = asyncio.create_task(sql_client.process_query(query))
                # rag_task = asyncio.create_task(rag_client.process_query(query))

                # sql_history, rag_history = await asyncio.gather(sql_task, rag_task)

                # #Extract the final answers/context from each pipeline's history
                # sql_answer = sql_history[-1]['context'][0]['text'] if sql_history and sql_history[-1]['role'] == 'assistant' else "No SQL answer found."
                # rag_answer = rag_history[-1]['context'][0]['text'] if rag_history and rag_history[-1]['role'] == 'assistant' else "No RAG answer found."

                # # Synthesize the final answer using both contexts
                # final_answer = await self._synthesize_hybrid_answer(query, sql_answer, rag_answer)
                
                #combine the histories for a complete view on the front end
                # we take the user query, then all steps from the sql agent, then all steps from RAG agent, then the final synthesized answer.
            #     combined_history = (
            #         [sql_history[0] + #user query
            #          sql_history[1:] +#sql agent steps
            #          rag_history[1:-1] + #rag agent steps (excluding the user query and final answer)
            #          [{"role": "assistant", "content": [{"type": "text", "text": final_answer}]}]] #final synthesized answer
            #     )
            #     return combined_history
            # finally:
            #     await sql_client.cleanup()
            #     await rag_client.cleanup()
            

