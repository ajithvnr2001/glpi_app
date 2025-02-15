import os
from fastapi import FastAPI, Request, BackgroundTasks
from glpi_connector import GLPIConnector
from llm_service import LLMService
from pdf_generator import PDFGenerator
from dotenv import load_dotenv
from typing import Dict, Any, List
import uvicorn
import json

load_dotenv()

app = FastAPI()

glpi_url = os.getenv("GLPI_URL")
app_token = os.getenv("GLPI_APP_TOKEN")
user_token = os.getenv("GLPI_USER_TOKEN")
glpi = GLPIConnector(glpi_url, app_token, user_token)
llm_service = LLMService()

async def process_ticket(ticket_id: int):
    try:
        ticket = glpi.get_ticket(ticket_id)
        if not ticket:
            print(f"Error: Could not retrieve ticket with ID {ticket_id}")
            return
        tickets=[ticket]

        query = "Give me a summary of this ticket."
        rag_result = llm_service.rag_completion(tickets, query)
        print(f"RAG Result: {rag_result}")

        pdf_generator = PDFGenerator(f"glpi_ticket_{ticket_id}.pdf")
        source_info = [{"source_id": ticket_id, "source_type": "glpi_ticket"}]
        pdf_generator.generate_report(
            f"GLPI Ticket Summary - ID: {ticket_id}", query, rag_result, source_info
        )
        print(f"Report generated: glpi_ticket_{ticket_id}.pdf")

    except Exception as e:
        print(f"Error processing ticket {ticket_id}: {e}")
    finally:
      glpi.kill_session()

@app.post("/webhook")
async def glpi_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        data: List[Dict[str, Any]] = await request.json()
        print(f"Received webhook data: {data}")

        for event in data:
            if event.get("event") == "add" and event.get("itemtype") == "Ticket":
                ticket_id = int(event.get("items_id"))
                background_tasks.add_task(process_ticket, ticket_id)
                return {"message": f"Ticket processing initiated for ID: {ticket_id}"}

        return {"message": "Webhook received, but no relevant event found."}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON payload"}
    except Exception as e:
      return {"error":str(e)}

@app.get("/test_llm")
async def test_llm_endpoint():
    test_prompt = "what is the capital of Assyria"
    response = llm_service.complete(prompt=test_prompt)
    return {"response":response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
