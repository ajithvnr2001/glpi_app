import requests
import json
from typing import List, Dict, Optional

class GLPIConnector:
    def __init__(self, glpi_url: str, app_token: str, user_token:Optional[str]=None):
        self.glpi_url = glpi_url
        self.app_token = app_token
        self.headers = {
            "Content-Type": "application/json",
            "App-Token": self.app_token,
        }
        self.session_token = None
        self.user_token=user_token

    def init_session(self) -> bool:
        """Initializes a GLPI session and gets the session token."""
        init_url = f"{self.glpi_url}/initSession"
        if self.user_token:
            self.headers["Authorization"]=f"user_token {self.user_token}"
        try:
            response = requests.get(init_url, headers=self.headers)
            response.raise_for_status()
            self.session_token = response.json().get("session_token")
            if self.session_token:
                self.headers["Session-Token"] = self.session_token
                if "Authorization" in self.headers:
                  del self.headers["Authorization"]
                return True
            else:
                print("Error: Could not initialize GLPI session.")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Error initializing session: {e}")
            return False

    def kill_session(self) -> bool:
        """Kills the current GLPI session"""
        kill_url = f"{self.glpi_url}/killSession"

        if not self.session_token:
          return True

        try:
            response = requests.get(kill_url, headers=self.headers)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error killing session: {e}")
            return False

    def get_tickets(self, range_str:str="0-10") -> List[Dict]:
        """Retrieves a list of tickets from GLPI."""
        if not self.session_token:
            if not self.init_session():
                return []

        tickets_url = f"{self.glpi_url}/Ticket?range={range_str}"
        try:
            response = requests.get(tickets_url, headers=self.headers)
            response.raise_for_status()
            tickets = response.json()

            extracted_tickets = []
            for ticket in tickets:
                extracted_tickets.append({
                    "id": ticket.get("id"),
                    "name": ticket.get("name"),
                    "content": ticket.get("content"),
                    "status": ticket.get("status"),
                    "date": ticket.get("date"),
                })
            return extracted_tickets

        except requests.exceptions.RequestException as e:
            print(f"Error retrieving tickets: {e}")
            return []
            
    def get_ticket(self,id) -> Dict:
        """Retrieves a specific ticket from GLPI."""
        if not self.session_token:
            if not self.init_session():
                return []

        tickets_url = f"{self.glpi_url}/Ticket/{id}"
        try:
            response = requests.get(tickets_url, headers=self.headers)
            response.raise_for_status()
            ticket = response.json()
            return ticket

        except requests.exceptions.RequestException as e:
            print(f"Error retrieving tickets: {e}")
            return []
