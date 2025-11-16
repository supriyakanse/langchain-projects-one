from pydantic import BaseModel
from datetime import datetime

class EmailData(BaseModel):
    subject: str
    sender: str
    date: datetime
    body: str
