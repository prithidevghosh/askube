from motor.motor_asyncio import AsyncIOMotorClient

client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client["askube"]

transcript_collection = db["transcript_data"]
chats_collection = db["chats_data"]
