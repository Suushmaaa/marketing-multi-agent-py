# load_data.py
import asyncio
from database.loader import DataLoader
from mcp.client import create_mcp_client
from config.settings import settings

async def main():
    # Connect to MCP
    client = create_mcp_client(
        name="data_loader",
        url=f"ws://localhost:9001",
        transport_type="websocket"
    )
    await client.connect()

    # Load data
    loader = DataLoader(data_dir=settings.DATA_DIR)
    success = await loader.load_all_data(client)

    print(f"Data loaded: {loader.get_stats()}")

    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
