import os
import requests
import asyncio
import logging
from alpaca_trade_api.stream import Stream

# --- CONFIGURATION ---
API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = "https://paper-api.alpaca.markets"
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

processed_events = set()


def send_discord_alert(message, color=None):
    if color == 'green':
        color_code = 5763719
    elif color == 'red':
        color_code = 15548997
    elif color == 'gold':
        color_code = 16776960
    else:
        color_code = 3447003

    data = {
        "embeds": [
            {
                "description": message,
                "color": color_code
            }
        ]
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=data)
    except Exception as e:
        logger.error(f"Failed to send Discord alert: {e}")


async def trade_update_handler(data):
    event = data.event
    order = data.order

    # 1. GENERATE A UNIQUE ID FOR THIS EVENT
    event_signature = f"{order['id']}_{event}_{order['filled_qty']}"

    # 2. CHECK IF WE SAW THIS ALREADY
    if event_signature in processed_events:
        return  
    # 3. ADD TO MEMORY
    processed_events.add(event_signature)

    if len(processed_events) > 1000:
        processed_events.clear()

    symbol = order['symbol']
    side = order['side'].upper()
    filled_qty = order['filled_qty']
    price = order['filled_avg_price']
    order_type = order['type']

    if event == 'fill' or event == 'partial_fill':

        total_value = float(filled_qty) * float(price) if price else 0
        client_id = order.get('client_order_id', '')

        icon = "🔔"
        color = "blue"
        action = f"{side} {symbol}"

        if "harvest" in client_id:
            icon = "🌾"
            color = "green"
            action = f"HARVEST WIN: {symbol}"
        elif "take_profit" in client_id:
            icon = "💰"
            color = "green"
            action = f"TAKE PROFIT: {symbol}"
        elif "trailing_stop" in client_id:
            icon = "🛡️"
            color = "gold"
            action = f"RATCHET STOP: {symbol}"
        elif "stop" in client_id:
            icon = "🛑"
            color = "red"
            action = f"STOP LOSS: {symbol}"
        elif "entry" in client_id:
            icon = "🚀"
            color = "blue"
            action = f"ENTRY: {symbol}"

        msg = (
            f"**{icon} ORDER FILLED**\n"
            f"**Action:** {action}\n"
            f"**Qty:** {filled_qty} @ ${float(price):.2f}\n"
            f"**Value:** ${total_value:,.2f}\n"
            f"**Type:** {order_type.upper()}"
        )

        logger.info(f"Sending Alert: {action}")
        send_discord_alert(msg, color)


async def auto_disconnect():
    # Wait 5 hours 55 minutes to restart cleanly before GitHub kills it
    await asyncio.sleep(21300)
    logger.warning("⏰ Time limit reached. Disconnecting...")
    os._exit(0)


def run_listener():
    logger.info("--- 🎧 DISCORD LISTENER ACTIVE (With De-Duplication & Logging) ---")

    stream = Stream(API_KEY, SECRET_KEY, base_url=BASE_URL, data_feed='iex')
    stream.subscribe_trade_updates(trade_update_handler)

    loop = asyncio.get_event_loop()
    loop.create_task(auto_disconnect())

    try:
        loop.run_until_complete(stream.run())
    except Exception as e:
        logger.error(f"Stream Error: {e}")
        time.sleep(5)
        run_listener()


if __name__ == "__main__":
    import time
    run_listener()
