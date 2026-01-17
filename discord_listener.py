import os
import requests
import asyncio
from alpaca_trade_api.stream import Stream

# --- CONFIGURATION ---
API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = "https://paper-api.alpaca.markets"

# 🔴 PASTE YOUR DISCORD WEBHOOK URL HERE
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')


def send_discord_alert(message, color=None):
    """
    Sends a styled message to Discord.
    Colors: Green (Success), Red (Failure/Stop), Blue (Info)
    """
    if color == 'green':
        color_code = 5763719  # 0x57F287
    elif color == 'red':
        color_code = 15548997  # 0xED4245
    elif color == 'gold':
        color_code = 16776960  # 0xFFFF00
    else:
        color_code = 3447003  # Blue

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
        print(f"Failed to send Discord alert: {e}")


async def trade_update_handler(data):
    """
    Listens for real-time trade updates from Alpaca.
    """
    # data is an object. We access attributes directly.
    event = data.event
    order = data.order

    symbol = order['symbol']
    side = order['side'].upper()
    qty = order['qty']
    price = order['filled_avg_price']
    order_type = order['type']

    # We only care about FILLS (Full or Partial)
    if event == 'fill' or event == 'partial_fill':

        # Calculate Value
        total_value = float(qty) * float(price) if price else 0

        # Determine Context (Entry vs Exit)
        # If we Bought to Open (Long) or Sold to Open (Short), it's an ENTRY
        # If we Sold to Close (Long) or Bought to Close (Short), it's an EXIT

        # Simple Logic: Look at the Client Order ID to guess the intent
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
            f"**Qty:** {qty} @ ${float(price):.2f}\n"
            f"**Value:** ${total_value:,.2f}\n"
            f"**Type:** {order_type.upper()}"
        )

        print(f"Sending Alert: {action}")
        send_discord_alert(msg, color)


def run_listener():
    print("--- 🎧 DISCORD LISTENER ACTIVE ---")
    print("Waiting for trade updates...")

    # Test Message
    send_discord_alert("🤖 **Alpaca Bot Connected**\nListening for fills...", "blue")

    stream = Stream(API_KEY, SECRET_KEY, base_url=BASE_URL, data_feed='iex')
    stream.subscribe_trade_updates(trade_update_handler)

    try:
        stream.run()
    except Exception as e:
        print(f"Stream Error: {e}")
        # Auto-restart on error
        time.sleep(5)
        run_listener()


if __name__ == "__main__":
    import time

    run_listener()