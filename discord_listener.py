import os
import requests
import asyncio
import logging
import nest_asyncio
from alpaca_trade_api.stream import Stream
from functools import partial

nest_asyncio.apply()

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


def send_discord_alert_sync(message, color=None):
    if not DISCORD_WEBHOOK_URL:
        logger.error("Discord Webhook URL not found")
        return

    if color == 'green':
        color_code = 5763719
    elif color == 'red':
        color_code = 15548997
    elif color == 'gold':
        color_code = 16776960
    else:
        color_code = 3447003

    data = {
        "embeds": [{
            "description": message,
            "color": color_code
        }]
    }
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data)
        if response.status_code not in [200,204]:
            print(f"❌ Discord Response Code: {response.status_code}")
        else:
            print("✅ Message sent successfully")
            pass
    except Exception as e:
        logger.error(f"Failed to send Discord alert: {e}")


async def send_discord_alert_async(message, color=None):
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(send_discord_alert_sync, message, color)
        )
    except Exception as e:
        logger.error(f"Async Alert Error: {e}")


async def trade_update_handler(data):
    try:
        event = data.event
        order = data.order

        event_signature = f"{order['id']}_{event}_{order['filled_qty']}"

        if event_signature in processed_events:
            return
        processed_events.add(event_signature)

        if len(processed_events) > 1000:
            processed_events.clear()

        symbol = order['symbol']
        side = order['side'].upper()
        filled_qty = order['filled_qty']
        price = order['filled_avg_price']
        order_type = order['type']

        is_option = len(symbol) > 6

        if event == 'fill':
            total_value = float(filled_qty) * float(price) if price else 0
            client_id = order.get('client_order_id', '')

            icon = "🔔";
            color = "blue";
            action = f"{side} {symbol}"
            if is_option:
                icon = "🧱";
                color = "magenta";
                action = f"OPTION: {symbol}"

            if "harvest" in client_id:
                icon = "🌾";
                color = "green";
                action = f"HARVEST WIN: {symbol}"
            elif "take_profit" in client_id:
                icon = "💰";
                color = "green";
                action = f"TAKE PROFIT: {symbol}"
            elif "trailing_stop" in client_id:
                icon = "🛡️";
                color = "gold";
                action = f"RATCHET STOP: {symbol}"
            elif "stop" in client_id:
                icon = "🛑";
                color = "red";
                action = f"STOP LOSS: {symbol}"
            elif "entry" in client_id:
                icon = "🚀";
                color = "blue";
                action = f"ENTRY: {symbol}"

            msg = (
                f"**{icon} ORDER FILLED**\n"
                f"**Action:** {action}\n"
                f"**Qty:** {filled_qty} @ ${float(price):.2f}\n"
                f"**Value:** ${total_value:,.2f}\n"
                f"**Type:** {order_type.upper()}"
            )

            logger.info(f"Sending Alert: {action}")
            await send_discord_alert_async(msg, color)

    except Exception as e:
        logger.error(f"Handler Error: {e}")


async def auto_disconnect():
    logger.info("Auto Disconnect Timer Started (5h 45m)")
    await asyncio.sleep(20700)
    logger.warning("⏰ Time limit reached. Performing Hard Exit...")
    os._exit(0)


async def main():
    logger.info("--- 🎧 DISCORD LISTENER ACTIVE (Non-Blocking Mode)")
    asyncio.create_task(auto_disconnect())

    while True:
        try:
            stream = Stream(API_KEY, SECRET_KEY, base_url=BASE_URL, data_feed='iex')
            stream.subscribe_trade_updates(trade_update_handler)

            logger.info("Sending Online Status to Discord...")
            await send_discord_alert_async(
                "🎧 **Discord Listener Connected & Online**\nWaiting for fills...",
                "green"
            )
            await stream.run()

        except Exception as e:
            logger.error(f"Stream Error: {e}")
            logger.info("Reconnecting...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Manual Stop Initiated....")
