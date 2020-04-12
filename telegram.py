def notify_me(self, message):
        import telegram
        chat_id="912789066"
        token="1080428925:AAGlC9w7G1UHH6bhXYkJVinaG3ToDTRttic"
        bot = telegram.Bot(token=token)
        bot.send_message(chat_id=chat_id, text=message)
