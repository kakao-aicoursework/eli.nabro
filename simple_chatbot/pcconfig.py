import pynecone as pc


class SimpleChatbotConfig(pc.Config):
    pass


config = SimpleChatbotConfig(
    app_name="simple_chatbot",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)
