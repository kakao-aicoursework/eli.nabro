import pynecone as pc

class KakaodevelopershelperbotConfig(pc.Config):
    pass

config = KakaodevelopershelperbotConfig(
    app_name="kakao_developers_helper_bot",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)