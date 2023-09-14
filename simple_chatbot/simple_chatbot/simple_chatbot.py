"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
import os
from datetime import datetime

import openai
from pynecone import Base
from pcconfig import config

import pynecone as pc

docs_url = "https://pynecone.io/docs/getting-started/introduction"
filename = f"{config.app_name}/{config.app_name}.py"

openai.api_key = os.environ['OPENAI_API_KEY']


def call_assistant(text) -> str:
    system_instruction = f"You are a helpful assistant."
    messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": text}]

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    answer = response['choices'][0]['message']['content']

    return answer


class Message(Base):
    question: str
    answer: str
    created_at: str


class State(pc.State):
    text: str = ""
    messages: list[Message] = []
    answer: str = ""

    def output(self) -> str:
        if not self.text.strip():
            return "Advise will appear here."
        answer = call_assistant(self.text)
        self.answer = answer
        return answer

    def post(self):
        self.messages = \
            [
                Message(
                    question=self.text,
                    answer=self.output(),
                    created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
                )
            ] + self.messages


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("Simple-ChatBot 📭", font_size="2rem"),
        pc.text(
            "input and post them as messages!",
            margin_top="0.5rem",
            color="#666",
        ),
        pc.input(
            placeholder="Question",
            on_blur=State.set_text,
            margin_top="1rem",
            border_color="#eaeaef"
        ),
    )


def smallcaps(text, **kwargs):
    return pc.text(
        text,
        font_size="0.7rem",
        font_weight="bold",
        text_transform="uppercase",
        letter_spacing="0.05rem",
        **kwargs,
    )


def output():
    return pc.box(
        pc.box(
            smallcaps(
                "Output",
                color="#aeaeaf",
                background_color="white",
                padding_x="0.1rem",
            ),
            position="absolute",
            top="-0.5rem",
        ),
        pc.text(State.answer),
        padding="1rem",
        border="1px solid #eaeaef",
        margin_top="1rem",
        border_radius="8px",
        position="relative",
    )


def text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
    )


def down_arrow():
    return pc.vstack(
        pc.icon(
            tag="arrow_down",
            color="#666",
        )
    )


def message(message: Message):
    return pc.box(
        pc.vstack(
            text_box(message.question),
            down_arrow(),
            text_box(message.answer),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def index() -> pc.Component:
    return pc.fragment(
        header(),
        output(),
        pc.button("Post", on_click=State.post, margin_top="1rem"),
        pc.vstack(
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        padding="2rem",
        max_width="600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index)
app.compile()
