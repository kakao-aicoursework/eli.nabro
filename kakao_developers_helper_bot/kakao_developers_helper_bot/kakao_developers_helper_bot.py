import os
from datetime import datetime

import openai
from pynecone import Base
import pynecone as pc
from typing import List

openai.api_key = os.environ['OPENAI_API_KEY']


class Message(Base):
    question: str
    answer: str
    created_at: str


def call_assistant(question: str, prev_messages: List[Message]) -> str:
    system_instruction = f"당신은 유능한 어시스턴트 입니다. 모든 질문은 3줄 이내로 답변하세요."
    messages = [
        {"role": "system", "content": system_instruction}
    ]
    for prev_message in prev_messages:
        messages.append({"role": "user", "content": prev_message.question})
        messages.append({"role": "assistant", "content": prev_message.answer})

    messages.append({"role": "user", "content": question})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    answer = response['choices'][0]['message']['content']

    return answer


class State(pc.State):
    text: str = ""
    messages: list[Message] = []
    answer: str = ""

    def output(self) -> str:
        if not self.text.strip():
            return "Advise will appear here."
        answer = call_assistant(self.text, self.messages)
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

    def delete(self):
        self.messages.clear()


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
        pc.button("Post", on_click=State.post, margin_top="1rem"),
        pc.button("Delete", on_click=State.delete, margin_top="1rem", margin_left="1rem"),
        pc.vstack(
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        padding="2rem",
        max_width="600px"
    )


app = pc.App(state=State)
app.add_page(index)
app.compile()
