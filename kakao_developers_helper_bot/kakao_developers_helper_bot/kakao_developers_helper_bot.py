import os
from datetime import datetime

import openai
from pynecone import Base
import pynecone as pc
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from pprint import pprint

openai.api_key = os.environ['OPENAI_API_KEY']
with open('assets/kakaosink.txt', 'r', encoding='utf-8') as file:
    kakao_sink_contents = file.read()

kakao_sink_path = os.path.join(os.getcwd(), "assets/kakaosink.txt")


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

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=messages)
    answer = response['choices'][0]['message']['content']

    return answer


functions = [{
    "name": "kakao_sink_information",
    "description": "kakao 의 신규 서비스 카카오싱크 ( kakaosink ) 에 대한 전반적인 정보를 가져옵니다. 이 정보는 기능, 과정, 도입안내를 포함합니",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}]


def function_call_assistant(question: str, prev_messages: List[Message]):
    system_instruction = f"당신은 유능한 어시스턴트 입니다."
    messages = [
        {"role": "system", "content": system_instruction}
    ]
    for prev_message in prev_messages:
        messages.append({"role": "user", "content": prev_message.question})
        messages.append({"role": "assistant", "content": prev_message.answer})

    messages.append({"role": "user", "content": question})

    for i in range(0, 3):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            functions=functions,
            function_call="auto",
            max_tokens=8192
        )

        if "function_call" in completion['choices'][0]['message']:
            function_name = completion["choices"][0]["message"]["function_call"]["name"]
            print(f"call {function_name}")
            messages.append({"role": "assistant", "content": f"call {function_name}"})
            messages.append({"role": "user",
                             "content": f"kakao_sink_information 함수의 결과입니다 \n\n {kakao_sink_contents} \n\n {question}"})
        else:
            return completion['choices'][0]['message']['content']

    return "반복 function 호출로 결과 load에 실패하였습니다"


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template


def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path),
        ),
        output_key=output_key,
        verbose=True,
    )


def lang_chain_call_assistant(question: str, prev_messages: List[Message]):
    assistant_llm = ChatOpenAI(temperature=0.1, max_tokens=8192, model="gpt-3.5-turbo-16k")
    kakao_sink_chain = create_chain(assistant_llm, kakao_sink_path, "kakao_sink")
    question_chain = LLMChain(
        llm=assistant_llm,
        prompt=ChatPromptTemplate.from_template(
            template=f"{question}",
        ),
        output_key="result",
        verbose=True,
    )

    preprocess_chain = SequentialChain(
        chains=[
            kakao_sink_chain,
        ],
        input_variables=[],
        output_variables=["kakao_sink"],
        verbose=True,
    )

    context = dict()
    context = preprocess_chain(context)

    context["answer"] = []
    context = question_chain(context)
    context["answer"].append(context["result"])

    contents = "\n\n".join(context["answer"])
    return contents


class State(pc.State):
    text: str = ""
    messages: list[Message] = []
    answer: str = ""

    def output(self, func: str) -> str:
        print(f"output {func}")
        if not self.text.strip():
            return "Advise will appear here."
        if func == "simple":
            self.answer = call_assistant(self.text, self.messages)
        elif func == "function_call":
            self.answer = function_call_assistant(self.text, self.messages)
        elif func == "lang_chain_call":
            self.answer = lang_chain_call_assistant(self.text, self.messages)

        return self.answer

    def post(self):
        self.messages = \
            [
                Message(
                    question=self.text,
                    answer=self.output("simple"),
                    created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
                )
            ] + self.messages

    def function_call_post(self):
        self.messages = \
            [
                Message(
                    question=self.text,
                    answer=self.output("function_call"),
                    created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
                )
            ] + self.messages

    def lang_chain_call(self):
        self.messages = \
            [
                Message(
                    question=self.text,
                    answer=self.output("lang_chain_call"),
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
        pc.button("Simple Post", on_click=State.post, margin_top="1rem"),
        pc.button("Function Call Post", on_click=State.function_call_post, margin_top="1rem", margin_left="1rem"),
        pc.button("LangChain Call Post", on_click=State.lang_chain_call, margin_top="1rem", margin_left="1rem"),
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
