import os

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import FileChatMessageHistory, ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

from kakao_developers_helper_bot.langchain_call.chroma_db_repository import ChromaDbRepository


class LangChainAssistant:
    _history_dir: str
    _intent_list_text: str
    _vector_db: ChromaDbRepository
    _parse_intent_chain: LLMChain

    def __init__(self, history_dir: str, template_dir: str, vector_db: ChromaDbRepository) -> None:
        self._history_dir = history_dir
        self._intent_list_text = os.path.join(template_dir, "intent_list.txt")
        self._vector_db = vector_db
        llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")

        # self.bug_step1_chain = self.create_chain(
        #     llm=llm,
        #     template_path=BUG_STEP1_PROMPT_TEMPLATE,
        #     output_key="bug_analysis",
        # )
        # self.bug_step2_chain = create_chain(
        #     llm=llm,
        #     template_path=BUG_STEP2_PROMPT_TEMPLATE,
        #     output_key="output",
        # )
        # enhance_step1_chain = create_chain(
        #     llm=llm,
        #     template_path=ENHANCE_STEP1_PROMPT_TEMPLATE,
        #     output_key="output",
        # )
        self._parse_intent_chain = self.create_chain(
            llm=llm,
            template_path=os.path.join(template_dir, "parse_intent.txt"),
            output_key="intent",
        )
        self._information_chain = self.create_chain(
            llm=llm,
            template_path=os.path.join(template_dir, "information_response.txt"),
            output_key="output",
        )
        # default_chain = create_chain(
        #     llm=llm, template_path=DEFAULT_RESPONSE_PROMPT_TEMPLATE, output_key="output"
        # )
        #
        # search_value_check_chain = create_chain(
        #     llm=llm,
        #     template_path=SEARCH_VALUE_CHECK_PROMPT_TEMPLATE,
        #     output_key="output",
        # )
        # search_compression_chain = create_chain(
        #     llm=llm,
        #     template_path=SEARCH_COMPRESSION_PROMPT_TEMPLATE,
        #     output_key="output",
        # )

    def read_prompt_template(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            prompt_template = f.read()

        return prompt_template

    def create_chain(self, llm, template_path, output_key) -> LLMChain:
        return LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_template(
                template=self.read_prompt_template(template_path)
            ),
            output_key=output_key,
            verbose=True,
        )

    def load_conversation_history(self, conversation_id: str) -> FileChatMessageHistory:
        file_path = os.path.join(self._history_dir, f"{conversation_id}.json")
        return FileChatMessageHistory(file_path)


    def get_chat_history(self, conversation_id: str):
        history = self.load_conversation_history(conversation_id)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="user_message",
            chat_memory=history,
        )

        return memory.buffer

    def log_user_message(self, history: FileChatMessageHistory, user_message: str):
        history.add_user_message(user_message)

    def log_bot_message(self, history: FileChatMessageHistory, bot_message: str):
        history.add_ai_message(bot_message)

    def generate_answer(self, user_message, conversation_id: str='fa1010') -> str:
        history_file = self.load_conversation_history(conversation_id)

        context = dict(user_message=user_message)
        context["intent_list"] = self.read_prompt_template(self._intent_list_text)

        while True:
            intent = self._parse_intent_chain.run(context)

            if intent == "information":
                context["related_documents"] = self._vector_db.query_db(context["user_message"])

                answer = self._information_chain.run(context)
                break
            elif intent == "history":
                context["chat_history"] = self.get_chat_history(conversation_id)
                context["user_message"] = f'{context["chat_history"]}\n{context["user_message"]}'
            else:
                answer = self._information_chain.run(context)
                break

        self.log_user_message(history_file, user_message)
        self.log_bot_message(history_file, answer)
        return answer
