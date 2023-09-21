import os

from langchain import LLMChain, GoogleSearchAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.memory import FileChatMessageHistory, ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool

from kakao_developers_helper_bot.langchain_call.chroma_db_repository import ChromaDbRepository

class LangChainAssistant:
    _history_dir: str
    _job_list_text: str
    _vector_db: ChromaDbRepository
    _parse_job_chain: LLMChain

    def __init__(self, history_dir: str, template_dir: str, vector_db: ChromaDbRepository) -> None:
        self._history_dir = history_dir
        self._job_list_text = os.path.join(template_dir, "job_list.txt")
        self._vector_db = vector_db
        llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")

        search = GoogleSearchAPIWrapper(
            google_api_key=os.environ['GOOGLE_API_KEY'],
            google_cse_id=os.environ['GOOGLE_CSE_ID']
        )

        self.search_tool = Tool(
            name="Google Search",
            description="Search Google for recent results.",
            func=search.run,
        )
        self._parse_job_chain = self.create_chain(
            llm=llm,
            template_path=os.path.join(template_dir, "parse_job.txt"),
            output_key="job",
        )
        self._information_chain = self.create_chain(
            llm=llm,
            template_path=os.path.join(template_dir, "information_response.txt"),
            output_key="output",
        )
        self.search_value_check_chain = self.create_chain(
            llm=llm,
            template_path=os.path.join(template_dir, "search_value_check.txt"),
            output_key="output",
        )
        self.search_compression_chain = self.create_chain(
            llm=llm,
            template_path=os.path.join(template_dir, "search_compress.txt"),
            output_key="output",
        )

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

    def query_web_search(self, user_message: str) -> str:
        context = {"user_message": user_message, "related_web_search_results": self.search_tool.run(user_message)}

        has_value = self.search_value_check_chain.run(context)

        print(has_value)
        if has_value == "Y":
            return self.search_compression_chain.run(context)
        else:
            return ""

    def log_user_message(self, history: FileChatMessageHistory, user_message: str):
        history.add_user_message(user_message)

    def log_bot_message(self, history: FileChatMessageHistory, bot_message: str):
        history.add_ai_message(bot_message)

    def generate_answer(self, user_message, conversation_id: str = 'fa1010') -> str:
        history_file = self.load_conversation_history(conversation_id)

        context = dict(user_message=user_message)
        context["job_list"] = self.read_prompt_template(self._job_list_text)

        while True:
            job = self._parse_job_chain.run(context)
            print(f"job: {job}")
            if job == "search_kakao_wiki":
                context["related_documents"] = self._vector_db.query_db(context["user_message"])

                answer = self._information_chain.run(context)
                break
            elif job == "history":
                chat_history = self.get_chat_history(conversation_id)
                context["user_message"] = f'<chat_history>\n{chat_history}\n</chat_history>\n<question>\n{context["user_message"]}\n</question>'
            else:
                context["related_documents"] = self.query_web_search(context["user_message"])
                answer = self._information_chain.run(context)
                break

        self.log_user_message(history_file, user_message)
        self.log_bot_message(history_file, answer)
        return answer
