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
        llm = ChatOpenAI(temperature=0.1, max_tokens=4096, model="gpt-3.5-turbo-16k")

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
        self._select_wiki_page_chain = self.create_chain(
            llm=llm,
            template_path=os.path.join(template_dir, "select_wiki_page.txt"),
            output_key="collection_name",
        )
        self._evaluate_check_chain = self.create_chain(
            llm=llm,
            template_path=os.path.join(template_dir, "evaluate_check.txt"),
            output_key="evaluate_check",
        )
        self._information_chain = self.create_chain(
            llm=llm,
            template_path=os.path.join(template_dir, "information_response.txt"),
            output_key="answer",
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
        action_count = 1
        context["job_list"] = self.read_prompt_template(self._job_list_text)
        context["action_history"] = ""
        context["chat_history"] = ""
        context["information"] = ""
        answer = ""

        while True:
            job = self._parse_job_chain.run(context)
            print(f"job: {job}")
            if job == "search_kakao_wiki" and action_count < 30:
                context["action_history"] = f'{context["action_history"]}\n{action_count}. 생각: 카카오 위키 페이지를 열람 해야겠다'
                action_count += 1
                wiki_page = self._select_wiki_page_chain.run(context)
                print(f'wiki_page: {wiki_page}')
                context["action_history"] = f'{context["action_history"]}\n{action_count}. 생각: 카카오 위키 중 {wiki_page}를 열람 해야겠다'
                action_count += 1
                context["search_result"] = self._vector_db.query_db(context["user_message"], collection_name=wiki_page)
                context["action_history"] = f'{context["action_history"]}\n{action_count}. 행동: {wiki_page}를 열람 했다'
                action_count += 1
                y_or_n = self._evaluate_check_chain.run(context)
                if y_or_n == "Y" or y_or_n == "y":
                    context["information"] = f'{context["information"]}\n카카오 위키: {wiki_page} 정보\n{context["search_result"]}'
                    context["action_history"] = f'{context["action_history"]}\n{action_count}. 판단: {wiki_page}를 정보는 사용자 질문을 대답하기에 적절하다'
                else:
                    context["action_history"] = f'{context["action_history"]}\n{action_count}. 판단: {wiki_page}를 정보는 사용자 질문을 대답하기에 적절하지 않다'
                action_count += 1
            elif job == "history" and action_count < 30:
                context["action_history"] = f'{context["action_history"]}\n{action_count}. 생각: 사용자의 이전 질문을 열람 해야겠다'
                action_count += 1
                chat_history = self.get_chat_history(conversation_id)
                context["action_history"] = f'{context["action_history"]}\n{action_count}. 행동: 사용자의 이전 질문을 열람 했다'
                action_count += 1
                context["chat_history"] = chat_history
            elif job == "search_internet" and action_count < 30:
                context["action_history"] = f'{context["action_history"]}\n{action_count}. 생각: 인터넷에서 검색 해야겠다'
                action_count += 1
                context["search_result"] = self.search_tool.run(user_message)
                context["action_history"] = f'{context["action_history"]}\n{action_count}. 행동: 인터넷에서 검색해서 자료를 획득했다'
                action_count += 1
                y_or_n = self._evaluate_check_chain.run(context)
                if y_or_n == "Y" or y_or_n == "y":
                    context["information"] = f'{context["information"]}\n인터넷 검색 자료: {wiki_page} 정보\n{context["search_result"]}'
                    context["action_history"] = f'{context["action_history"]}\n{action_count}. 판단: 인터넷 정보는 사용자 질문을 대답하기에 적절하다'
                else:
                    context["action_history"] = f'{context["action_history"]}\n{action_count}. 판단: 인터넷 정보는 사용자 질문을 대답하기에 적절하지 않다'
                action_count += 1
            else:
                answer = self._information_chain.run(context)
                break

        print(context["action_history"])
        self.log_user_message(history_file, user_message)
        self.log_bot_message(history_file, answer)
        return answer
