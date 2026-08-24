"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
from langchain_core.tracers import context
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
# TODO: Đổi thành tên của bạn — phải là duy nhất trong Hub của bạn
PROMPT_V1_NAME = "cuong-rag-prompt-v1"   # ví dụ: "nguyen-rag-v1"
PROMPT_V2_NAME = "cuong-rag-prompt-v2"   # ví dụ: "nguyen-rag-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# TODO: Viết SYSTEM_V1 — phong cách ngắn gọn, trả lời 2-4 câu
# Gợi ý: "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời.
#          Giữ câu trả lời ngắn gọn (2-4 câu). ..."
SYSTEM_V1 = (
    "Bạn là một trợ lý AI hữu ích và chính xác.\n"
    "Chỉ sử dụng thông tin có trong context được cung cấp để trả lời câu hỏi.\n"
    "Nếu context không chứa đủ thông tin để trả lời, hãy nói rằng bạn không biết "
    "thay vì tự suy đoán hoặc sử dụng kiến thức bên ngoài.\n"
    "Hãy giữ câu trả lời ngắn gọn, súc tích (2-4 câu).\n"
    "Không lặp lại thông tin không cần thiết, đi thẳng vào vấn đề.\n"
)
#          Giữ câu trả lời ngắn gọn (2-4 câu)."

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# TODO: Viết SYSTEM_V2 — phong cách có cấu trúc, expert tone, 3-5 câu
# Gợi ý: "Bạn là chuyên gia AI. Đọc kỹ context, xác định facts liên quan,
#          viết câu trả lời rõ ràng và có tổ chức (3-5 câu). ..."
SYSTEM_V2 = ("Bạn là chuyên gia có 10 năm kinh nghiệm trong lĩnh vực của bạn."
            "Đọc kỹ context, xác định facts liên quan, viết câu trả lời rõ ràng và có tổ chức."
            "Nếu câu trả lời cho câu hỏi chỉ là xác suất thì phải thêm chú thích \n "
            "'Đây chỉ là lời khuyên cho bạn tham khảo, không có giá trị thay cho quyết định của bạn'.\n"
            "Luôn kiểm chứng thông tin trước khi trả lời, tra toàn bộ tài liệu. Nếu như tài liệu cũ "
            "thì phải xem xét có tài liệu mới chưa. Nếu như có mà chưa được cập nhật thì không được trả lời"
            "không được bịa đặt thông tin. Nếu như người dùng hỏi ngoài lĩnh vực của bạn, trả lời:"
            "'Câu hỏi của bạn nằm ngoài phạm vi lĩnh vực của tôi, nên tôi không có câu trả lời cho vấn đề này'"
            "Không lặp lại thông tin không cần thiết, đi thẳng vào vấn đề.\n"
            "Viết câu trả lời rõ ràng trong 3 -5 câu."

)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    Gợi ý: client.push_prompt(name, object=template, description="...")
    """
    # TODO: Push PROMPT_V1 — bọc trong try/except để xử lý lỗi
    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – ngắn gọn")   # client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – ngắn gọn")
        print(f"✅ Đã push V1 → {url}")
    except Exception as e:
        print(f"⚠️  V1 lỗi: {e}")

    # TODO: Push PROMPT_V2 — bọc trong try/except
    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – có cấu trúc")   # client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – có cấu trúc")
        print(f"✅ Đã push V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2 lỗi: {e}")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Fallback về template local nếu Hub không khả dụng.

    Gợi ý: client.pull_prompt(name) → ChatPromptTemplate

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    # TODO: Pull PROMPT_V1_NAME, fallback về PROMPT_V1 nếu lỗi
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V1_NAME}'")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V2_NAME}'")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version}


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=1000, chunk_overlap=100)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    print("📤 Đang đẩy prompts lên LangSmith Hub...")
    push_prompts_to_hub(client)

    print("📥 Đang kéo prompts từ LangSmith Hub...")
    prompts = pull_prompts_from_hub(client)

    print("📚 Đang tạo Vectorstore...")
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    v1_count, v2_count = 0, 0
    import time
    print("\n🔀 Đang xử lý câu hỏi qua A/B Router...")
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"

        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, question, version_tag)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}] [{request_id}] [prompt-{version_tag}] {question[:55]}...")
        time.sleep(4)

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
