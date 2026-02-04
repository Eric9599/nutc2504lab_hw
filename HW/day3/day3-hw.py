import re
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

model = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="day3hw",
    model="Qwen/Qwen3-VL-8B-Instruct",
    temperature=1.5
)


# --- 2. 定義資料結構 (State) ---
class GraphState(TypedDict):
    raw_segments: List[dict]  # 存放 SRT 解析後的字典列表
    full_text: str  # 存放純文字內容
    transcript_result: str  # 存放整理好的逐字稿 (注意：名稱要跟下面函式 return 的 key 一樣)
    summary_result: str  # 存放 AI 摘要結果


# --- 3. 輔助函式 (解析 SRT) ---
def parse_srt_file(file_path):
    """最精簡的 SRT 解析器，直接讀取並回傳 List"""
    segments = []

    # 直接開檔讀取 (假設檔案一定存在且編碼正確)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 使用 Regex 抓取：時間軸 --> 時間軸 內容
    pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+(.*?)(?=\n\n|\Z)', re.DOTALL)
    matches = pattern.findall(content)

    for start, end, text in matches:
        segments.append({
            "start": start,
            "end": end,
            "text": text.replace('\n', ' ').strip()
        })

    return segments


# --- 4. 定義節點 (Nodes) ---

def asr_node(state: GraphState):
    print("ASR Node 讀取檔案中...")

    srt_path = "out/133.srt"
    txt_path = "out/133.txt"

    segments = parse_srt_file(srt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # 回傳給 State
    return {"raw_segments": segments, "full_text": full_text}


def transcript_node(state: GraphState):
    print("Transcript Node")

    segments = state["raw_segments"]
    output_str = ""

    for seg in segments:
        output_str += f"[{seg['start']} -> {seg['end']}] {seg['text']}\n"

    # 回傳 Key 必須對應 State 定義的 transcript_result
    return {"transcript_result": output_str}


def summary_node(state: GraphState):
    print("Summary Node")

    text = state["full_text"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個專業的會議記錄員。"),
        ("human", "請將以下內容整理成關鍵摘要：\n\n{content}")
    ])

    chain = prompt | model | StrOutputParser()
    result = chain.invoke({"content": text})

    return {"summary_result": result}


# --- 5. 建立圖形 (Graph) ---

workflow = StateGraph(GraphState)

# 加入節點
workflow.add_node("asr_agent", asr_node)
workflow.add_node("transcript_writer", transcript_node)
workflow.add_node("summarizer", summary_node)

# 設定流程
workflow.set_entry_point("asr_agent")
workflow.add_edge("asr_agent", "transcript_writer")  # 分流 1
workflow.add_edge("asr_agent", "summarizer")  # 分流 2
workflow.add_edge("transcript_writer", END)
workflow.add_edge("summarizer", END)

# 編譯
app = workflow.compile()

print("開始執行...\n")

final_state = app.invoke({})

print("-" * 30)
print("【逐字稿預覽】")
print(final_state["transcript_result"][:2000])
print("-" * 30)
print("【重點摘要】")
print(final_state["summary_result"])