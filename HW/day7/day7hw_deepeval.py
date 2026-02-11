import pandas as pd
import os
import ast
import json
import re
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric  # ✅ 新增這個
)

# ================= 配置區域 =================
LLM_BASE_URL = "https://ws-03.wade0426.me/v1"
LLM_API_KEY = "day6hwdeepeval"
LLM_MODEL_NAME = "/models/Qwen3-30B-A3B-Instruct-2507-FP8"

INPUT_FILE_PREDICT = "HW/output.csv"
INPUT_FILE_GT = "HW/ground_truth.csv"
OUTPUT_FILE = "HW/output_with_5_metrics.csv"  # 改名為 5 metrics

COL_QUESTION = "questions"
COL_ANSWER = "answer"
COL_CONTEXTS = "contexts"
COL_GROUND_TRUTH = "ground_truth"


class Gemma3Wrapper(DeepEvalBaseLLM):
    def __init__(self, model_name=LLM_MODEL_NAME):
        self.model_name = model_name
        self.model = ChatOpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            model=model_name,
            temperature=0.1,
            max_retries=3,
            request_timeout=120
        )

    def load_model(self):
        return self.model

    def _clean_output(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"```[a-zA-Z]*", "", text)
        text = text.replace("```", "")

        first_brace = text.find("{")
        first_bracket = text.find("[")
        start_idx = -1
        char_pair = ""

        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            start_idx = first_brace
            char_pair = "}"
        elif first_bracket != -1:
            start_idx = first_bracket
            char_pair = "]"

        if start_idx != -1:
            end_idx = text.rfind(char_pair)
            if end_idx != -1 and end_idx > start_idx:
                return text[start_idx: end_idx + 1]
        return text

    def generate(self, prompt: str, schema=None) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                messages = [
                    SystemMessage(content="You are a strict JSON generator. Output ONLY valid JSON."),
                    HumanMessage(content=prompt)
                ]
                response = self.model.invoke(messages)
                content = self._clean_output(response.content)

                try:
                    json.loads(content)
                    return content
                except json.JSONDecodeError:
                    fixed_content = content.replace("'", '"').replace("True", "true").replace("False", "false")
                    json.loads(fixed_content)
                    return fixed_content

            except Exception as e:
                print(f"   ⚠️ [JSON Error] 重試 {attempt + 1}/{max_retries}: {e}")
                time.sleep(1)

        print("   ❌ [Failure] 無法生成有效 JSON，回傳空物件。")
        return "{}"

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


def main():
    if not os.path.exists(INPUT_FILE_PREDICT) or not os.path.exists(INPUT_FILE_GT):
        print(f"❌ 錯誤: 找不到輸入檔案")
        return

    print(f"📂 讀取檔案中...")
    df_pred = pd.read_csv(INPUT_FILE_PREDICT)
    df_gt = pd.read_csv(INPUT_FILE_GT)

    # 合併資料
    if COL_GROUND_TRUTH in df_gt.columns:
        df_pred[COL_GROUND_TRUTH] = df_gt[COL_GROUND_TRUTH]
    if COL_CONTEXTS in df_gt.columns:
        df_pred[COL_CONTEXTS] = df_gt[COL_CONTEXTS]

    # 【測試模式】只取前 5 筆 (若要跑全部請註解掉這行)
    # df = df_pred.head(5)
    df = df_pred
    print(f"📊 預計評估 {len(df)} 筆資料 (共 5 項指標)...")

    try:
        custom_llm = Gemma3Wrapper()
    except Exception as e:
        print(f"❌ 模型初始化失敗: {e}")
        return

    # ✅ 定義 5 大指標
    metrics_to_run = [
        FaithfulnessMetric(threshold=0.5, model=custom_llm, include_reason=True, strict_mode=False),
        AnswerRelevancyMetric(threshold=0.5, model=custom_llm, include_reason=True, strict_mode=False),
        ContextualRecallMetric(threshold=0.5, model=custom_llm, include_reason=True, strict_mode=False),
        ContextualPrecisionMetric(threshold=0.5, model=custom_llm, include_reason=True, strict_mode=False),
        ContextualRelevancyMetric(threshold=0.5, model=custom_llm, include_reason=True, strict_mode=False)  # ✅ 新增
    ]

    results_data = []

    print("🚀 開始 DeepEval 評估 (5 Metrics)...")

    for index, row in df.iterrows():
        print(f"\n--- 正在處理第 {index + 1} / {len(df)} 題 ---")
        row_data = row.to_dict()

        # 處理 Context
        raw_context = row.get(COL_CONTEXTS, [])
        retrieval_context = []
        if isinstance(raw_context, str):
            try:
                retrieval_context = ast.literal_eval(raw_context)
            except:
                retrieval_context = [raw_context]
        elif isinstance(raw_context, list):
            retrieval_context = raw_context

        retrieval_context = [str(c) for c in retrieval_context if str(c).strip()]

        # 處理 Ground Truth
        ground_truth = row.get(COL_GROUND_TRUTH, None)
        expected_output = str(ground_truth) if pd.notna(ground_truth) else ""

        if not expected_output.strip() or expected_output.lower() == "nan":
            print("   ⚠️ Ground Truth 為空，暫用 Context 替代...")
            expected_output = "\n".join(retrieval_context)

        test_case = LLMTestCase(
            input=str(row[COL_QUESTION]),
            actual_output=str(row[COL_ANSWER]),
            retrieval_context=retrieval_context,
            expected_output=expected_output
        )

        for metric in metrics_to_run:
            metric_name = metric.__class__.__name__

            if ("Recall" in metric_name or "Precision" in metric_name) and not expected_output:
                row_data[f"{metric_name}_Score"] = -1
                row_data[f"{metric_name}_Reason"] = "No Ground Truth"
                continue

            try:
                metric.measure(test_case)
                print(f"   > {metric_name}: {metric.score:.2f}")

                row_data[f"{metric_name}_Score"] = metric.score
                row_data[f"{metric_name}_Reason"] = metric.reason

                time.sleep(1)

            except Exception as e:
                print(f"   ! {metric_name} Error: {e}")
                row_data[f"{metric_name}_Score"] = -1
                row_data[f"{metric_name}_Reason"] = "Error"

        results_data.append(row_data)

        # 每一題存一次，避免當機全滅
        if (index + 1) % 1 == 0:
            pd.DataFrame(results_data).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

        print("   (Cooling down 3s...)")
        time.sleep(3)

    final_df = pd.DataFrame(results_data)
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ 評估完成！結果已儲存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()